
import os
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.backend as K

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    print(f"[LightEA] TensorFlow GPU: {len(gpus)} GPUs")
else:
    print("[LightEA] TensorFlow: CPU mode")


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    try:
        return tensor.numpy()
    except:
        return K.eval(tensor)

HAS_FAISS = False
HAS_FAISS_GPU = False
_faiss_gpu_res = None

try:
    import faiss
    HAS_FAISS = True
    _want_gpu = os.environ.get("LIGHTEA_FAISS_GPU", "0") == "1"
    if _want_gpu and hasattr(faiss, "StandardGpuResources"):
        HAS_FAISS_GPU = True
        print("[LightEA] FAISS: GPU mode (lazy init, auto CPU fallback on OOM)")
    else:
        HAS_FAISS_GPU = False
        if _want_gpu:
            print("[LightEA] FAISS: CPU mode (faiss-gpu not available)")
        else:
            print("[LightEA] FAISS: CPU mode (set LIGHTEA_FAISS_GPU=1 to try GPU)")
except ImportError:
    print("[LightEA] Warning: faiss not installed")


def _get_faiss_gpu_res():
    global _faiss_gpu_res
    if HAS_FAISS_GPU and _faiss_gpu_res is None:
        _faiss_gpu_res = faiss.StandardGpuResources()
    return _faiss_gpu_res


LARGE_GRAPH_THRESHOLD = int(os.environ.get("LIGHTEA_LARGE_GRAPH_THRESHOLD", 500000))
FAISS_GPU_LIMIT_GB = float(os.environ.get("LIGHTEA_FAISS_GPU_LIMIT_GB", "25"))
BATCH_SIZE = int(os.environ.get("LIGHTEA_BATCH_SIZE", "128"))


def _scan_max_ent_id_from_alignment(path):
    max_id = -1
    candidates = [
        'sup_ent_ids', 'ref_ent_ids', 'sup_pairs', 'ref_pairs',
        'test_pairs', 'train_pairs', 'valid_pairs', 'all_pairs',
    ]
    for fname in candidates:
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath) as f:
                for line in f:
                    for tok in line.strip().split():
                        try:
                            v = int(tok)
                            if v > max_id:
                                max_id = v
                        except ValueError:
                            pass
        except Exception:
            pass
    return max_id


def load_graph(path):
    if not path.endswith('/'):
        path = path + '/'
    
    if os.path.exists(path + "graph_cache.pkl"):
        cached = pickle.load(open(path + "graph_cache.pkl", "rb"))

        cached_node_size = cached[0]
        max_align = _scan_max_ent_id_from_alignment(path)
        needed = max_align + 1
        if needed > cached_node_size:
            print(f"[LightEA] Cached node_size {cached_node_size} too small, "
                  f"extending to {needed} (alignment file has entity id {max_align})")
            cached = list(cached)
            cached[0] = needed
            with open(path + "graph_cache.pkl", "wb") as f:
                pickle.dump(cached, f)
        return cached
    
    triples = []
    with open(path + "triples_1") as f:
        for line in f.readlines():
            h, r, t = [int(x) for x in line.strip().split("\t")]
            triples.append([h, t, 2*r])
            triples.append([t, h, 2*r+1])
    with open(path + "triples_2") as f:
        for line in f.readlines():
            h, r, t = [int(x) for x in line.strip().split("\t")]
            triples.append([h, t, 2*r])
            triples.append([t, h, 2*r+1])
    
    triples = np.unique(triples, axis=0)
    node_size, rel_size = np.max(triples) + 1, np.max(triples[:, 2]) + 1
    
    ent_tuple, triples_idx = [], []
    ent_ent_s, rel_ent_s, ent_rel_s = {}, set(), set()
    last, index = (-1, -1), -1
    
    for i in range(node_size):
        ent_ent_s[(i, i)] = 0
    
    for h, t, r in triples:
        ent_ent_s[(h, h)] += 1
        ent_ent_s[(t, t)] += 1
        
        if (h, t) != last:
            last = (h, t)
            index += 1
            ent_tuple.append([h, t])
            ent_ent_s[(h, t)] = 0
        
        triples_idx.append([index, r])
        ent_ent_s[(h, t)] += 1
        rel_ent_s.add((r, h))
        ent_rel_s.add((t, r))
    
    ent_tuple = np.array(ent_tuple)
    triples_idx = np.unique(np.array(triples_idx), axis=0)
    
    ent_ent = np.unique(np.array(list(ent_ent_s.keys())), axis=0)
    ent_ent_val = np.array([ent_ent_s[(x, y)] for x, y in ent_ent]).astype("float32")
    rel_ent = np.unique(np.array(list(rel_ent_s)), axis=0)
    ent_rel = np.unique(np.array(list(ent_rel_s)), axis=0)

    max_align = _scan_max_ent_id_from_alignment(path)
    if max_align + 1 > node_size:
        print(f"[LightEA] Extending node_size from {node_size} to {max_align + 1} "
              f"(alignment file has entity id {max_align})")
        node_size = int(max_align + 1)
    
    graph_data = [node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel]
    pickle.dump(graph_data, open(path + "graph_cache.pkl", "wb"))
    return graph_data


def random_projection(x, out_dim):
    random_vec = K.l2_normalize(tf.random.normal((x.shape[-1], out_dim)), axis=-1)
    return K.dot(x, random_vec)


def batch_sparse_matmul(sparse_tensor, dense_tensor, batch_size=None, save_mem=False):
    if batch_size is None:
        batch_size = BATCH_SIZE
    results = []
    for i in range(dense_tensor.shape[-1] // batch_size + 1):
        temp_result = tf.sparse.sparse_dense_matmul(
            sparse_tensor, 
            dense_tensor[:, i * batch_size:(i + 1) * batch_size]
        )
        if save_mem:
            temp_result = to_numpy(temp_result)
        results.append(temp_result)
    if save_mem:
        return np.concatenate(results, -1)
    else:
        return K.concatenate(results, -1)


def _merge_topk(sims_a, idx_a, sims_b, idx_b, top_k):
    combined_sims = np.concatenate([sims_a, sims_b], axis=1)
    combined_idx = np.concatenate([idx_a, idx_b], axis=1)
    
    part = np.argpartition(-combined_sims, top_k - 1, axis=1)[:, :top_k]
    part_sims = np.take_along_axis(combined_sims, part, axis=1)
    sort_order = np.argsort(-part_sims, axis=1)
    final_cols = np.take_along_axis(part, sort_order, axis=1)
    
    final_sims = np.take_along_axis(combined_sims, final_cols, axis=1)
    final_idx = np.take_along_axis(combined_idx, final_cols, axis=1)
    return final_sims, final_idx


def _attempt_chunked_search(features_l, features_r, top_k, gpu_res, dim, chunk_rows):
    n_r = features_r.shape[0]
    n_chunks = (n_r + chunk_rows - 1) // chunk_rows
    
    all_sims = None
    all_idx = None
    
    for ci in range(n_chunks):
        start = ci * chunk_rows
        end = min(start + chunk_rows, n_r)
        chunk = np.ascontiguousarray(features_r[start:end])
        
        try:
            idx = faiss.IndexFlatIP(dim)
            idx = faiss.index_cpu_to_gpu(gpu_res, 0, idx)
            idx.add(chunk)
            c_sims, c_idx = idx.search(features_l, top_k)
            c_idx = c_idx.astype(np.int64) + start
            del idx
        except RuntimeError as e:
            msg = str(e).lower()
            if 'memory' in msg or 'alloc' in msg or 'oom' in msg:
                return None
            else:
                raise
        
        if all_sims is None:
            all_sims = c_sims
            all_idx = c_idx
        else:
            all_sims, all_idx = _merge_topk(all_sims, all_idx, c_sims, c_idx, top_k)
        
        if (ci + 1) % 5 == 0 or ci + 1 == n_chunks:
            print(f"  [chunk {ci+1}/{n_chunks}] done (rows {start:,}..{end:,})")
    
    return all_sims, all_idx


def _chunked_gpu_faiss_search(features_l, features_r, top_k, gpu_res, dim):
    n_l, n_r = features_l.shape[0], features_r.shape[0]
    features_l_gb = n_l * dim * 4 / (1024**3)
    
    avail_chunk_gb = max((FAISS_GPU_LIMIT_GB - features_l_gb) / 3.0, 0.5)
    chunk_rows = int(avail_chunk_gb * 1024**3 / (dim * 4))
    chunk_rows = max(chunk_rows, 20000)
    chunk_rows = min(chunk_rows, n_r)
    
    min_chunk_rows = 8000
    
    for attempt in range(5):
        if chunk_rows < min_chunk_rows:
            print(f"[LightEA] Chunks too small ({chunk_rows} < {min_chunk_rows}), "
                  f"giving up chunked GPU path")
            return None
        
        chunk_gb = chunk_rows * dim * 4 / (1024**3)
        n_chunks = (n_r + chunk_rows - 1) // chunk_rows
        
        if attempt == 0:
            print(f"[LightEA] Chunked GPU search: {n_chunks} chunks × {chunk_rows:,} rows "
                  f"(features_l={features_l_gb:.1f} GB, each chunk={chunk_gb:.2f} GB)")
        else:
            print(f"[LightEA] Retry {attempt}: halving to {n_chunks} chunks × "
                  f"{chunk_rows:,} rows (each chunk={chunk_gb:.2f} GB)")
        
        result = _attempt_chunked_search(features_l, features_r, top_k, gpu_res, dim, chunk_rows)
        if result is not None:
            return result
        
        chunk_rows //= 2
    
    return None


def _faiss_search(features_l, features_r, dim, param, mode, top_k):
    measure = faiss.METRIC_INNER_PRODUCT
    n_l, d = features_l.shape
    n_r = features_r.shape[0]

    features_l_gb = n_l * d * 4 / (1024**3)
    features_r_gb = n_r * d * 4 / (1024**3)
    est_total_gpu_gb = (features_l_gb + features_r_gb) * 1.5
    
    def _cpu_path(reason=""):
        if reason:
            print(f"[LightEA] Using FAISS CPU: {reason}")
        idx = faiss.index_factory(dim, param, measure)
        if mode != "test":
            idx.nprobe = 16
        idx.train(features_r)
        idx.add(features_r)
        return idx.search(features_l, top_k)
    
    if not HAS_FAISS_GPU:
        return _cpu_path()
    
    _res = _get_faiss_gpu_res()
    if _res is None:
        return _cpu_path()
    
    if est_total_gpu_gb <= FAISS_GPU_LIMIT_GB:
        try:
            idx = faiss.index_factory(dim, param, measure)
            if mode != "test":
                idx.nprobe = 16
            idx = faiss.index_cpu_to_gpu(_res, 0, idx)
            idx.train(features_r)
            idx.add(features_r)
            sims, indices = idx.search(features_l, top_k)
            del idx
            return sims, indices
        except RuntimeError as e:
            msg = str(e).lower()
            if 'memory' in msg or 'alloc' in msg or 'oom' in msg:
                print(f"[LightEA] Single-shot FAISS-GPU OOM "
                      f"(features_l={features_l_gb:.1f} GB + features_r={features_r_gb:.1f} GB "
                      f"+ workspace), trying chunked GPU search...")
            else:
                raise
    else:
        print(f"[LightEA] Estimated GPU need ~{est_total_gpu_gb:.1f} GB > "
              f"LIGHTEA_FAISS_GPU_LIMIT_GB={FAISS_GPU_LIMIT_GB:.0f} GB "
              f"(features_l={features_l_gb:.1f} + features_r={features_r_gb:.1f}), "
              f"using chunked GPU search...")
    
    if mode == "test" and param == 'Flat':
        result = _chunked_gpu_faiss_search(features_l, features_r, top_k, _res, d)
        if result is not None:
            return result
    
    return _cpu_path(reason=f"GPU paths exhausted, using CPU "
                            f"(features_r={features_r_gb:.1f} GB, may take 30min-2hr)")


def sparse_sinkhorn_sims(left, right, features, top_k=500, iteration=15, mode="test"):
    features_l = features[left].copy()
    features_r = features[right].copy()
    
    faiss.normalize_L2(features_l)
    faiss.normalize_L2(features_r)
    
    dim = features_l.shape[1]
    if mode == "test":
        param = 'Flat'
    else:
        param = 'IVF256(RCQ2x5),PQ32'

    sims, index = _faiss_search(features_l, features_r, dim, param, mode, top_k)
    
    row_sims = K.exp(sims.flatten() / 0.02)
    index = K.flatten(index.astype("int32"))
    
    size = len(left)
    row_index = K.transpose([K.arange(size * top_k) // top_k, index, K.arange(size * top_k)])
    col_index = tf.gather(row_index, tf.argsort(row_index[:, 1]))
    covert_idx = tf.argsort(col_index[:, 2])
    
    for _ in range(iteration):
        row_sims = row_sims / tf.gather(
            indices=row_index[:, 0],
            params=tf.math.segment_sum(row_sims, row_index[:, 0])
        )
        col_sims = tf.gather(row_sims, col_index[:, 2])
        col_sims = col_sims / tf.gather(
            indices=col_index[:, 1],
            params=tf.math.segment_sum(col_sims, col_index[:, 1])
        )
        row_sims = tf.gather(col_sims, covert_idx)
    
    return K.reshape(row_index[:, 1], (-1, top_k)), K.reshape(row_sims, (-1, top_k))


def test(test_pair, features, top_k=500, iteration=15):
    left, right = test_pair[:, 0], np.unique(test_pair[:, 1])
    index, sims = sparse_sinkhorn_sims(left, right, features, top_k, iteration, "test")
    ranks = to_numpy(tf.argsort(-sims, -1))
    index = to_numpy(index)
    
    wrong_list, right_list = [], []
    h1, h10, mrr = 0, 0, 0
    pos = np.zeros(np.max(right) + 1)
    pos[right] = np.arange(len(right))
    
    for i in range(len(test_pair)):
        rank = np.where(pos[test_pair[i, 1]] == index[i, ranks[i]])[0]
        if len(rank) != 0:
            if rank[0] == 0:
                h1 += 1
                right_list.append(test_pair[i])
            else:
                wrong_list.append((test_pair[i], right[index[i, ranks[i]][0]]))
            if rank[0] < 10:
                h10 += 1
            mrr += 1 / (rank[0] + 1)
    
    print("Hits@1: %.4f Hits@10: %.4f MRR: %.4f" % (
        h1 / len(test_pair), h10 / len(test_pair), mrr / len(test_pair)
    ))
    
    return h1 / len(test_pair), h10 / len(test_pair), mrr / len(test_pair)


def _l2_normalize_np_inplace(arr):
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    np.maximum(norm, 1e-12, out=norm)
    arr /= norm
    return arr


class LightEAWrapper:
    
    def __init__(self, data_path, ent_dim=1024, rel_dim=None, depth=2, top_k=500, seed=12306):
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.depth = depth
        self.top_k = top_k
        self.data_path = data_path if data_path.endswith('/') else data_path + '/'
        
        graph_data = load_graph(self.data_path)
        (self.node_size, self.rel_size, self.ent_tuple, self.triples_idx,
         self.ent_ent, self.ent_ent_val, self.rel_ent, self.ent_rel) = graph_data
        
        original_ent_dim = ent_dim
        ent_dim_auto_downsized = False

        env_ent_dim = os.environ.get("LIGHTEA_ENT_DIM")
        if env_ent_dim:
            ent_dim = int(env_ent_dim)
            print(f"[LightEA] ent_dim overridden by LIGHTEA_ENT_DIM env to {ent_dim}")
        else:
            INT32_MAX = 2_147_483_647
            max_safe = INT32_MAX // max(self.node_size, 1)
            if ent_dim > max_safe:
                for c in [512, 256, 128, 64]:
                    if c <= max_safe:
                        print(f"[LightEA] Auto-downsizing ent_dim {ent_dim} -> {c} "
                              f"(node_size={self.node_size}; {ent_dim} would overflow "
                              f"TF int32 kernel: {self.node_size}*{ent_dim}="
                              f"{self.node_size * ent_dim} > {INT32_MAX})")
                        ent_dim = c
                        ent_dim_auto_downsized = True
                        break
        
        self.ent_dim = ent_dim

        env_rel_dim = os.environ.get("LIGHTEA_REL_DIM")
        if env_rel_dim:
            rel_dim = int(env_rel_dim)
            print(f"[LightEA] rel_dim overridden by LIGHTEA_REL_DIM env to {rel_dim}")
            self.rel_dim = rel_dim
        elif rel_dim is not None:
            if ent_dim_auto_downsized:
                scale_factor = ent_dim / original_ent_dim
                new_rel_dim = max(int(rel_dim * scale_factor), 64)
                if new_rel_dim != rel_dim:
                    print(f"[LightEA] Auto-scaling rel_dim {rel_dim} -> {new_rel_dim} "
                          f"(preserving ent_dim:rel_dim ratio after ent_dim downsize; "
                          f"features will be {new_rel_dim}*16={new_rel_dim*16}-dim instead of "
                          f"{rel_dim}*16={rel_dim*16}-dim)")
                    rel_dim = new_rel_dim
            self.rel_dim = rel_dim
        else:

            if self.node_size >= 60000:
                self.rel_dim = ent_dim // 3
                print(f"[LightEA] Large dataset detected (node_size={self.node_size} >= 60K), "
                      f"using rel_dim = ent_dim // 3 = {self.rel_dim} (aligned with paper)")
            else:
                self.rel_dim = ent_dim // 2
                print(f"[LightEA] Small dataset (node_size={self.node_size} < 60K), "
                      f"using rel_dim = ent_dim // 2 = {self.rel_dim} (aligned with paper)")
        
        self.mini_dim = 16
        
        self.features = None
        self.train_pair = None
        
        print(f"[LightEA] Loaded graph: {self.node_size} nodes, {self.rel_size} relations, "
              f"ent_dim={self.ent_dim}, rel_dim={self.rel_dim}, "
              f"final_feature_dim={self.rel_dim * self.mini_dim}")

        try:
            _warmup_a = tf.random.normal((128, 128))
            _warmup_b = tf.random.normal((128, 128))
            _warmup_c = tf.matmul(_warmup_a, _warmup_b)
            _ = to_numpy(_warmup_c)
            _warmup_d = tf.nn.l2_normalize(_warmup_c, axis=-1)
            _ = to_numpy(_warmup_d)
            del _warmup_a, _warmup_b, _warmup_c, _warmup_d
            print("[LightEA] cuBLAS/cuDNN warmup: OK")
        except Exception as e:
            print(f"[LightEA] cuBLAS warmup failed (non-fatal): {e}")
    
    def get_features(self, train_pair, extra_feature=None):
        ent_dim = self.ent_dim
        node_size = self.node_size
        rel_size = self.rel_size
        rel_dim = self.rel_dim
        mini_dim = self.mini_dim

        use_low_mem = node_size > LARGE_GRAPH_THRESHOLD
        if use_low_mem:
            print(f"[LightEA] Large graph detected ({node_size} > {LARGE_GRAPH_THRESHOLD}), "
                  f"using low-memory path (CPU concat/normalize + save_mem)")
        
        if extra_feature is not None:
            ent_feature = extra_feature
        else:
            if use_low_mem:

                np.random.seed(self.seed)
                random_vec_np = np.random.randn(len(train_pair), ent_dim).astype(np.float32)
                nrm = np.linalg.norm(random_vec_np, axis=-1, keepdims=True)
                random_vec_np /= np.maximum(nrm, 1e-12)
                expanded_np = np.repeat(random_vec_np, 2, axis=0)
                ent_feature_np = np.zeros((node_size, ent_dim), dtype=np.float32)
                ent_feature_np[train_pair.reshape(-1)] = expanded_np
                ent_feature = tf.constant(ent_feature_np)
                del random_vec_np, expanded_np, ent_feature_np
            else:
                random_vec = K.l2_normalize(tf.random.normal((len(train_pair), ent_dim)), axis=-1)
                ent_feature = tf.tensor_scatter_nd_update(
                    tf.zeros((node_size, ent_dim)),
                    train_pair.reshape((-1, 1)),
                    tf.repeat(random_vec, 2, axis=0)
                )
        rel_feature = tf.zeros((rel_size, ent_feature.shape[-1]))
        
        ent_ent_graph = tf.SparseTensor(
            indices=self.ent_ent,
            values=self.ent_ent_val,
            dense_shape=(node_size, node_size)
        )
        rel_ent_graph = tf.SparseTensor(
            indices=self.rel_ent,
            values=K.ones(self.rel_ent.shape[0]),
            dense_shape=(rel_size, node_size)
        )
        ent_rel_graph = tf.SparseTensor(
            indices=self.ent_rel,
            values=K.ones(self.ent_rel.shape[0]),
            dense_shape=(node_size, rel_size)
        )
        
        ent_list, rel_list = [ent_feature], [rel_feature]
        for i in range(self.depth):
            new_rel_feature = batch_sparse_matmul(rel_ent_graph, ent_feature)
            new_rel_feature = tf.nn.l2_normalize(new_rel_feature, axis=-1)
            
            new_ent_feature = batch_sparse_matmul(ent_ent_graph, ent_feature)
            new_ent_feature += batch_sparse_matmul(ent_rel_graph, rel_feature)
            new_ent_feature = tf.nn.l2_normalize(new_ent_feature, axis=-1)
            
            ent_feature = new_ent_feature
            rel_feature = new_rel_feature
            ent_list.append(ent_feature)
            rel_list.append(rel_feature)
        
        if use_low_mem:

            ent_arrays = [to_numpy(e) for e in ent_list]
            ent_list.clear()
            del ent_feature
            ent_np = np.concatenate(ent_arrays, axis=1).astype(np.float32)
            del ent_arrays
            _l2_normalize_np_inplace(ent_np)
            ent_feature = tf.constant(ent_np)
            del ent_np
            rel_feature = K.l2_normalize(K.concatenate(rel_list, 1), -1)
        else:
            ent_feature = K.l2_normalize(K.concatenate(ent_list, 1), -1)
            rel_feature = K.l2_normalize(K.concatenate(rel_list, 1), -1)
        
        rel_feature = random_projection(rel_feature, rel_dim)
        
        batch_size = ent_feature.shape[-1] // mini_dim
        sparse_graph = tf.SparseTensor(
            indices=self.triples_idx,
            values=K.ones(self.triples_idx.shape[0]),
            dense_shape=(np.max(self.triples_idx) + 1, rel_size)
        )
        adj_value = batch_sparse_matmul(sparse_graph, rel_feature, save_mem=use_low_mem)
        
        features_list = []
        for batch in range(rel_dim // batch_size + 1):
            temp_list = []
            for head in range(batch_size):
                if batch * batch_size + head >= rel_dim:
                    break
                adj_col = adj_value[:, batch * batch_size + head]
                if isinstance(adj_col, np.ndarray):
                    adj_col = tf.constant(adj_col)
                sparse_graph = tf.SparseTensor(
                    indices=self.ent_tuple,
                    values=adj_col,
                    dense_shape=(node_size, node_size)
                )
                feature = batch_sparse_matmul(sparse_graph, random_projection(ent_feature, mini_dim))
                if use_low_mem:
                    feature = to_numpy(feature)
                temp_list.append(feature)
            if len(temp_list):
                if use_low_mem:
                    features_list.append(np.concatenate(temp_list, axis=-1))
                else:
                    features_list.append(to_numpy(K.concatenate(temp_list, -1)))
        
        features = np.concatenate(features_list, axis=-1)
        
        faiss.normalize_L2(features)
        
        if extra_feature is not None:
            features = np.concatenate([to_numpy(ent_feature), features], axis=-1)
        
        return features
    
    def train(self, train_pairs, epochs=1):
        self.train_pair = np.array(train_pairs, dtype=np.int64)
        
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        
        print(f"[LightEA] Training with {len(self.train_pair)} pairs")
        
        self.features = self.get_features(self.train_pair)
        print(f"[LightEA] Features shape: {self.features.shape}")
        
        return self
    
    def get_embeddings(self):
        if self.features is None:
            raise RuntimeError("Please call train() first")
        return self.features
    
    def evaluate(self, test_pairs, top_k=None):
        if self.features is None:
            raise RuntimeError("Please call train() first")
        
        if top_k is None:
            top_k = self.top_k
        
        test_pair = np.array(test_pairs, dtype=np.int64)
        h1, h10, mrr = test(test_pair, self.features, top_k)
        
        return h1, h10, mrr
    
    def evaluate_original(self, test_pairs, negative_pairs=None, top_k=None, iteration=15):
        if self.features is None:
            raise RuntimeError("Please call train() first")
        
        if top_k is None:
            top_k = self.top_k
        
        test_pair = np.array(test_pairs, dtype=np.int64)
        left = test_pair[:, 0]
        right = np.unique(test_pair[:, 1])
        
        right_id_to_idx = {int(r): idx for idx, r in enumerate(right)}
        
        index, sims = sparse_sinkhorn_sims(left, right, self.features, top_k, iteration, "test")
        ranks = to_numpy(tf.argsort(-sims, -1))
        index = to_numpy(index)
        
        h1, h5, h10, mrr = 0, 0, 0, 0
        pos = np.zeros(np.max(right) + 1, dtype=np.int64)
        pos[right] = np.arange(len(right))
        
        neg_applied_count = 0
        
        for i in range(len(test_pair)):
            src_id = int(test_pair[i, 0])
            correct_tgt_id = int(test_pair[i, 1])
            correct_tgt_idx = int(pos[correct_tgt_id])
            
            neg_idx_set = set()
            if negative_pairs and src_id in negative_pairs:
                for neg_tgt_id in negative_pairs[src_id]:
                    neg_tgt_id = int(neg_tgt_id)
                    if neg_tgt_id in right_id_to_idx:
                        neg_idx_set.add(right_id_to_idx[neg_tgt_id])
                if neg_idx_set:
                    neg_applied_count += 1
            
            sorted_candidates = index[i, ranks[i]]
            
            rank = 0
            found = False
            for cand_idx in sorted_candidates:
                if cand_idx == correct_tgt_idx:
                    found = True
                    break
                if cand_idx not in neg_idx_set:
                    rank += 1
            
            if found:
                if rank == 0:
                    h1 += 1
                if rank < 5:
                    h5 += 1
                if rank < 10:
                    h10 += 1
                mrr += 1 / (rank + 1)
        
        if negative_pairs and neg_applied_count > 0:
            print(f"  [Negative Info] Applied to {neg_applied_count}/{len(test_pair)} test pairs")
        
        n = len(test_pair)
        return h1/n, h5/n, h10/n, mrr/n
    
    def compute_similarity_matrix(self, src_entities, tgt_entities, top_k=None):
        if self.features is None:
            raise RuntimeError("Please call train() first")
        
        src_entities = np.array(src_entities)
        tgt_entities = np.array(tgt_entities)
        
        features_src = self.features[src_entities].copy()
        features_tgt = self.features[tgt_entities].copy()
        
        faiss.normalize_L2(features_src)
        faiss.normalize_L2(features_tgt)
        
        return np.dot(features_src, features_tgt.T)