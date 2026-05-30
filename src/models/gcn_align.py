
import os
import numpy as np
import scipy.sparse as sp
import scipy.spatial.distance
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf1 = tf.compat.v1

def _ensure_tf1_graph_mode():
    try:
        tf1.disable_eager_execution()
    except Exception:
        pass


def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def func(KG):
    head = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f


def ifunc(KG):
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if


def get_weighted_adj(e, KG):
    r2f = func(KG)
    r2if = ifunc(KG)
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
        else:
            M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
        else:
            M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[1])
        col.append(key[0])
        data.append(M[key])
    return sp.coo_matrix((data, (row, col)), shape=(e, e))


def loadfile(fn, num=1):
    ret = []
    if not os.path.exists(fn):
        return ret
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                if i < len(th):
                    x.append(int(th[i]))
            if len(x) == num:
                ret.append(tuple(x))
    return ret


def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        if not os.path.exists(fn):
            continue
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if len(th) >= 2:
                    ent2id[th[1]] = int(th[0])
    return ent2id


def loadattr(fns, e, ent2id):
    cnt = {}
    for fn in fns:
        if not os.path.exists(fn):
            continue
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    
    if not cnt:
        return sp.eye(e)
    
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    num_features = min(len(fre), 2000)
    attr2id = {}
    for i in range(num_features):
        attr2id[fre[i][0]] = i
    
    M = {}
    for fn in fns:
        if not os.path.exists(fn):
            continue
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            M[(ent2id[th[0]], attr2id[th[i]])] = 1.0
    
    row, col, data = [], [], []
    for key in M:
        row.append(key[0])
        col.append(key[1])
        data.append(M[key])
    
    if not row:
        return sp.eye(e)
    
    return sp.coo_matrix((data, (row, col)), shape=(e, num_features))


def get_hits(vec, test_pair, top_k=(1, 5, 10, 50)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    
    top_lr = [0] * len(top_k)
    mrr = 0.0
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        mrr += 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    
    n = len(test_pair)
    results = {f'hits@{k}': top_lr[j] / n for j, k in enumerate(top_k)}
    results['mrr'] = mrr / n
    return results


def get_combine_hits(se_vec, ae_vec, beta, test_pair, top_k=(1, 5, 10, 50)):
    vec = np.concatenate([se_vec * beta, ae_vec * (1.0 - beta)], axis=1)
    return get_hits(vec, test_pair, top_k)


class GCNAlignWrapper:
    
    def __init__(self, data_path, se_dim=1000, ae_dim=100, beta=0.9, 
                 learning_rate=20.0, gamma=3.0, k=5, seed=12306, use_ae=True):
        _ensure_tf1_graph_mode()
        self.seed = seed
        np.random.seed(seed)
        
        self.data_path = data_path if data_path.endswith('/') else data_path + '/'
        self.se_dim = se_dim
        self.ae_dim = ae_dim
        self.beta = beta
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.k = k
        self.use_ae = use_ae
        
        self._load_data()
        
        if self.ae_input_dim == self.num_entities:
            print(f"[GCN-Align] WARNING: No valid attribute files, switching to SE-only mode")
            self.use_ae = False
        
        if not self.use_ae:
            self.beta = 1.0
            print(f"[GCN-Align] Running in SE-only mode")
        
        self.graph = None
        self.sess = None
        self.vec_se = None
        self.vec_ae = None
        self.features = None
        self.train_pair = None
        self.is_initialized = False
        
        print(f"[GCN-Align] Loaded: {self.num_entities} entities, {len(self.KG)} triples")
    
    def _load_data(self):
        Es = [self.data_path + 'ent_ids_1', self.data_path + 'ent_ids_2']
        Ts = [self.data_path + 'triples_1', self.data_path + 'triples_2']
        As = [self.data_path + 'training_attrs_1', self.data_path + 'training_attrs_2']
        
        e1 = set(loadfile(Es[0], 1))
        e2 = set(loadfile(Es[1], 1))
        self.num_entities = len(e1 | e2)
        
        self.ent2id = get_ent2id(Es)
        self.KG = loadfile(Ts[0], 3) + loadfile(Ts[1], 3)
        
        self.adj = get_weighted_adj(self.num_entities, self.KG)
        self.support_tuple = preprocess_adj(self.adj)
        
        self.attr = loadattr(As, self.num_entities, self.ent2id)
        self.ae_tuple = sparse_to_tuple(sp.coo_matrix(self.attr))
        self.ae_input_dim = self.attr.shape[1]
        
        print(f"[GCN-Align] Adj shape: {self.adj.shape}, Attr shape: {self.attr.shape}")
    
    def _build_graph(self, train_pairs):
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            tf1.set_random_seed(self.seed)
            
            t = len(train_pairs)
            e = self.num_entities
            
            self.ph_support = tf1.sparse_placeholder(tf.float32, name='support')
            self.ph_neg_left = tf1.placeholder(tf.int32, [t * self.k], name='neg_left')
            self.ph_neg_right = tf1.placeholder(tf.int32, [t * self.k], name='neg_right')
            self.ph_neg2_left = tf1.placeholder(tf.int32, [t * self.k], name='neg2_left')
            self.ph_neg2_right = tf1.placeholder(tf.int32, [t * self.k], name='neg2_right')
            
            
            with tf1.variable_scope('gcn_se'):
                stddev = 1.0 / math.sqrt(e)
                self.weights_se_var = tf1.Variable(
                    tf1.truncated_normal([e, self.se_dim], stddev=stddev),
                    name='weights'
                )
                self.weights_se = tf.nn.l2_normalize(self.weights_se_var, axis=1)
            
            h1_se = tf.sparse.sparse_dense_matmul(self.ph_support, self.weights_se)
            h1_se = tf.nn.relu(h1_se)
            
            self.output_se = tf.sparse.sparse_dense_matmul(self.ph_support, h1_se)
            
            self.loss_se = self._compute_loss(self.output_se, train_pairs)
            
            optimizer_se = tf1.train.GradientDescentOptimizer(self.learning_rate)
            self.train_op_se = optimizer_se.minimize(self.loss_se)
            
            if self.use_ae:
                self.ph_features = tf1.sparse_placeholder(tf.float32, name='features')
                
                with tf1.variable_scope('gcn_ae'):
                    stddev = 1.0 / math.sqrt(self.ae_input_dim)
                    self.weights_ae_var = tf1.Variable(
                        tf1.truncated_normal([self.ae_input_dim, self.ae_dim], stddev=stddev),
                        name='weights'
                    )
                    self.weights_ae = tf.nn.l2_normalize(self.weights_ae_var, axis=1)
                
                pre_sup_ae = tf.sparse.sparse_dense_matmul(self.ph_features, self.weights_ae)
                h1_ae = tf.sparse.sparse_dense_matmul(self.ph_support, pre_sup_ae)
                h1_ae = tf.nn.relu(h1_ae)
                
                self.output_ae = tf.sparse.sparse_dense_matmul(self.ph_support, h1_ae)
                
                self.loss_ae = self._compute_loss(self.output_ae, train_pairs)
                optimizer_ae = tf1.train.GradientDescentOptimizer(self.learning_rate)
                self.train_op_ae = optimizer_ae.minimize(self.loss_ae)
            else:
                self.ph_features = None
                self.output_ae = None
                self.loss_ae = None
                self.train_op_ae = None
            
            self.init_op = tf1.global_variables_initializer()
    
    def _compute_loss(self, output, train_pairs):
        left = train_pairs[:, 0]
        right = train_pairs[:, 1]
        t = len(train_pairs)
        k = self.k
        gamma = self.gamma
        
        left_x = tf.nn.embedding_lookup(output, left)
        right_x = tf.nn.embedding_lookup(output, right)
        A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
        
        neg_l_x = tf.nn.embedding_lookup(output, self.ph_neg_left)
        neg_r_x = tf.nn.embedding_lookup(output, self.ph_neg_right)
        B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
        C = -tf.reshape(B, [t, k])
        D = A + gamma
        L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
        
        neg2_l_x = tf.nn.embedding_lookup(output, self.ph_neg2_left)
        neg2_r_x = tf.nn.embedding_lookup(output, self.ph_neg2_right)
        B2 = tf.reduce_sum(tf.abs(neg2_l_x - neg2_r_x), 1)
        C2 = -tf.reshape(B2, [t, k])
        L2 = tf.nn.relu(tf.add(C2, tf.reshape(D, [t, 1])))
        
        return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)
    
    def _get_feed_dict(self, neg_left, neg_right, neg2_left, neg2_right):
        coords, values, shape = self.support_tuple
        support_feed = tf1.SparseTensorValue(
            indices=np.array(coords, dtype=np.int64),
            values=np.array(values, dtype=np.float32),
            dense_shape=shape
        )
        
        feed_dict = {
            self.ph_support: support_feed,
            self.ph_neg_left: neg_left,
            self.ph_neg_right: neg_right,
            self.ph_neg2_left: neg2_left,
            self.ph_neg2_right: neg2_right,
        }
        
        if self.use_ae and self.ph_features is not None:
            coords, values, shape = self.ae_tuple
            features_feed = tf1.SparseTensorValue(
                indices=np.array(coords, dtype=np.int64),
                values=np.array(values, dtype=np.float32),
                dense_shape=shape
            )
            feed_dict[self.ph_features] = features_feed
        
        return feed_dict
    
    def train(self, train_pairs, epochs=2000, verbose=True):
        self.train_pair = np.array(train_pairs, dtype=np.int32)
        
        np.random.seed(self.seed)
        
        if self.sess is not None:
            self.sess.close()
        
        self._build_graph(self.train_pair)
        
        config = tf1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf1.Session(graph=self.graph, config=config)
        self.sess.run(self.init_op)
        
        self.is_initialized = True
        
        if verbose:
            print(f"[GCN-Align] Training with {len(self.train_pair)} pairs for {epochs} epochs")
        
        t = len(self.train_pair)
        k = self.k
        e = self.num_entities
        
        L = np.ones((t, k)) * (self.train_pair[:, 0].reshape((t, 1)))
        neg_left = L.reshape((t * k,)).astype(np.int32)
        L = np.ones((t, k)) * (self.train_pair[:, 1].reshape((t, 1)))
        neg2_right = L.reshape((t * k,)).astype(np.int32)
        
        for epoch in range(epochs):
            if epoch % 10 == 0:
                neg2_left = np.random.choice(e, t * k).astype(np.int32)
                neg_right = np.random.choice(e, t * k).astype(np.int32)
            
            feed_dict = self._get_feed_dict(neg_left, neg_right, neg2_left, neg2_right)
            
            _, loss_se = self.sess.run([self.train_op_se, self.loss_se], feed_dict=feed_dict)
            
            if self.use_ae and self.train_op_ae is not None:
                _, loss_ae = self.sess.run([self.train_op_ae, self.loss_ae], feed_dict=feed_dict)
            else:
                loss_ae = 0.0
            
            if verbose and (epoch + 1) % 100 == 0:
                if self.use_ae:
                    print(f"  Epoch {epoch+1}/{epochs}, SE_loss={loss_se:.5f}, AE_loss={loss_ae:.5f}")
                else:
                    print(f"  Epoch {epoch+1}/{epochs}, SE_loss={loss_se:.5f}")
        
        self._update_embeddings()
        
        if verbose:
            print(f"[GCN-Align] Training complete. Embedding shape: {self.features.shape}")
        
        return self
    
    def _update_embeddings(self):
        coords, values, shape = self.support_tuple
        support_feed = tf1.SparseTensorValue(
            indices=np.array(coords, dtype=np.int64),
            values=np.array(values, dtype=np.float32),
            dense_shape=shape
        )
        
        t = len(self.train_pair)
        k = self.k
        dummy = np.zeros(t * k, dtype=np.int32)
        
        feed_dict = {
            self.ph_support: support_feed,
            self.ph_neg_left: dummy,
            self.ph_neg_right: dummy,
            self.ph_neg2_left: dummy,
            self.ph_neg2_right: dummy,
        }
        
        if self.use_ae and self.ph_features is not None:
            coords, values, shape = self.ae_tuple
            features_feed = tf1.SparseTensorValue(
                indices=np.array(coords, dtype=np.int64),
                values=np.array(values, dtype=np.float32),
                dense_shape=shape
            )
            feed_dict[self.ph_features] = features_feed
        
        self.vec_se = self.sess.run(self.output_se, feed_dict=feed_dict)
        
        if self.use_ae and self.output_ae is not None:
            self.vec_ae = self.sess.run(self.output_ae, feed_dict=feed_dict)
            self.features = np.concatenate([
                self.vec_se * self.beta,
                self.vec_ae * (1.0 - self.beta)
            ], axis=1)
        else:
            self.vec_ae = None
            self.features = self.vec_se
    
    def get_embeddings(self):
        if self.features is None:
            raise RuntimeError("Please call train() first")
        return self.features
    
    def get_se_embeddings(self):
        if self.vec_se is None:
            raise RuntimeError("Please call train() first")
        return self.vec_se
    
    def get_ae_embeddings(self):
        if self.vec_ae is None:
            raise RuntimeError("AE embeddings not available")
        return self.vec_ae
    
    def evaluate(self, test_pairs, mode='combine'):
        if self.features is None:
            raise RuntimeError("Please call train() first")
        
        test_pairs = list(test_pairs)
        
        if mode == 'se' or not self.use_ae:
            results = get_hits(self.vec_se, test_pairs)
        elif mode == 'ae':
            if self.vec_ae is None:
                results = get_hits(self.vec_se, test_pairs)
            else:
                results = get_hits(self.vec_ae, test_pairs)
        else:
            if self.vec_ae is not None:
                results = get_combine_hits(self.vec_se, self.vec_ae, self.beta, test_pairs)
            else:
                results = get_hits(self.vec_se, test_pairs)
        
        print(f"[GCN-Align] Hits@1={results['hits@1']:.4f}, "
              f"Hits@5={results['hits@5']:.4f}, Hits@10={results['hits@10']:.4f}, MRR={results['mrr']:.4f}")
        
        return results
    
    def evaluate_original(self, test_pairs, negative_pairs=None, top_k=None, iteration=None):
        if self.features is None:
            raise RuntimeError("Please call train() first")
        
        test_pairs = np.array(test_pairs)
        vec = self.features
        
        
        Lvec = np.array([vec[e1] for e1, e2 in test_pairs])
        Rvec = np.array([vec[e2] for e1, e2 in test_pairs])
        
        sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
        
        neg_applied_count = 0
        if negative_pairs:
            right_list = [int(e2) for e1, e2 in test_pairs]
            for i, (src_id, tgt_id) in enumerate(test_pairs):
                src_id = int(src_id)
                if src_id in negative_pairs:
                    for neg_tgt_id in negative_pairs[src_id]:
                        neg_tgt_id = int(neg_tgt_id)
                        for j, r in enumerate(right_list):
                            if r == neg_tgt_id:
                                sim[i, j] = np.inf
                                neg_applied_count += 1
        
        if neg_applied_count > 0:
            print(f"  [Negative Info] Applied {neg_applied_count} negative constraints")
        
        h1, h5, h10, mrr = 0, 0, 0, 0.0
        
        for i in range(len(test_pairs)):
            rank = sim[i, :].argsort()
            rank_index = np.where(rank == i)[0][0]
            
            if rank_index == 0:
                h1 += 1
            if rank_index < 5:
                h5 += 1
            if rank_index < 10:
                h10 += 1
            mrr += 1.0 / (rank_index + 1)
        
        n = len(test_pairs)
        h1, h5, h10, mrr = h1/n, h5/n, h10/n, mrr/n
        
        print(f"[GCN-Align] Hits@1={h1:.4f}, Hits@5={h5:.4f}, Hits@10={h10:.4f}, MRR={mrr:.4f}")
        
        return h1, h5, h10, mrr
    
    def compute_similarity_matrix(self, src_entities, tgt_entities, top_k=None):
        if self.features is None:
            raise RuntimeError("Please call train() first")
        
        src_entities = np.array(src_entities)
        tgt_entities = np.array(tgt_entities)
        
        features_src = self.features[src_entities]
        features_tgt = self.features[tgt_entities]
        
        dist = scipy.spatial.distance.cdist(features_src, features_tgt, metric='cityblock')
        return -dist
    
    def train_al(self, train_pairs, epochs=200, verbose=True, reinit=False):
        return self.train(train_pairs, epochs=epochs, verbose=verbose)
    
    def __del__(self):
        if hasattr(self, 'sess') and self.sess is not None:
            try:
                self.sess.close()
            except:
                pass
