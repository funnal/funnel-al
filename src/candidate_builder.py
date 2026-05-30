import numpy as np
from collections import defaultdict

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False


class CandidateBuilder:
    def __init__(self, data_loader, top_k: int = 50, k_csls: int = 10,
                 chunk_size: int = 4096, search_expansion: int = 200,
                 use_gpu_faiss: bool = True):
        self.data_loader = data_loader
        self.top_k = int(top_k)
        self.k_csls = int(k_csls)
        self.chunk_size = int(chunk_size)
        self.search_expansion = int(max(search_expansion, 2 * top_k))
        self.use_gpu_faiss = use_gpu_faiss
        self.candidates = {}
        self.candidate_scores = {}
        self.target_entities = None

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x / n

    def _build_faiss_index(self, features: np.ndarray):
        n, d = features.shape
        features = np.ascontiguousarray(features, dtype=np.float32)

        if n < 10000:
            index = faiss.IndexFlatIP(d)
        else:
            nlist = max(int(np.sqrt(n)), 16)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = max(nlist // 4, 8)
            index.train(features)

        if self.use_gpu_faiss:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except (AttributeError, Exception):
                pass

        index.add(features)
        return index

    def _compute_csls_twostage(self, source_features, target_features):
        src = self._l2_normalize(source_features)
        tgt = self._l2_normalize(target_features)

        n_src = src.shape[0]
        n_tgt = tgt.shape[0]

        M = min(self.search_expansion, n_tgt)
        k_csls = min(self.k_csls, n_tgt)
        k = min(self.top_k, n_tgt)

        tgt_index = self._build_faiss_index(tgt)
        cos_sims_s2t, nn_idx_s2t = tgt_index.search(src, M)
        r_s = np.mean(cos_sims_s2t[:, :k_csls], axis=1)

        src_index = self._build_faiss_index(src)
        cos_sims_t2s, _ = src_index.search(tgt, k_csls)
        r_t = np.mean(cos_sims_t2s, axis=1)

        topk_idx = np.zeros((n_src, k), dtype=np.int64)
        topk_scores = np.zeros((n_src, k), dtype=np.float32)

        for i in range(n_src):
            cand_indices = nn_idx_s2t[i]
            cand_cos_sims = cos_sims_s2t[i]
            csls_scores = 2 * cand_cos_sims - r_s[i] - r_t[cand_indices]

            if k < M:
                part = np.argpartition(csls_scores, -k)[-k:]
                part_scores = csls_scores[part]
                order = np.argsort(part_scores)[::-1]
                best_k = part[order]
            else:
                best_k = np.argsort(csls_scores)[::-1][:k]

            topk_idx[i] = cand_indices[best_k]
            topk_scores[i] = csls_scores[best_k]

        return topk_idx, topk_scores

    def _compute_csls_exact(self, source_features, target_features):
        src = self._l2_normalize(source_features)
        tgt = self._l2_normalize(target_features)

        n_src, n_tgt = src.shape[0], tgt.shape[0]
        k = min(self.top_k, n_tgt)
        k_s = min(self.k_csls, n_tgt)
        k_t = min(self.k_csls, n_src)

        cos_sim = np.dot(src, tgt.T)
        r_s = np.mean(np.partition(cos_sim, -k_s, axis=1)[:, -k_s:], axis=1, keepdims=True)
        r_t = np.mean(np.partition(cos_sim, -k_t, axis=0)[-k_t:, :], axis=0, keepdims=True)
        csls_scores = 2 * cos_sim - r_s - r_t

        part = np.argpartition(csls_scores, -k, axis=1)[:, -k:]
        part_scores = np.take_along_axis(csls_scores, part, axis=1)
        order = np.argsort(part_scores, axis=1)[:, ::-1]
        topk_idx = np.take_along_axis(part, order, axis=1)
        topk_scores = np.take_along_axis(csls_scores, topk_idx, axis=1)

        return topk_idx, topk_scores

    def build_candidates_by_structure(self, features: np.ndarray):
        source_ents = list(self.data_loader.get_source_entities())
        target_ents = list(self.data_loader.get_target_entities())
        self.target_entities = np.array(target_ents, dtype=np.int64)

        if len(source_ents) == 0 or len(target_ents) == 0:
            self.candidates = {}
            self.candidate_scores = {}
            return self.candidates

        source_features = features[source_ents].astype('float32', copy=False)
        target_features = features[target_ents].astype('float32', copy=False)

        n_src = len(source_ents)
        n_tgt = len(target_ents)

        if _HAS_FAISS and n_src * n_tgt > 1e7:
            topk_idx, topk_scores = self._compute_csls_twostage(source_features, target_features)
        else:
            topk_idx, topk_scores = self._compute_csls_exact(source_features, target_features)

        for i, src_ent in enumerate(source_ents):
            self.candidates[src_ent] = [target_ents[j] for j in topk_idx[i]]
            self.candidate_scores[src_ent] = topk_scores[i].tolist()

        return self.candidates

    def update_candidates(self, features: np.ndarray):
        return self.build_candidates_by_structure(features)

    def get_candidates(self, source_ent: int):
        return self.candidates.get(source_ent, [])

    def get_candidate_scores(self, source_ent: int):
        return self.candidate_scores.get(source_ent, [])

    def get_all_candidates(self):
        return self.candidates

    def get_reverse_candidates(self):
        reverse_candidates = defaultdict(list)
        for src_ent, cands in self.candidates.items():
            for tgt_ent in cands:
                reverse_candidates[tgt_ent].append(src_ent)
        return dict(reverse_candidates)
