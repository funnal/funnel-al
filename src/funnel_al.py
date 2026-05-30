import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Set


def _rank_normalize(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    if n == 0:
        return arr.astype(np.float32, copy=False)
    if n == 1:
        return np.array([1.0], dtype=np.float32)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    return ranks / (n - 1)


def _safe_rank_norm_masked(values: np.ndarray, mask_invalid: np.ndarray) -> np.ndarray:
    m = len(values)
    out = np.zeros(m, dtype=np.float32)
    valid = ~mask_invalid
    if valid.sum() == 0:
        return out
    v = values[valid]
    order = np.argsort(v)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(order), dtype=np.float32)
    if len(order) > 1:
        ranks /= (len(order) - 1)
    else:
        ranks = np.ones_like(ranks)
    out[valid] = ranks
    return out


class FunnelAL:

    def __init__(
        self,
        candidates_dict: Dict[int, List[int]],
        candidate_scores: Dict[int, np.ndarray],
        adj_dict_src: Dict[int, Set[int]],
        embeddings: Optional[np.ndarray] = None,
        adj_dict_tgt: Optional[Dict[int, Set[int]]] = None,
        reverse_candidates: Optional[Dict[int, List[int]]] = None,
        alpha: float = 4.0,
        gamma: float = 3.0,
        cov_eta: float = 20.0,
        funnel_topk: int = 10,
        inst_topk: int = 10,
        inst_lambda: float = 0.5,
        prev_rankings: Optional[Dict[int, List[int]]] = None,
        prefilter_ratio: float = 1.0,
        walk_steps: int = 4,
        walk_restart: float = 0.15,
        method: str = "funnel",
        seed: int = 42,
        verbose: bool = True,
        **_ignored,
    ):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.cov_eta = float(cov_eta)
        self.funnel_topk = int(funnel_topk)
        self.inst_topk = int(inst_topk)
        self.inst_lambda = float(inst_lambda)
        self.prev_rankings = prev_rankings or {}

        self.candidates_dict = candidates_dict or {}
        self.candidate_scores = candidate_scores or {}
        self.adj_dict_src = adj_dict_src or {}
        self.adj_dict_tgt = adj_dict_tgt or {}
        self.reverse_candidates = reverse_candidates or {}
        self.embeddings = embeddings

        self.prefilter_ratio = float(prefilter_ratio)
        self.walk_steps = int(walk_steps)
        self.walk_restart = float(walk_restart)

        self.method = method.lower()
        if self.method == "funnel_u":
            self.alpha = 0.0
            self.gamma = 0.0
        elif self.method == "funnel_uc":
            self.gamma = 0.0
        elif self.method == "funnel_ut":
            self.alpha = 0.0

        self.rng = np.random.RandomState(seed)
        self.verbose = verbose
        self._W_src = None

    def _compute_uncertainty(self, entities: List[int]) -> Dict[int, float]:
        n = len(entities)
        K = max(self.funnel_topk, 3)
        raw = np.empty(n, dtype=np.float32)
        missing = 0

        for i, s in enumerate(entities):
            cs = self.candidate_scores.get(s)
            if cs is None or len(cs) < 2:
                raw[i] = 1.0
                missing += 1
                continue

            cs = np.asarray(cs, dtype=np.float64)
            sorted_cs = np.sort(cs)[::-1]
            topK = sorted_cs[:K] if len(sorted_cs) >= K else sorted_cs
            L = len(topK)

            if L < 2:
                raw[i] = 1.0
                missing += 1
                continue

            score_range = float(topK[0] - topK[-1])
            if score_range < 1e-8:
                raw[i] = 1.0
                continue

            total_weighted_decay = 0.0
            weight_sum = 0.0
            for k in range(L - 1):
                w_k = float(L - 1 - k)
                decay_k = float(topK[k] - topK[k + 1]) / score_range
                total_weighted_decay += w_k * decay_k
                weight_sum += w_k

            avg_decay = total_weighted_decay / weight_sum if weight_sum > 0 else 0.0
            raw[i] = 1.0 - avg_decay

        out = _rank_normalize(raw)

        return {entities[i]: float(out[i]) for i in range(n)}

    def _compute_temporal_instability(self, entities: List[int]) -> np.ndarray:
        n = len(entities)
        K = self.inst_topk
        lam = self.inst_lambda
        raw = np.zeros(n, dtype=np.float32)

        if not self.prev_rankings:
            return raw

        n_with_prev = 0
        for i, s in enumerate(entities):
            prev_list = self.prev_rankings.get(s)
            if prev_list is None or len(prev_list) == 0:
                continue

            cands = self.candidates_dict.get(s)
            cs = self.candidate_scores.get(s)
            if cands is None or cs is None or len(cands) == 0:
                continue

            cs_arr = np.asarray(cs, dtype=np.float64)
            top_k = min(K, len(cands))
            top_indices = np.argsort(-cs_arr)[:top_k]
            curr_list = [cands[j] for j in top_indices]

            prev_topk = prev_list[:K]
            prev_set = set(prev_topk)
            curr_set = set(curr_list)

            union = prev_set | curr_set
            intersection = prev_set & curr_set

            jaccard_dist = 1.0 - len(intersection) / max(len(union), 1)

            prev_rank = {c: r for r, c in enumerate(prev_topk)}
            curr_rank = {c: r for r, c in enumerate(curr_list)}

            total_disp = 0.0
            for c in union:
                r_prev = prev_rank.get(c, K)
                r_curr = curr_rank.get(c, K)
                total_disp += abs(r_prev - r_curr)

            max_disp = K * len(union)
            footrule_dist = total_disp / max(max_disp, 1)

            raw[i] = (1.0 - lam) * jaccard_dist + lam * footrule_dist
            n_with_prev += 1

        if n_with_prev > 0:
            out = _rank_normalize(raw)
        else:
            out = raw

        return out

    def _build_W_src(self) -> Optional[sp.csr_matrix]:
        if self._W_src is not None:
            return self._W_src
        adj = self.adj_dict_src
        if not adj:
            return None
        n = max(adj.keys()) + 1
        rows, cols, vals = [], [], []
        for u, neis in adj.items():
            if not neis:
                continue
            inv = 1.0 / len(neis)
            for v in neis:
                if v >= n:
                    n = v + 1
                rows.append(u); cols.append(v); vals.append(inv)
        if not rows:
            self._W_src = sp.csr_matrix((n, n), dtype=np.float32)
            return self._W_src
        n = max(n, max(max(rows), max(cols)) + 1)
        self._W_src = sp.csr_matrix(
            (np.asarray(vals, dtype=np.float32),
             (np.asarray(rows), np.asarray(cols))),
            shape=(n, n),
        )
        return self._W_src

    def _batch_ppr(self, starts: List[int], W: sp.csr_matrix) -> sp.csr_matrix:
        if W is None or W.shape[0] == 0 or len(starts) == 0:
            return sp.csr_matrix((max(len(starts), 1), 0), dtype=np.float32)
        n = W.shape[0]
        m = len(starts)
        c = self.walk_restart
        init_rows, init_cols = [], []
        for i, s in enumerate(starts):
            if 0 <= s < n:
                init_rows.append(i); init_cols.append(s)
        init = sp.csr_matrix(
            (np.ones(len(init_rows), dtype=np.float32),
             (np.asarray(init_rows), np.asarray(init_cols))),
            shape=(m, n),
        )
        P = init.copy()
        for _ in range(self.walk_steps):
            P = (1.0 - c) * (P @ W) + c * init
        return P

    def _greedy_select(
        self,
        candidates: List[int],
        u_pool: np.ndarray,
        u_full_dict: Dict[int, float],
        t_pool: np.ndarray,
        budget: int,
    ) -> List[int]:
        m = len(candidates)
        if m == 0 or budget <= 0:
            return []
        budget = min(budget, m)

        W = self._build_W_src()
        P_src = self._batch_ppr(candidates, W)
        n_src = P_src.shape[1] if P_src.shape[1] > 0 else 1
        U_v_src = np.zeros(n_src, dtype=np.float32)
        for v, u in u_full_dict.items():
            if 0 <= v < n_src:
                U_v_src[v] = float(u)
        cov_src = np.zeros(n_src, dtype=np.float32)

        t_boost = 1.0 + self.gamma * t_pool

        selected_idx = []
        selected_mask = np.zeros(m, dtype=bool)

        for _ in range(budget):
            if self.cov_eta > 1e-6:
                w_src = U_v_src * np.exp(-self.cov_eta * cov_src)
            else:
                w_src = U_v_src * (1.0 - cov_src)
            gain_c = P_src.dot(w_src)
            gain_c[selected_mask] = -np.inf
            c_norm = _safe_rank_norm_masked(gain_c, selected_mask)

            score = u_pool * (1.0 + self.alpha * c_norm) * t_boost
            score[selected_mask] = -np.inf
            score = score + 1e-12 * self.rng.rand(m)

            best_i = int(np.argmax(score))
            selected_idx.append(best_i)
            selected_mask[best_i] = True

            row_src = P_src.getrow(best_i).toarray().ravel()
            cov_src = 1.0 - (1.0 - cov_src) * (1.0 - row_src)
            cov_src = np.clip(cov_src, 0.0, 1.0)

        selected = [int(candidates[i]) for i in selected_idx]
        return selected

    def select(self, unlabeled_src: List[int], budget: int,
               unlabeled_tgt: Optional[List[int]] = None) -> List[int]:
        if not unlabeled_src or budget <= 0:
            return []
        budget = min(budget, len(unlabeled_src))

        u_full = self._compute_uncertainty(unlabeled_src)

        if self.alpha < 1e-6 and self.gamma < 1e-6:
            return sorted(unlabeled_src, key=lambda x: -u_full[x])[:budget]

        t_full = self._compute_temporal_instability(unlabeled_src)

        if self.alpha < 1e-6 and self.gamma > 1e-6:
            t_boost = 1.0 + self.gamma * t_full
            u_arr = np.array([u_full[s] for s in unlabeled_src], dtype=np.float32)
            combined = u_arr * t_boost
            top_idx = np.argsort(-combined)[:budget]
            return [unlabeled_src[i] for i in top_idx]

        if self.prefilter_ratio < 0.999:
            n_keep = int(np.ceil(len(unlabeled_src) * self.prefilter_ratio))
            n_keep = max(n_keep, 2 * budget)
            n_keep = min(n_keep, len(unlabeled_src))
            u_arr = np.array([u_full[s] for s in unlabeled_src], dtype=np.float32)
            ranked = np.argsort(-u_arr)
            prefilter_idx = ranked[:n_keep]
            prefiltered = [unlabeled_src[i] for i in prefilter_idx]
            t_pool = t_full[prefilter_idx]
        else:
            prefiltered = list(unlabeled_src)
            t_pool = t_full

        u_pool = np.array([u_full[s] for s in prefiltered], dtype=np.float32)

        selected = self._greedy_select(
            candidates=prefiltered,
            u_pool=u_pool,
            u_full_dict=u_full,
            t_pool=t_pool,
            budget=budget,
        )
        return selected


class FunnelStrategy:

    _ACCEPTED = {
        "alpha", "gamma", "cov_eta", "funnel_topk",
        "inst_topk", "inst_lambda", "prev_rankings",
        "prefilter_ratio", "walk_steps", "walk_restart",
        "method", "seed", "verbose",
    }

    def __init__(self, candidates_dict, candidate_scores, adj_dict_src,
                 adj_dict_tgt=None, reverse_candidates=None, embeddings=None,
                 **kwargs):
        accepted = {k: v for k, v in kwargs.items() if k in self._ACCEPTED}
        self.al = FunnelAL(
            candidates_dict=candidates_dict,
            candidate_scores=candidate_scores,
            adj_dict_src=adj_dict_src,
            embeddings=embeddings,
            adj_dict_tgt=adj_dict_tgt,
            reverse_candidates=reverse_candidates,
            **accepted,
        )

    def select(self, unlabeled, budget, unlabeled_tgt=None):
        return self.al.select(unlabeled, budget, unlabeled_tgt=unlabeled_tgt)
