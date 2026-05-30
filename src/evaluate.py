import numpy as np
import tensorflow as tf


class Evaluate:

    def __init__(self, dev_pair):
        self.dev_pair = np.array(dev_pair)

    def CSLS_cal(self, Lvec, Rvec, evaluate=True, batch_size=1024, k=10):
        L_sim_list, R_sim_list = [], []

        for epoch in range(0, len(Lvec), batch_size):
            L_batch = Lvec[epoch:epoch + batch_size]
            L_sim_list.append(np.dot(L_batch, Rvec.T))

        for epoch in range(0, len(Rvec), batch_size):
            R_batch = Rvec[epoch:epoch + batch_size]
            R_sim_list.append(np.dot(R_batch, Lvec.T))

        L_sim = np.vstack(L_sim_list)
        R_sim = np.vstack(R_sim_list)

        LR = np.mean(np.partition(L_sim, -k, axis=1)[:, -k:], axis=1)
        RL = np.mean(np.partition(R_sim, -k, axis=1)[:, -k:], axis=1)

        if evaluate:
            csls = 2 * L_sim - LR[:, np.newaxis] - RL[np.newaxis, :]
            results = []
            for i in range(len(self.dev_pair)):
                correct_idx = i
                sims = csls[i]
                correct_sim = sims[correct_idx]
                rank = np.sum(sims > correct_sim)
                results.append([i, rank])
            return np.array(results)
        else:
            csls_lr = 2 * L_sim - LR[:, np.newaxis] - RL[np.newaxis, :]
            csls_rl = 2 * R_sim - RL[:, np.newaxis] - LR[np.newaxis, :]
            r_rank = np.argsort(-csls_lr, axis=1)[:, 0]
            l_rank = np.argsort(-csls_rl, axis=1)[:, 0]
            return r_rank, l_rank

    def test(self, Lvec, Rvec, k=10):
        results = self.CSLS_cal(Lvec, Rvec, evaluate=True, k=k)
        hits1 = np.sum(results[:, 1] < 1) / len(Lvec)
        hits5 = np.sum(results[:, 1] < 5) / len(Lvec)
        hits10 = np.sum(results[:, 1] < 10) / len(Lvec)
        mrr = np.mean(1.0 / (results[:, 1] + 1))
        print(f"Hits@1: {hits1:.4f}  Hits@5: {hits5:.4f}  Hits@10: {hits10:.4f}  MRR: {mrr:.4f}")
        return results


def compute_csls_sim_matrix(Lvec, Rvec, k=10):
    sim = np.dot(Lvec, Rvec.T)
    LR = np.mean(np.partition(sim, -k, axis=1)[:, -k:], axis=1)
    sim_T = sim.T
    RL = np.mean(np.partition(sim_T, -k, axis=1)[:, -k:], axis=1)
    csls = 2 * sim - LR[:, np.newaxis] - RL[np.newaxis, :]
    return csls


def evaluate_with_negative_info(Lvec, Rvec, src_ids, tgt_ids, negative_pairs, k=10):
    n = len(Lvec)
    csls = compute_csls_sim_matrix(Lvec, Rvec, k=k)

    tgt_id_to_idx = {int(tgt_id): idx for idx, tgt_id in enumerate(tgt_ids)}

    neg_applied = 0
    for src_idx, src_id in enumerate(src_ids):
        src_id = int(src_id)
        if src_id in negative_pairs:
            for rejected_tgt_id in negative_pairs[src_id]:
                rejected_tgt_id = int(rejected_tgt_id)
                if rejected_tgt_id in tgt_id_to_idx:
                    tgt_idx = tgt_id_to_idx[rejected_tgt_id]
                    csls[src_idx, tgt_idx] = -np.inf
                    neg_applied += 1

    if neg_applied > 0:
        print(f"  [Negative Info] Applied {neg_applied} constraints")

    ranks = []
    for i in range(n):
        sims = csls[i]
        correct_sim = sims[i]
        rank = np.sum(sims > correct_sim)
        ranks.append(rank)

    ranks = np.array(ranks)
    hits1 = np.mean(ranks < 1)
    hits5 = np.mean(ranks < 5)
    hits10 = np.mean(ranks < 10)
    mrr = np.mean(1.0 / (ranks + 1))
    return hits1, hits5, hits10, mrr


evaluate = Evaluate
