#!/usr/bin/env python3
import os
import sys

os.environ.setdefault('LIGHTEA_FAISS_GPU', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import warnings
warnings.filterwarnings('ignore')

import math
import argparse
import json
import random
import numpy as np
from tqdm import trange


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        try:
            tf.compat.v1.set_random_seed(seed)
        except Exception:
            pass
    except Exception:
        pass


def _peek(name, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


_QUIET = "--quiet" in sys.argv
_SEED = int(_peek("--seed", "42"))
set_all_seeds(_SEED)

if _QUIET:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    warnings.filterwarnings("ignore")

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import keras.backend as K

set_all_seeds(_SEED)

from src.utils import load_triples, load_alignment_pair, get_matrix
from src.evaluate import evaluate, evaluate_with_negative_info
from src.models import build_duala_models, predict_embeddings, HAS_LIGHTEA, HAS_GCNALIGN
from src.pool_loader import PoolDataLoader
from src.candidate_builder import CandidateBuilder
from src.io_utils import write_jsonl, write_tsv, read_labels_tsv
from src.funnel_al import FunnelStrategy

if HAS_LIGHTEA:
    from src.models.lightea import LightEAWrapper
if HAS_GCNALIGN:
    from src.models.gcn_align import GCNAlignWrapper

try:
    from src.data_loader import KGDataLoader
except Exception:
    KGDataLoader = None


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_adj_dict(edges):
    adj = {}
    for h, t in edges:
        h, t = int(h), int(t)
        adj.setdefault(h, set()).add(t)
        adj.setdefault(t, set()).add(h)
    return adj


def split_adj_by_entity_range(adj_dict, src_entities, tgt_entities):
    src_set = set(src_entities)
    tgt_set = set(tgt_entities)
    adj_src, adj_tgt = {}, {}
    for e, neighbors in adj_dict.items():
        if e in src_set:
            adj_src[e] = neighbors & src_set
        if e in tgt_set:
            adj_tgt[e] = neighbors & tgt_set
    return adj_src, adj_tgt


def metrics(results):
    if results is None:
        return (0., 0., 0., 0.)
    ranks = np.array(results)[:, 1].astype(np.float32)
    n = float(len(ranks)) if len(ranks) > 0 else 1.
    return (
        float(np.sum(ranks < 1)) / n,
        float(np.sum(ranks < 5)) / n,
        float(np.sum(ranks < 10)) / n,
        float(np.sum(1. / (ranks + 1.))) / n,
    )


def derive_dataset_name(data_path: str) -> str:
    p = data_path.rstrip("/").rstrip("\\")
    return os.path.basename(p) or "unknown"


def derive_output_dir(args) -> str:
    if args.output_dir:
        return args.output_dir
    ds = derive_dataset_name(args.data_path)
    return os.path.join(args.exp_root, ds, args.model, args.strategy, f"seed{args.seed}")


def load_data_raw(lang):
    if not lang.endswith('/') and not lang.endswith('\\'):
        lang += '/'
    entity1, rel1, triples1 = load_triples(lang + 'triples_1')
    entity2, rel2, triples2 = load_triples(lang + 'triples_2')
    if "_en" in lang:
        alignment_pair = load_alignment_pair(lang + 'ref_ent_ids')
    else:
        train_pair_raw = load_alignment_pair(lang + 'sup_ent_ids')
        dev_pair_raw = load_alignment_pair(lang + 'ref_ent_ids')
        alignment_pair = train_pair_raw + dev_pair_raw
    adj_matrix, r_index, r_val, adj_features, rel_features = get_matrix(
        triples1 + triples2, entity1.union(entity2), rel1.union(rel2)
    )
    src_entities = list(entity1)
    tgt_entities = list(entity2)
    return (alignment_pair, adj_matrix, np.array(r_index), np.array(r_val),
            adj_features, rel_features, src_entities, tgt_entities)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/zh_en/")
    ap.add_argument("--exp_root", default="results")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--target_ratio", type=float, default=0.30)
    ap.add_argument("--init_ratio", type=float, default=0.05)
    ap.add_argument("--step_ratio", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=10)

    ap.add_argument("--strategy", default="funnel",
                    choices=["funnel", "funnel_uc", "funnel_ut", "funnel_u"])

    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--funnel_gamma", type=float, default=3.0)
    ap.add_argument("--cov_eta", type=float, default=20.0)
    ap.add_argument("--funnel_topk", type=int, default=10)
    ap.add_argument("--inst_topk", type=int, default=10)
    ap.add_argument("--inst_lambda", type=float, default=0.5)

    ap.add_argument("--prefilter_ratio", type=float, default=1.0)
    ap.add_argument("--walk_steps", type=int, default=4)
    ap.add_argument("--walk_restart", type=float, default=0.15)

    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--cand_topk", type=int, default=50)

    ap.add_argument("--epochs_first", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--node_hidden", type=int, default=128)
    ap.add_argument("--dropout_rate", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--hinge_gamma", type=float, default=1.0)
    ap.add_argument("--depth", type=int, default=2)

    ap.add_argument("--model", type=str, default="duala",
                    choices=["duala", "lightea", "gcn_align"])
    ap.add_argument("--ent_dim", type=int, default=1024)
    ap.add_argument("--rel_dim", type=int, default=512)
    ap.add_argument("--se_dim", type=int, default=200)
    ap.add_argument("--ae_dim", type=int, default=100)
    ap.add_argument("--gcn_beta", type=float, default=0.9)
    ap.add_argument("--gcn_lr", type=float, default=20.0)
    ap.add_argument("--gcn_gamma", type=float, default=3.0)
    ap.add_argument("--gcn_k", type=int, default=5)
    ap.add_argument("--gcn_epochs", type=int, default=2000)
    ap.add_argument("--gcn_epochs_al", type=int, default=500)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--simulate_oracle", action="store_true")
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args()

    if args.model == "lightea" and not HAS_LIGHTEA:
        args.model = "duala"
    if args.model == "gcn_align" and not HAS_GCNALIGN:
        args.model = "duala"

    out_dir = derive_output_dir(args)
    args.output_dir = out_dir

    print(f"\n{'='*60}")
    print(f"[SEED] {args.seed}  [MODEL] {args.model}  [STRATEGY] {args.strategy}")
    print(f"[DATASET] {derive_dataset_name(args.data_path)}")
    print(f"[OUTPUT] {out_dir}")
    print(f"[PARAMS] alpha={args.alpha:.2f} gamma={args.funnel_gamma:.2f} "
          f"cov_eta={args.cov_eta:.1f} funnel_topk={args.funnel_topk} "
          f"inst_topk={args.inst_topk} inst_lambda={args.inst_lambda:.2f}")
    print(f"{'='*60}")
    set_all_seeds(args.seed)

    if args.model == "duala":
        try:
            tf.compat.v1.disable_eager_execution()
        except Exception:
            pass
        cfg = tf.compat.v1.ConfigProto()
        cfg.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=cfg)
        try:
            K.set_session(sess)
        except Exception:
            pass
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    set_all_seeds(args.seed)

    ensure_dir(out_dir)
    prog_path = os.path.join(out_dir, "progress.jsonl")
    if os.path.exists(prog_path):
        os.remove(prog_path)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump({k: (int(v) if isinstance(v, (np.integer,)) else v)
                   for k, v in vars(args).items()
                   if not k.startswith("_")}, f, ensure_ascii=False, indent=2)

    print(f"\nLoading data...")
    (all_pairs, adj_matrix, r_index, r_val, adj_features, rel_features,
     all_src_entities, all_tgt_entities) = load_data_raw(args.data_path)

    adj_edges = np.stack(adj_matrix.nonzero(), axis=1)
    node_size = adj_features.shape[0]
    rel_size = rel_features.shape[1]
    triple_size = len(adj_edges)

    adj_dict_full = build_adj_dict(adj_edges)
    adj_dict_src, adj_dict_tgt = split_adj_by_entity_range(
        adj_dict_full, all_src_entities, all_tgt_entities
    )

    rel_matrix = np.stack(rel_features.nonzero(), axis=1)
    ent_matrix = np.stack(adj_features.nonzero(), axis=1)

    gold = {int(p[0]): int(p[1]) for p in all_pairs}

    total = len(all_pairs)
    target = int(math.ceil(args.target_ratio * total))
    init_budget = int(math.ceil(args.init_ratio * total))
    step_budget = int(math.ceil(args.step_ratio * total))

    if args.model == "duala":
        train_model, feature_model = build_duala_models(
            node_size=node_size, rel_size=rel_size, triple_size=triple_size,
            node_hidden=args.node_hidden, dropout_rate=args.dropout_rate,
            gamma=args.hinge_gamma, lr=args.lr, depth=args.depth,
        )
    else:
        train_model, feature_model = None, None

    rng = np.random.RandomState(args.seed)

    unlabeled_pairs = list(map(tuple, all_pairs))
    unlabeled_src = [p[0] for p in unlabeled_pairs]
    unlabeled_tgt = [p[1] for p in unlabeled_pairs]
    labeled_pairs = []
    negative_pairs = {}

    base_loader = None
    if KGDataLoader:
        try:
            base_loader = KGDataLoader(args.data_path)
        except Exception:
            pass
    if not base_loader:
        class _DummyLoader:
            id2ent_1 = {}
            id2ent_2 = {}
        base_loader = _DummyLoader()

    candidates_dict = {}
    prev_rankings = {}


    for rnd in range(args.rounds):
        labeled_now = len(labeled_pairs)
        if labeled_now >= target or len(unlabeled_src) == 0:
            break
        ratio = labeled_now / float(total)
        print(f"\nRound {rnd} | Labeled: {labeled_now} ({ratio:.1%}) | "
              f"Unlabeled: {len(unlabeled_src)} | Negative: {len(negative_pairs)}")

        if rnd == 0:
            budget = min(init_budget, len(unlabeled_src))
            indices = rng.choice(len(unlabeled_src), budget, replace=False)
            query = [unlabeled_src[i] for i in indices]
            h1, h5, h10, mrr = 0., 0., 0., 0.

        else:
            if args.model == "lightea" and HAS_LIGHTEA:
                if not getattr(args, '_lightea_model', None):
                    args._lightea_model = LightEAWrapper(
                        args.data_path, ent_dim=args.ent_dim,
                        rel_dim=args.rel_dim, seed=args.seed,
                    )
                args._lightea_model.train(np.array(labeled_pairs, dtype=np.int64))
                vec = args._lightea_model.get_embeddings()

            elif args.model == "gcn_align" and HAS_GCNALIGN:
                if not getattr(args, '_gcnalign_model', None):
                    args._gcnalign_model = GCNAlignWrapper(
                        args.data_path, se_dim=args.se_dim, ae_dim=args.ae_dim,
                        beta=args.gcn_beta, learning_rate=args.gcn_lr,
                        gamma=args.gcn_gamma, k=args.gcn_k, seed=args.seed,
                    )
                    args._gcnalign_model.train(
                        np.array(labeled_pairs, dtype=np.int64),
                        epochs=args.gcn_epochs, verbose=not args.quiet,
                    )
                else:
                    args._gcnalign_model.train_al(
                        np.array(labeled_pairs, dtype=np.int64),
                        epochs=args.gcn_epochs_al, verbose=not args.quiet,
                    )
                vec = args._gcnalign_model.get_embeddings()

            else:
                K.clear_session()
                tf.compat.v1.reset_default_graph()
                set_all_seeds(args.seed + rnd)
                cfg = tf.compat.v1.ConfigProto()
                cfg.gpu_options.allow_growth = True
                sess = tf.compat.v1.Session(config=cfg)
                try:
                    K.set_session(sess)
                except Exception:
                    pass
                train_model, feature_model = build_duala_models(
                    node_size=node_size, rel_size=rel_size,
                    triple_size=triple_size,
                    node_hidden=args.node_hidden,
                    dropout_rate=args.dropout_rate,
                    gamma=args.hinge_gamma, lr=args.lr, depth=args.depth,
                )
                train_pair = np.array(labeled_pairs, dtype=np.int32)
                for _ in trange(args.epochs_first, desc=f"R{rnd}",
                                disable=args.quiet):
                    idx = rng.permutation(len(train_pair))
                    for bi in range(len(train_pair) // args.batch_size + 1):
                        pairs = train_pair[idx][bi*args.batch_size:
                                                (bi+1)*args.batch_size]
                        if len(pairs) == 0:
                            continue
                        inputs = [np.expand_dims(x, 0) for x in
                                  [adj_edges, r_index, r_val,
                                   rel_matrix, ent_matrix, pairs]]
                        train_model.train_on_batch(inputs)
                vec = predict_embeddings(
                    feature_model, adj_edges=adj_edges, r_index=r_index,
                    r_val=r_val, rel_matrix=rel_matrix, ent_matrix=ent_matrix,
                )

            rest_pairs = np.array(
                [(s, gold[s]) for s in unlabeled_src if s in gold],
                dtype=np.int32,
            )
            if len(rest_pairs) > 0:
                if args.model == "lightea" and HAS_LIGHTEA:
                    h1, h5, h10, mrr = args._lightea_model.evaluate_original(
                        rest_pairs, negative_pairs=negative_pairs
                    )
                elif args.model == "gcn_align" and HAS_GCNALIGN:
                    h1, h5, h10, mrr = args._gcnalign_model.evaluate_original(
                        rest_pairs, negative_pairs=negative_pairs
                    )
                else:
                    evaluater = evaluate(rest_pairs)
                    Lv = np.array([vec[int(e)] for e in rest_pairs[:, 0]])
                    Rv = np.array([vec[int(e)] for e in rest_pairs[:, 1]])
                    Lv = Lv / (np.linalg.norm(Lv, axis=-1, keepdims=True) + 1e-8)
                    Rv = Rv / (np.linalg.norm(Rv, axis=-1, keepdims=True) + 1e-8)
                    h1, h5, h10, mrr = metrics(evaluater.test(Lv, Rv))
                    print(f"[Eval] H@1={h1:.4f} H@5={h5:.4f} MRR={mrr:.4f}")
            else:
                h1, h5, h10, mrr = 0., 0., 0., 0.

            budget = min(step_budget, len(unlabeled_src))

            pool_loader = PoolDataLoader.from_base(
                base_loader, unlabeled_src, unlabeled_tgt
            )
            cb = CandidateBuilder(pool_loader, top_k=args.cand_topk, k_csls=10)
            cb.build_candidates_by_structure(vec)
            candidates_dict = cb.get_all_candidates()
            reverse_candidates = cb.get_reverse_candidates()

            candidate_scores = {}
            vec_normed = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)
            for src in unlabeled_src:
                cands = candidates_dict.get(src, [])
                if cands:
                    sims = np.dot(vec_normed[cands], vec_normed[src])
                    candidate_scores[src] = sims

            curr_rankings = {}
            _inst_k = args.inst_topk
            for src in unlabeled_src:
                cands = candidates_dict.get(src, [])
                cs = candidate_scores.get(src)
                if cands and cs is not None:
                    cs_arr = np.asarray(cs, dtype=np.float64)
                    top_k = min(_inst_k, len(cands))
                    top_indices = np.argsort(-cs_arr)[:top_k]
                    curr_rankings[src] = [cands[j] for j in top_indices]

            strategy = FunnelStrategy(
                candidates_dict=candidates_dict,
                candidate_scores=candidate_scores,
                adj_dict_src=adj_dict_src,
                adj_dict_tgt=adj_dict_tgt,
                reverse_candidates=reverse_candidates,
                embeddings=vec_normed,
                method=args.strategy,
                alpha=args.alpha,
                gamma=args.funnel_gamma,
                cov_eta=args.cov_eta,
                funnel_topk=args.funnel_topk,
                inst_topk=args.inst_topk,
                inst_lambda=args.inst_lambda,
                prev_rankings=prev_rankings,
                prefilter_ratio=args.prefilter_ratio,
                walk_steps=args.walk_steps,
                walk_restart=args.walk_restart,
                seed=args.seed,
                verbose=not args.quiet,
            )
            query = strategy.select(unlabeled_src, budget,
                                    unlabeled_tgt=unlabeled_tgt)
            prev_rankings = curr_rankings

        rnd_dir = os.path.join(out_dir, f"round_{rnd}")
        ensure_dir(rnd_dir)

        if args.simulate_oracle:
            labels = {src: gold.get(src, -1) for src in query}
        else:
            rows = [{"source_id": int(src)} for src in query]
            write_jsonl(os.path.join(rnd_dir, "to_label.jsonl"), rows)
            lp = os.path.join(rnd_dir, "labels.tsv")
            labels = read_labels_tsv(lp)
            if not labels:
                break

        if candidates_dict:
            for src in query:
                src = int(src)
                correct_tgt = gold.get(src, -1)
                if correct_tgt == -1:
                    continue
                cands = candidates_dict.get(src, [])
                if cands and correct_tgt not in cands:
                    if src not in negative_pairs:
                        negative_pairs[src] = []
                    for c in cands:
                        if c not in negative_pairs[src]:
                            negative_pairs[src].append(c)

        new_pairs = [(int(s), int(t)) for s, t in labels.items()
                     if int(t) != -1 and s in gold]
        max_can_add = target - len(labeled_pairs)
        if len(new_pairs) > max_can_add:
            new_pairs = new_pairs[:max_can_add]
        labeled_pairs.extend(new_pairs)

        queried_src_set = set(int(s) for s in query)
        labeled_tgt_set = set(p[1] for p in new_pairs)
        unlabeled_src = [s for s in unlabeled_src if s not in queried_src_set]
        unlabeled_tgt = [t for t in unlabeled_tgt if t not in labeled_tgt_set]

        append_jsonl(prog_path, {
            "round": rnd,
            "labeled": len(labeled_pairs),
            "ratio": len(labeled_pairs) / total,
            "added": len(new_pairs),
            "hits1": h1, "hits5": h5, "hits10": h10, "mrr": mrr,
            "seed": args.seed, "model": args.model,
            "dataset": derive_dataset_name(args.data_path),
        })
        print(f"[INFO] +{len(new_pairs)} | Total: {len(labeled_pairs)} "
              f"({len(labeled_pairs)/total:.1%})")


    if labeled_pairs and len(unlabeled_src) > 0:
        print(f"\n[Final Evaluation]")
        if args.model == "lightea" and HAS_LIGHTEA:
            if not getattr(args, '_lightea_model', None):
                args._lightea_model = LightEAWrapper(
                    args.data_path, ent_dim=args.ent_dim,
                    rel_dim=args.rel_dim, seed=args.seed,
                )
            args._lightea_model.train(np.array(labeled_pairs, dtype=np.int64))
            vec = args._lightea_model.get_embeddings()
        elif args.model == "gcn_align" and HAS_GCNALIGN:
            if not getattr(args, '_gcnalign_model', None):
                args._gcnalign_model = GCNAlignWrapper(
                    args.data_path, se_dim=args.se_dim, ae_dim=args.ae_dim,
                    beta=args.gcn_beta, learning_rate=args.gcn_lr,
                    gamma=args.gcn_gamma, k=args.gcn_k, seed=args.seed,
                )
            args._gcnalign_model.train(
                np.array(labeled_pairs, dtype=np.int64),
                epochs=args.gcn_epochs, verbose=not args.quiet,
            )
            vec = args._gcnalign_model.get_embeddings()
        else:
            K.clear_session()
            tf.compat.v1.reset_default_graph()
            set_all_seeds(args.seed)
            cfg = tf.compat.v1.ConfigProto()
            cfg.gpu_options.allow_growth = True
            sess = tf.compat.v1.Session(config=cfg)
            try:
                K.set_session(sess)
            except Exception:
                pass
            train_model, feature_model = build_duala_models(
                node_size=node_size, rel_size=rel_size,
                triple_size=triple_size,
                node_hidden=args.node_hidden, dropout_rate=args.dropout_rate,
                gamma=args.hinge_gamma, lr=args.lr, depth=args.depth,
            )
            train_pair = np.array(labeled_pairs, dtype=np.int32)
            for _ in trange(args.epochs_first, desc="Final", disable=args.quiet):
                idx = rng.permutation(len(train_pair))
                for bi in range(len(train_pair) // args.batch_size + 1):
                    pairs = train_pair[idx][bi*args.batch_size:
                                            (bi+1)*args.batch_size]
                    if len(pairs) == 0:
                        continue
                    inputs = [np.expand_dims(x, 0) for x in
                              [adj_edges, r_index, r_val,
                               rel_matrix, ent_matrix, pairs]]
                    train_model.train_on_batch(inputs)
            vec = predict_embeddings(
                feature_model, adj_edges=adj_edges, r_index=r_index,
                r_val=r_val, rel_matrix=rel_matrix, ent_matrix=ent_matrix,
            )

        rest_pairs = np.array(
            [(s, gold[s]) for s in unlabeled_src if s in gold], dtype=np.int32
        )
        if len(rest_pairs) > 0:
            if args.model == "lightea" and HAS_LIGHTEA:
                h1, h5, h10, mrr = args._lightea_model.evaluate_original(
                    rest_pairs, negative_pairs=negative_pairs
                )
            elif args.model == "gcn_align" and HAS_GCNALIGN:
                h1, h5, h10, mrr = args._gcnalign_model.evaluate_original(
                    rest_pairs, negative_pairs=negative_pairs
                )
            else:
                src_ids = rest_pairs[:, 0]
                tgt_ids = rest_pairs[:, 1]
                Lv = np.array([vec[int(e)] for e in src_ids])
                Rv = np.array([vec[int(e)] for e in tgt_ids])
                Lv = Lv / (np.linalg.norm(Lv, axis=-1, keepdims=True) + 1e-8)
                Rv = Rv / (np.linalg.norm(Rv, axis=-1, keepdims=True) + 1e-8)
                h1, h5, h10, mrr = evaluate_with_negative_info(
                    Lv, Rv, src_ids, tgt_ids, negative_pairs, k=10
                )

            print(f"[FINAL] H@1={h1:.4f} H@5={h5:.4f} H@10={h10:.4f} MRR={mrr:.4f}")
            append_jsonl(prog_path, {
                "round": -1, "final": True,
                "labeled": len(labeled_pairs),
                "ratio": len(labeled_pairs) / total,
                "hits1": h1, "hits5": h5, "hits10": h10, "mrr": mrr,
                "seed": args.seed, "model": args.model,
                "dataset": derive_dataset_name(args.data_path),
            })


if __name__ == "__main__":
    main()
