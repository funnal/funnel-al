import numpy as np
import os
import pickle
from collections import defaultdict


class KGDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.ent2id_1 = {}
        self.ent2id_2 = {}
        self.id2ent_1 = {}
        self.id2ent_2 = {}
        self.triples_1 = []
        self.triples_2 = []
        self.ref_ent_pairs = []
        self.adj_1 = defaultdict(set)
        self.adj_2 = defaultdict(set)
        self.node_size = 0
        self.rel_size = 0
        self._load_all()

    def _load_all(self):
        cache_path = os.path.join(self.data_path, "data_cache.pkl")
        if os.path.exists(cache_path):
            self._load_from_cache(cache_path)
        else:
            self._load_entities()
            self._load_triples()
            self._load_ref_pairs()
            self._build_adjacency()
            self._save_to_cache(cache_path)

    def _load_entities(self):
        with open(os.path.join(self.data_path, "ent_ids_1"), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    ent_id, ent_uri = int(parts[0]), parts[1]
                    self.ent2id_1[ent_uri] = ent_id
                    self.id2ent_1[ent_id] = ent_uri

        with open(os.path.join(self.data_path, "ent_ids_2"), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    ent_id, ent_uri = int(parts[0]), parts[1]
                    self.ent2id_2[ent_uri] = ent_id
                    self.id2ent_2[ent_id] = ent_uri

        self.node_size = max(
            max(self.ent2id_1.values()) if self.ent2id_1 else 0,
            max(self.ent2id_2.values()) if self.ent2id_2 else 0
        ) + 1

    def _load_triples(self):
        rel_set = set()
        with open(os.path.join(self.data_path, "triples_1"), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                    self.triples_1.append((h, r, t))
                    rel_set.add(r)

        with open(os.path.join(self.data_path, "triples_2"), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                    self.triples_2.append((h, r, t))
                    rel_set.add(r)

        self.rel_size = max(rel_set) + 1 if rel_set else 0

    def _load_ref_pairs(self):
        with open(os.path.join(self.data_path, "ref_ent_ids"), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    e1, e2 = int(parts[0]), int(parts[1])
                    self.ref_ent_pairs.append((e1, e2))
        self.ref_ent_pairs = np.array(self.ref_ent_pairs)

    def _build_adjacency(self):
        for h, r, t in self.triples_1:
            self.adj_1[h].add(t)
            self.adj_1[t].add(h)
        for h, r, t in self.triples_2:
            self.adj_2[h].add(t)
            self.adj_2[t].add(h)

    def _save_to_cache(self, cache_path):
        data = {
            "ent2id_1": self.ent2id_1,
            "ent2id_2": self.ent2id_2,
            "id2ent_1": self.id2ent_1,
            "id2ent_2": self.id2ent_2,
            "triples_1": self.triples_1,
            "triples_2": self.triples_2,
            "ref_ent_pairs": self.ref_ent_pairs,
            "adj_1": dict(self.adj_1),
            "adj_2": dict(self.adj_2),
            "node_size": self.node_size,
            "rel_size": self.rel_size,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def _load_from_cache(self, cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        self.ent2id_1 = data["ent2id_1"]
        self.ent2id_2 = data["ent2id_2"]
        self.id2ent_1 = data["id2ent_1"]
        self.id2ent_2 = data["id2ent_2"]
        self.triples_1 = data["triples_1"]
        self.triples_2 = data["triples_2"]
        self.ref_ent_pairs = data["ref_ent_pairs"]
        self.adj_1 = defaultdict(set, {k: set(v) for k, v in data["adj_1"].items()})
        self.adj_2 = defaultdict(set, {k: set(v) for k, v in data["adj_2"].items()})
        self.node_size = data["node_size"]
        self.rel_size = data["rel_size"]

    def get_source_entities(self):
        return list(self.ent2id_1.values())

    def get_target_entities(self):
        return list(self.ent2id_2.values())
