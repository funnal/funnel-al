from typing import List


class PoolDataLoader(object):
    def __init__(self, source_entities, target_entities, id2ent_1=None, id2ent_2=None):
        self.source_entities = list(source_entities)
        self.target_entities = list(target_entities)
        self.id2ent_1 = dict(id2ent_1 or {})
        self.id2ent_2 = dict(id2ent_2 or {})

    @classmethod
    def from_base(cls, base_loader, source_entities, target_entities):
        id2ent_1 = getattr(base_loader, "id2ent_1", {}) or {}
        id2ent_2 = getattr(base_loader, "id2ent_2", {}) or {}
        return cls(source_entities, target_entities, id2ent_1=id2ent_1, id2ent_2=id2ent_2)

    def get_source_entities(self):
        return self.source_entities

    def get_target_entities(self):
        return self.target_entities
