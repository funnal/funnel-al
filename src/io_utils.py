import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


def write_jsonl(path: str, rows: Iterable[dict]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_tsv(path: str, rows: Iterable[Tuple[int, str]]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for k, v in rows:
            f.write(f"{k}\t{v}\n")


def read_labels_tsv(path: str) -> Dict[int, int]:
    labels = {}
    p = Path(path)
    if not p.exists():
        return labels
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            src = int(parts[0])
            tgt = int(parts[1])
            labels[src] = tgt
    return labels
