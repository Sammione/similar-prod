"""Thin FAISS wrapper (cosine similarity via inner product on L2-normalised vectors)."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, dim: int, path):
        self.dim = dim
        self.path = Path(path)
        if self.path.exists():
            self.index = faiss.read_index(str(self.path))
            if self.index.d != dim:
                raise ValueError(f"{self.path} has dim {self.index.d}, model gives {dim}. "
                                 "You changed the model: delete the data dir and re-index.")
        else:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def __len__(self):
        return int(self.index.ntotal)

    def add(self, ids, vecs):
        self.index.add_with_ids(np.ascontiguousarray(vecs, dtype="float32"),
                                np.asarray(ids, dtype="int64"))

    def search(self, vecs, k: int):
        vecs = np.ascontiguousarray(np.atleast_2d(vecs), dtype="float32")
        if self.index.ntotal == 0:
            n = vecs.shape[0]
            return np.zeros((n, 0), "float32"), np.zeros((n, 0), "int64")
        return self.index.search(vecs, min(k, self.index.ntotal))

    def get(self, ids) -> np.ndarray:
        out = []
        for i in ids:
            try:
                out.append(self.index.reconstruct(int(i)))
            except RuntimeError:
                pass
        return np.stack(out) if out else np.zeros((0, self.dim), "float32")

    def remove(self, ids):
        if len(ids):
            self.index.remove_ids(np.asarray(ids, dtype="int64"))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.path))
