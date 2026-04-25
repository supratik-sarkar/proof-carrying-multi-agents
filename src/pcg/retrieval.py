"""
Retrieval over a per-example evidence pool.

Two index types:
    - `BM25Index`:  always available, dependency-free path to a working pipeline
    - `DenseIndex`: sentence-transformers embedding + FAISS exact NN, much
                    stronger but requires a model download

Each Prover branch picks one or the other (or one of each) so that the k
branches in Theorem 2 are *operationally* independent: different retrievers
satisfy the (delta, kappa)-independence predicate via different tool steps.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from pcg.datasets.base import EvidenceItem


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25Index:
    """Plain Okapi BM25, sufficient for paragraph-level retrieval over the
    small per-example pools we work with (5-50 docs).

    Why not always use a library? Two reasons:
      1. Determinism — bit-stable scores across machines is important for
         reproducibility, and pip wheels of `rank_bm25` change tokenization
         silently between releases.
      2. Visibility — the verifier can replay the exact scoring formula.

    Standard BM25 with k1=1.5, b=0.75. Stop words NOT removed (they help
    multi-hop bridge questions).
    """

    docs_tokens: list[list[str]]
    df: Counter
    avgdl: float
    n: int
    items: tuple[EvidenceItem, ...]
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, items: Sequence[EvidenceItem], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        toks = [_tokenize(it.text) for it in items]
        n = len(toks)
        df: Counter = Counter()
        for d in toks:
            df.update(set(d))
        avgdl = (sum(len(d) for d in toks) / n) if n else 0.0
        return cls(
            docs_tokens=toks, df=df, avgdl=avgdl, n=n,
            items=tuple(items), k1=k1, b=b,
        )

    def _idf(self, term: str) -> float:
        # BM25+ smoothing-friendly IDF; never negative.
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_idx: int) -> float:
        if self.n == 0 or doc_idx >= self.n:
            return 0.0
        q_terms = _tokenize(query)
        doc = self.docs_tokens[doc_idx]
        if not doc:
            return 0.0
        tf = Counter(doc)
        dl = len(doc)
        s = 0.0
        for q in q_terms:
            if q not in tf:
                continue
            num = tf[q] * (self.k1 + 1)
            den = tf[q] + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1.0))
            s += self._idf(q) * num / den
        return s

    def search(self, query: str, top_k: int = 5) -> list[tuple[EvidenceItem, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.n)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.items[i], s) for i, s in scores[:top_k]]


# ---------------------------------------------------------------------------
# Dense retrieval (lazy: model loads on first use)
# ---------------------------------------------------------------------------


@dataclass
class DenseIndex:
    """Sentence-transformer embedding + cosine similarity.

    Uses `sentence-transformers/all-MiniLM-L6-v2` by default. Lazy: the
    model is loaded only on the first `build` or `search` call.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    items: tuple[EvidenceItem, ...] = ()
    _embeddings: object = None     # numpy.ndarray, set in build()
    _model: object = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)

    @classmethod
    def build(
        cls,
        items: Sequence[EvidenceItem],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "DenseIndex":
        idx = cls(model_name=model_name, items=tuple(items))
        idx._load_model()
        if not items:
            import numpy as np
            idx._embeddings = np.zeros((0, 384), dtype="float32")
            return idx
        texts = [it.text for it in items]
        embs = idx._model.encode(  # type: ignore[union-attr]
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        idx._embeddings = embs
        return idx

    def search(self, query: str, top_k: int = 5) -> list[tuple[EvidenceItem, float]]:
        import numpy as np
        if not self.items:
            return []
        self._load_model()
        q_emb = self._model.encode(  # type: ignore[union-attr]
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        sims = (self._embeddings @ q_emb)  # type: ignore[operator]
        order = np.argsort(-sims)[:top_k]
        return [(self.items[int(i)], float(sims[int(i)])) for i in order]


# ---------------------------------------------------------------------------
# Hybrid retrieval — used by Prover branches that need diverse retrievers
# ---------------------------------------------------------------------------


def hybrid_search(
    indices: Iterable[BM25Index | DenseIndex],
    query: str,
    top_k: int = 5,
    weights: Sequence[float] | None = None,
) -> list[tuple[EvidenceItem, float]]:
    """Reciprocal-rank-fused hybrid retrieval. Used to construct branches
    whose tool steps differ (BM25 vs. dense) so that they qualify as
    (delta, kappa)-independent.

    Standard RRF formula: sum_i 1 / (rank_i + 60). Weights are multiplied
    onto each index's contribution.
    """
    indices_list = list(indices)
    weights = weights or [1.0] * len(indices_list)
    if len(weights) != len(indices_list):
        raise ValueError("weights must align with indices")

    rrf: dict[str, float] = {}
    item_by_id: dict[str, EvidenceItem] = {}
    for w, idx in zip(weights, indices_list):
        results = idx.search(query, top_k=top_k * 2)
        for r, (item, _) in enumerate(results):
            rrf[item.id] = rrf.get(item.id, 0.0) + w / (r + 60)
            item_by_id[item.id] = item

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return [(item_by_id[i], s) for i, s in ranked[:top_k]]
