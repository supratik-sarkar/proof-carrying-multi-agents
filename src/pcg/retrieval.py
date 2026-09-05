"""
Retrieval over a per-example evidence pool.

Two index types:
    - `BM25Index`:  always available, dependency-free path to a working pipeline
    - `DenseIndex`: sentence-transformers embedding + FAISS exact NN, much
                    stronger but requires a model download

Each Prover branch picks one or the other, or a hybrid of both, so that the k
branches expose operationally separated retrieval paths. Different retrievers
reduce shared tool dependence and support the (delta, kappa, gamma)-separation
diagnostics used by the redundancy analysis.
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
    whose tool steps differ, for example BM25 versus dense retrieval, so that
    they support (delta, kappa, gamma)-separation diagnostics.

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



# ---------------------------------------------------------------------------
# DenseIndex (sentence-transformers)
# ---------------------------------------------------------------------------


_DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_dense_model_cache = {}


def _load_dense_model():
    name = _DENSE_MODEL_NAME
    if name not in _dense_model_cache:
        from sentence_transformers import SentenceTransformer
        _dense_model_cache[name] = SentenceTransformer(name)
    return _dense_model_cache[name]


@dataclass
class DenseIndex:
    """Sentence-Transformers all-MiniLM-L6-v2 cosine-similarity retrieval.

    Operationally separated from BM25: dense embeddings capture paraphrastic
    similarity that BM25 cannot. Replayable because the model is deterministic
    given the same input and weights.
    """

    items: tuple[EvidenceItem, ...]
    doc_embeddings: object   # numpy array (n, d)

    @classmethod
    def build(cls, items):
        import numpy as np
        model = _load_dense_model()
        texts = [it.title + ": " + it.text for it in items] if items else []
        if not texts:
            return cls(items=tuple(items), doc_embeddings=np.zeros((0, 384), dtype="float32"))
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return cls(items=tuple(items), doc_embeddings=embs)

    def search(self, query: str, top_k: int = 4):
        import numpy as np
        if len(self.items) == 0:
            return []
        model = _load_dense_model()
        q = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.doc_embeddings @ q
        idxs = np.argsort(-sims)[:top_k]
        return [(self.items[i], float(sims[i])) for i in idxs]


# ---------------------------------------------------------------------------
# HybridIndex (BM25 + Dense, score fusion)
# ---------------------------------------------------------------------------


@dataclass
class HybridIndex:
    """Reciprocal-rank fusion of BM25 + Dense rankings."""

    bm25: BM25Index
    dense: DenseIndex
    rrf_k: int = 60

    @classmethod
    def build(cls, items):
        return cls(bm25=BM25Index.build(items), dense=DenseIndex.build(items))

    def search(self, query: str, top_k: int = 4):
        bm25_hits = self.bm25.search(query, top_k=max(top_k * 3, 10))
        dense_hits = self.dense.search(query, top_k=max(top_k * 3, 10))
        # RRF: score = sum over rankers of 1 / (k + rank)
        fused: dict = {}
        for rank, (item, _) in enumerate(bm25_hits):
            fused.setdefault(item.id, [item, 0.0])[1] += 1.0 / (self.rrf_k + rank)
        for rank, (item, _) in enumerate(dense_hits):
            fused.setdefault(item.id, [item, 0.0])[1] += 1.0 / (self.rrf_k + rank)
        ranked = sorted(fused.values(), key=lambda x: -x[1])[:top_k]
        return [(item, score) for item, score in ranked]


# ---------------------------------------------------------------------------
# ParaphraseBM25 (deterministic query paraphrase)
# ---------------------------------------------------------------------------


_QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "for", "with", "by", "to",
    "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "did", "does", "do", "had", "has", "have",
})


def _paraphrase_query(q: str) -> str:
    """Deterministic content-word-only paraphrase.

    Drops function words and reorders content tokens alphabetically. Same
    information content, different token ordering — exercises BM25 over a
    structurally different query without changing semantics.
    """
    toks = _tokenize(q)
    content = sorted({t for t in toks if t not in _QUERY_STOPWORDS and len(t) > 1})
    return " ".join(content) if content else q


@dataclass
class ParaphraseBM25Index:
    """BM25 over the same evidence pool, but the query is deterministically
    paraphrased before scoring."""

    bm25: BM25Index

    @classmethod
    def build(cls, items):
        return cls(bm25=BM25Index.build(items))

    def search(self, query: str, top_k: int = 4):
        return self.bm25.search(_paraphrase_query(query), top_k=top_k)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def build_retriever(kind: str, items):
    """Build a retriever by kind. All retrievers expose `.search(query, top_k)`
    returning a list of (EvidenceItem, score) tuples."""
    kind = (kind or "bm25").lower()
    if kind in ("bm25", "bm25_default"):
        return BM25Index.build(items)
    if kind in ("dense", "minilm", "sentence_transformer"):
        return DenseIndex.build(items)
    if kind in ("hybrid", "rrf", "bm25_dense_rrf"):
        return HybridIndex.build(items)
    if kind in ("bm25_with_query_paraphrase", "paraphrase_bm25", "paraphrase"):
        return ParaphraseBM25Index.build(items)
    raise ValueError(f"Unknown retriever kind: {kind!r}")
