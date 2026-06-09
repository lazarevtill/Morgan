"""Tiny in-memory BM25. Sufficient for a single user's memory volume in Phase 1; swappable for a
real index later without touching callers."""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Bm25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._docs: dict[str, list[str]] = {}
        self._df: Counter[str] = Counter()

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._docs:
            for term in set(self._docs[doc_id]):
                self._df[term] -= 1
        tokens = _tokenize(text)
        self._docs[doc_id] = tokens
        for term in set(tokens):
            self._df[term] += 1

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        q_terms = _tokenize(query)
        if not q_terms or not self._docs:
            return []
        n = len(self._docs)
        avgdl = sum(len(d) for d in self._docs.values()) / n
        scores: dict[str, float] = {}
        for doc_id, tokens in self._docs.items():
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                df = max(self._df.get(term, 0), 1)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                freq = tf[term]
                denom = freq + self._k1 * (1 - self._b + self._b * dl / avgdl)
                score += idf * (freq * (self._k1 + 1)) / denom
            if score > 0:
                scores[doc_id] = score
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]
