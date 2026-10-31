"""Sparse lexical retrievers: TF-IDF, BM25, and reciprocal-rank-fusion hybrid.

All retrievers are fitted on a *fitting corpus* (statistics only), then index an arbitrary
document set. This keeps IDF/length statistics from ever touching evaluation text.
"""
from __future__ import annotations

import re

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from . import config

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, max_features=config.TFIDF_MAX_FEATURES, ngram=config.TFIDF_NGRAM):
        self.vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                   sublinear_tf=True, ngram_range=ngram, max_features=max_features)
        self.X = None

    def fit(self, fit_texts):
        self.vec.fit(fit_texts)
        return self

    def index(self, docs):
        self.X = self.vec.transform(docs).tocsr()
        return self

    def scores(self, query: str, rows=None) -> np.ndarray:
        q = self.vec.transform([query])
        X = self.X if rows is None else self.X[rows]
        return np.asarray((X @ q.T).todense()).ravel()


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1=config.BM25_K1, b=config.BM25_B):
        self.k1, self.b = k1, b
        self.cv = CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
        self.W = None

    def fit(self, fit_texts):
        C = self.cv.fit_transform(fit_texts)
        n = C.shape[0]
        df = np.asarray((C > 0).sum(axis=0)).ravel()
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        self.avgdl = float(C.sum() / n)
        return self

    def index(self, docs):
        C = self.cv.transform(docs).tocsr().astype(np.float64)
        dl = np.asarray(C.sum(axis=1)).ravel()
        C = C.tocoo()
        denom = C.data + self.k1 * (1 - self.b + self.b * dl[C.row] / self.avgdl)
        data = self.idf[C.col] * C.data * (self.k1 + 1) / denom
        self.W = sp.csr_matrix((data, (C.row, C.col)), shape=C.shape)
        return self

    def scores(self, query: str, rows=None) -> np.ndarray:
        q = self.cv.transform([query])
        q.data[:] = 1.0  # binary query term weights
        W = self.W if rows is None else self.W[rows]
        return np.asarray((W @ q.T).todense()).ravel()


class RRFHybrid:
    """Reciprocal rank fusion of several retrievers that share the same indexed documents."""
    name = "hybrid_rrf"

    def __init__(self, parts, k=config.RRF_K):
        self.parts, self.k = parts, k

    def scores(self, query: str, rows=None) -> np.ndarray:
        total = None
        for p in self.parts:
            s = p.scores(query, rows)
            order = np.lexsort((np.arange(len(s)), -s))
            rank = np.empty(len(s)); rank[order] = np.arange(1, len(s) + 1)
            contrib = 1.0 / (self.k + rank)
            total = contrib if total is None else total + contrib
        return total
