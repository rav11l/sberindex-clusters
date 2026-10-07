"""Правила построения рёбер: «степень экономической близости МО».

attr   — близость по признакам снимка: kNN-граф с самонастраивающимся гауссовым ядром
         (Zelnik-Manor, Perona, 2004): w_ij = exp(-d_ij² / (σ_i σ_j)), σ_i — расстояние до k-го соседа.
comove — совместное движение: корреляция месячных изменений расходов по шести рядам за 12 месяцев
         после вычитания общего для страны движения; k соседей с наибольшей положительной корреляцией.
hybrid — взвешенная сумма двух графов после нормировки каждого на средний взвешенный degree.

Все графы неориентированные, симметризация по максимуму: ребро есть, если хотя бы один из
концов входит в k ближайших соседей другого.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def _symmetrize(W: sparse.csr_matrix) -> sparse.csr_matrix:
    W = W.maximum(W.T).tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    return W


def knn_attr(Z: pd.DataFrame, k: int) -> sparse.csr_matrix:
    X = Z.to_numpy()
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    sigma = np.maximum(dist[:, -1], 1e-9)
    rows = np.repeat(np.arange(len(X)), k)
    w = np.exp(-dist.ravel() ** 2 / (sigma[rows] * sigma[idx.ravel()]))
    W = sparse.csr_matrix((w, (rows, idx.ravel())), shape=(len(X), len(X)))
    return _symmetrize(W)


def knn_comove(M: pd.DataFrame, k: int) -> sparse.csr_matrix:
    A = M.to_numpy()
    C = (A @ A.T) / A.shape[1]
    np.fill_diagonal(C, -np.inf)
    idx = np.argpartition(-C, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(len(A)), k)
    w = np.clip(C[rows, idx.ravel()], 0, None)
    W = sparse.csr_matrix((w, (rows, idx.ravel())), shape=C.shape)
    return _symmetrize(W)


def _norm(W: sparse.csr_matrix) -> sparse.csr_matrix:
    return W / max(np.asarray(W.sum(axis=1)).mean(), 1e-12)


def hybrid(Wa: sparse.csr_matrix, Wc: sparse.csr_matrix, alpha: float) -> sparse.csr_matrix:
    return _symmetrize((alpha * _norm(Wa) + (1 - alpha) * _norm(Wc)).tocsr())


def build(rule: str, Z: pd.DataFrame, M: pd.DataFrame, k: int, alpha: float = 0.5) -> sparse.csr_matrix:
    if rule == "attr":
        return knn_attr(Z, k)
    if rule == "comove":
        return knn_comove(M.loc[Z.index], k)
    if rule == "hybrid":
        return hybrid(knn_attr(Z, k), knn_comove(M.loc[Z.index], k), alpha)
    raise ValueError(rule)


def summary(W: sparse.csr_matrix) -> dict:
    import scipy.sparse.csgraph as cg
    n_comp, _ = cg.connected_components(W, directed=False)
    deg = np.diff(W.indptr)
    return {"nodes": W.shape[0], "edges": int(W.nnz // 2), "mean_degree": float(deg.mean()),
            "components": int(n_comp), "total_weight": float(W.sum() / 2)}
