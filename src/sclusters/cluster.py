"""Методы кластеризации и метрики качества.

leiden   — Leiden (Traag, Waltman, van Eck, 2019), RB-конфигурационная модель с параметром
           разрешения; число кластеров выбирает сам метод. Leiden работает только со
           структурой графа: признаки МО попадают в него через веса рёбер attr и hybrid.
spectral — спектральная кластеризация на той же матрице смежности, число кластеров = как у Leiden.
kmeans   — k-средних на признаках без графа, то же число кластеров. Контроль: что даёт сеть
           сверх признаков.
"""
from __future__ import annotations

import warnings

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)


def to_igraph(W: sparse.csr_matrix) -> ig.Graph:
    U = sparse.triu(W, k=1).tocoo()
    g = ig.Graph(n=W.shape[0], edges=list(zip(U.row.tolist(), U.col.tolist())), directed=False)
    g.es["weight"] = U.data.tolist()
    return g


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """0 — самый крупный кластер. Делает номера воспроизводимыми между запусками."""
    s = pd.Series(labels)
    order = s.value_counts().sort_index().sort_values(ascending=False, kind="stable").index
    return s.map({c: i for i, c in enumerate(order)}).to_numpy()


def merge_small(labels: np.ndarray, W: sparse.csr_matrix, Z: np.ndarray, min_size: int) -> np.ndarray:
    """Кластеры меньше min_size переносятся в кластер, с которым у узла больше всего веса рёбер;
    без рёбер к крупным кластерам — к ближайшему центроиду по признакам."""
    labels = labels.copy()
    while True:
        sizes = pd.Series(labels).value_counts()
        small = sizes[sizes < min_size].index
        big = sizes[sizes >= min_size].index
        if len(small) == 0 or len(big) == 0:
            break
        cent = {c: Z[labels == c].mean(axis=0) for c in big}
        for i in np.where(np.isin(labels, small))[0]:
            row = W.getrow(i)
            w = pd.Series(row.data, index=labels[row.indices]).groupby(level=0).sum()
            w = w[w.index.isin(big)]
            if len(w):
                labels[i] = w.idxmax()
            else:
                labels[i] = min(big, key=lambda c: np.sum((Z[i] - cent[c]) ** 2))
    return relabel_by_size(labels)


def leiden(W: sparse.csr_matrix, resolution: float = 1.0, seed: int = 0, n_restarts: int = 10,
           initial: np.ndarray | None = None) -> np.ndarray:
    g = to_igraph(W)
    best, best_q = None, -np.inf
    for r in range(n_restarts):
        kw = dict(weights="weight", resolution_parameter=resolution, seed=seed + r)
        if initial is not None:
            kw["initial_membership"] = list(map(int, initial))
        p = la.find_partition(g, la.RBConfigurationVertexPartition, **kw)
        if p.quality() > best_q:
            best, best_q = np.array(p.membership), p.quality()
        if initial is not None:
            break  # с заданным начальным разбиением запуск детерминирован по зерну; одного хватает
    return best


def spectral(W: sparse.csr_matrix, k: int, seed: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="cluster_qr",
                                random_state=seed)
        return sc.fit_predict(W)


def kmeans(Z: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(Z)


def run(method: str, W, Z: np.ndarray, k: int | None, cfg: dict, seed: int, initial=None) -> np.ndarray:
    c = cfg["cluster"]
    if method == "leiden":
        lab = leiden(W, c["resolution"], seed, c["n_restarts"], initial)
    elif method == "spectral":
        lab = spectral(W, k, seed)
    elif method == "kmeans":
        lab = kmeans(Z, k, seed)
    else:
        raise ValueError(method)
    return merge_small(np.asarray(lab), W, Z, c["min_size"])


# ---------- метрики ----------

def modularity(W: sparse.csr_matrix, labels: np.ndarray) -> float:
    g = to_igraph(W)
    return float(g.modularity(list(map(int, labels)), weights="weight"))


def conductance(W: sparse.csr_matrix, labels: np.ndarray) -> float:
    """Средняя по кластерам доля веса рёбер, уходящих наружу: cut / min(vol, vol_rest). Меньше — лучше."""
    deg = np.asarray(W.sum(axis=1)).ravel()
    vol_all = deg.sum()
    out = []
    for c in np.unique(labels):
        m = labels == c
        vol = deg[m].sum()
        internal = W[m][:, m].sum()
        cut = vol - internal
        out.append(cut / max(min(vol, vol_all - vol), 1e-12))
    return float(np.mean(out))


def quality(W, Z: np.ndarray, labels: np.ndarray) -> dict:
    k = len(np.unique(labels))
    res = {"k": k, "modularity": modularity(W, labels), "conductance": conductance(W, labels)}
    if k > 1:
        res.update({"silhouette": float(silhouette_score(Z, labels)),
                    "calinski_harabasz": float(calinski_harabasz_score(Z, labels)),
                    "davies_bouldin": float(davies_bouldin_score(Z, labels))})
    sizes = np.bincount(labels)
    res["largest_share"] = float(sizes.max() / sizes.sum())
    return res


def ari(a, b) -> float:
    return float(adjusted_rand_score(a, b))
