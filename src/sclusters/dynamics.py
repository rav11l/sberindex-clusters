"""Динамика кластеров: сопоставление между снимками, события, переходы МО.

Номер кластера Leiden на каждом снимке произволен. Кластеры соседних снимков сопоставляются
по коэффициенту Жаккара состава (венгерский алгоритм), сквозной номер «линии» наследуется,
если Жаккар не ниже порога. По матрице пересечений классифицируются события
(Greene, Doyle, Cunningham, 2010): продолжение, рождение, исчезновение, слияние, распад.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def jaccard(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Матрица Жаккара кластеров a (строки) и b (столбцы) по общим МО."""
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    ct = pd.crosstab(a, b)
    sa, sb = ct.sum(axis=1), ct.sum(axis=0)
    return ct / (sa.to_numpy()[:, None] + sb.to_numpy()[None, :] - ct)


def track(partitions: dict[pd.Timestamp, pd.Series], threshold: float = 0.3):
    """Сквозная нумерация кластеров. Возвращает (линии по снимкам, события)."""
    months = sorted(partitions)
    lines = {}
    first = partitions[months[0]]
    lines[months[0]] = first.copy()
    next_id = int(first.max()) + 1
    events = []
    for prev, cur in zip(months[:-1], months[1:]):
        a, b = lines[prev], partitions[cur]
        J = jaccard(a, b)
        r, c = linear_sum_assignment(-J.to_numpy())
        mapping = {}
        for i, j in zip(r, c):
            if J.iat[i, j] >= threshold:
                mapping[J.columns[j]] = J.index[i]
        for col in J.columns:
            if col not in mapping:
                mapping[col] = next_id
                next_id += 1
        lines[cur] = b.map(mapping)
        # события по матрице пересечений: значимая доля — не меньше порога в одну из сторон
        ct = pd.crosstab(a.loc[a.index.intersection(b.index)], b.loc[a.index.intersection(b.index)])
        share_from = ct.div(ct.sum(axis=1), axis=0)   # куда ушли МО кластера a
        share_to = ct.div(ct.sum(axis=0), axis=1)     # откуда пришли МО кластера b
        for i in J.index:
            heirs = [mapping[j] for j in J.columns if share_from.at[i, j] >= threshold]
            if len(heirs) == 0:
                events.append({"month": cur, "event": "исчезновение", "line": int(i), "with": ""})
            elif len(heirs) > 1:
                events.append({"month": cur, "event": "распад", "line": int(i), "with": ",".join(map(str, sorted(heirs)))})
        for j in J.columns:
            src = [i for i in J.index if share_to.at[i, j] >= threshold]
            if len(src) == 0:
                events.append({"month": cur, "event": "рождение", "line": int(mapping[j]), "with": ""})
            elif len(src) > 1:
                events.append({"month": cur, "event": "слияние", "line": int(mapping[j]), "with": ",".join(map(str, sorted(src)))})
    L = pd.DataFrame(lines)
    L.columns = [str(pd.Timestamp(m).date()) for m in L.columns]
    return L, pd.DataFrame(events, columns=["month", "event", "line", "with"])


def churn(L: pd.DataFrame) -> pd.DataFrame:
    """Доля МО, сменивших линию между соседними снимками, и ARI соседних снимков."""
    from sklearn.metrics import adjusted_rand_score
    rows = []
    cols = list(L.columns)
    for a, b in zip(cols[:-1], cols[1:]):
        rows.append({"from": a, "to": b, "moved_share": float((L[a] != L[b]).mean()),
                     "ari": float(adjusted_rand_score(L[a], L[b]))})
    return pd.DataFrame(rows)


def mo_mobility(L: pd.DataFrame) -> pd.DataFrame:
    """Сколько раз МО меняло линию за год и сколько разных линий посетило."""
    ch = (L.ne(L.shift(axis=1))).iloc[:, 1:].sum(axis=1)
    return pd.DataFrame({"switches": ch, "n_lines": L.nunique(axis=1),
                         "first": L.iloc[:, 0], "last": L.iloc[:, -1]})


def transitions(L: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return pd.crosstab(L[a], L[b])
