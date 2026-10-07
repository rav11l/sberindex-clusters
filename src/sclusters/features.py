"""Признаки узлов (МО) на момент снимка.

Все признаки снимка t считаются только по данным по месяц t включительно, поэтому разбиение
на снимке t можно использовать для прогноза из точки t (внешняя проверка, validate.py).

Три блока:
  level     — log уровня расходов «Все категории» минус медиана по МО того же месяца;
  structure — доли пяти категорий и «прочего» (остаток до «Всех категорий»);
  growth    — годовой log-прирост по «Всем категориям» и пяти категориям минус медиана того же
              месяца. Годовой прирост существует только с 01.2024, поэтому в первые месяцы окно
              growth короче window.
Вычитание медианы того же месяца снимает общую для страны сезонность и динамику: остаётся
положение МО относительно панели. Тот же приём дал основной результат в sberindex-shocks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OTHER = "Прочее"


def wide(cube: pd.DataFrame, col: str) -> pd.DataFrame:
    return cube[col].unstack("ds").sort_index(axis=1)


def shares(cube: pd.DataFrame, total: str, categories: list[str]) -> pd.DataFrame:
    s = cube[categories].div(cube[total], axis=0)
    s[OTHER] = (1 - s.sum(axis=1)).clip(lower=0)
    return s.clip(0, 1)


def _window(cols: pd.DatetimeIndex, t: pd.Timestamp, w: int) -> list:
    return [c for c in cols if t - pd.DateOffset(months=w - 1) <= c <= t]


def _demean(df: pd.DataFrame) -> pd.DataFrame:
    """Вычитает медиану по МО в каждом месяце (столбце)."""
    return df.sub(df.median(axis=0), axis=1)


def snapshot_features(cube: pd.DataFrame, t: pd.Timestamp, total: str, categories: list[str],
                      window: int = 3) -> tuple[pd.DataFrame, dict]:
    """Признаки МО на снимке t. Возвращает (X_raw, blocks) — сырые значения и состав блоков."""
    t = pd.Timestamp(t)
    cols = wide(cube, total).columns
    win = _window(cols, t, window)
    feats, blocks = {}, {"level": [], "structure": [], "growth": []}

    lt = np.log(wide(cube, total))
    feats["level"] = _demean(lt)[win].mean(axis=1)
    blocks["level"].append("level")

    sh = shares(cube, total, categories)
    for c in categories + [OTHER]:
        name = f"share:{c}"
        feats[name] = sh[c].unstack("ds")[win].mean(axis=1)
        blocks["structure"].append(name)

    gwin = [c for c in win if c - pd.DateOffset(months=12) >= cols.min()]
    if gwin:
        for c in [total] + categories:
            lw = np.log(wide(cube, c))
            g = _demean(lw - lw.shift(12, axis=1))
            name = f"growth:{c}"
            feats[name] = g[gwin].mean(axis=1)
            blocks["growth"].append(name)
    return pd.DataFrame(feats), blocks


def standardize(X: pd.DataFrame, blocks: dict, weights: dict, clip: float = 5.0) -> pd.DataFrame:
    """Робастный z-счёт (медиана и МКР/1,349), обрезка ±clip, вес блока делится между признаками
    так, чтобы вклад блока в евклидово расстояние не зависел от числа признаков в нём."""
    med = X.median()
    iqr = (X.quantile(0.75) - X.quantile(0.25)) / 1.349
    Z = ((X - med) / iqr.replace(0, np.nan)).clip(-clip, clip).fillna(0)
    for b, cols in blocks.items():
        cols = [c for c in cols if c in Z]
        if cols:
            w = float(weights.get(b, 1.0))
            if w == 0:
                Z = Z.drop(columns=cols)
            else:
                Z[cols] = Z[cols] * np.sqrt(w / len(cols))
    return Z


def comove_matrix(cube: pd.DataFrame, t: pd.Timestamp, total: str, categories: list[str],
                  window: int = 12) -> pd.DataFrame:
    """Профиль совместного движения: месячные log-изменения шести рядов за `window` месяцев по t,
    минус медиана по МО того же месяца, стандартизованные внутри каждого ряда МО.
    Строка — МО, столбцы — (категория, месяц). Корреляция строк = их скалярное произведение / L."""
    t = pd.Timestamp(t)
    parts = []
    for c in [total] + categories:
        lw = np.log(wide(cube, c))
        d = _demean(lw.diff(axis=1))
        win = _window(d.columns, t, window)
        d = d[win].dropna(axis=1, how="all")
        d = d.sub(d.mean(axis=1), axis=0).div(d.std(axis=1, ddof=0).replace(0, np.nan), axis=0).fillna(0)
        d.columns = [f"{c}|{x.date()}" for x in d.columns]
        parts.append(d)
    return pd.concat(parts, axis=1)
