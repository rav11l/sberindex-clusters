"""Внешняя проверка разбиения через прогноз.

Если кластер — это реальный тип локальной экономики, у его МО должна быть общая динамика.
Проверка: контрольная модель factor_only из sberindex-shocks, в которой общий фактор считается
не по всей стране, а внутри кластера. Модель не обучает ни одного параметра: МО держит своё
последнее отклонение от фактора, фактор продлевается сезонным наивным прогнозом с поправкой
на средний годовой прирост за последние `window` месяцев.

Разбиение для точки прогноза T строится по снимку T, то есть только по данным до T включительно.
Контроль — случайные разбиения тех же размеров (плацебо): выигрыш, который даёт любое деление
панели на группы такого размера, не засчитывается.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_factor_forecast(wide: pd.DataFrame, labels: pd.Series, origin: pd.Timestamp, horizons,
                          window: int = 3, growth: str = "group") -> pd.DataFrame:
    """Прогноз уровней на T+h для всех МО: factor_only с фактором своей группы.

    growth="group"    — и сезонный профиль, и годовой прирост берутся у фактора группы;
    growth="national" — сезонный профиль у группы, годовой прирост у общего фактора страны.
    """
    lg = np.log(wide.loc[:, wide.columns <= origin].to_numpy())
    lab = labels.reindex(wide.index).to_numpy()
    nat = np.median(lg, axis=0)
    g_nat = np.mean(nat[-window:] - nat[-window - 12:-12])
    out = np.full((len(lg), len(horizons)), np.nan)
    for c in pd.unique(lab):
        m = lab == c
        cf = np.median(lg[m], axis=0)
        g = g_nat if growth == "national" else np.mean(cf[-window:] - cf[-window - 12:-12])
        dev = lg[m, -1] - cf[-1]
        for j, h in enumerate(horizons):
            out[m, j] = np.exp(dev + cf[-12 + h - 1] + g)
    return pd.DataFrame(out, index=wide.index, columns=list(horizons))


def mae(wide: pd.DataFrame, pred: pd.DataFrame, origin: pd.Timestamp, horizons) -> pd.DataFrame:
    rows = []
    for h in horizons:
        tgt = origin + pd.DateOffset(months=h)
        if tgt not in wide.columns:
            continue
        ae = (wide[tgt] - pred[h]).abs()
        rows.append(pd.DataFrame({"mo": wide.index, "h": h, "ae": ae.to_numpy()}))
    return pd.concat(rows)


def evaluate(wide: pd.DataFrame, partitions: dict, origins, horizons, window: int,
             n_placebo: int, seed: int = 0, growth: str = "group") -> tuple[pd.DataFrame, pd.DataFrame]:
    """partitions: {(name, origin): pd.Series меток по МО}. Возвращает (сводка, ошибки по парам)."""
    rng = np.random.default_rng(seed)
    errs = []
    for o in origins:
        o = pd.Timestamp(o)
        nat = pd.Series(0, index=wide.index)
        p = group_factor_forecast(wide, nat, o, horizons, window)
        errs.append(mae(wide, p, o, horizons).assign(partition="national", origin=o))
        for (name, po), lab in partitions.items():
            if pd.Timestamp(po) != o:
                continue
            lab = lab.reindex(wide.index)
            p = group_factor_forecast(wide, lab, o, horizons, window, growth)
            errs.append(mae(wide, p, o, horizons).assign(partition=name, origin=o))
            for r in range(n_placebo):
                perm = pd.Series(rng.permutation(lab.to_numpy()), index=lab.index)
                p = group_factor_forecast(wide, perm, o, horizons, window, growth)
                # плацебо хранится только как средняя ошибка повтора: построчно это миллионы строк
                e = mae(wide, p, o, horizons)
                errs.append(pd.DataFrame({"partition": [f"{name}~placebo"], "origin": [o], "rep": [r],
                                          "ae_sum": [e["ae"].sum()], "n": [len(e)]}))
    E = pd.concat(errs, ignore_index=True)
    return summarize(E), E[~E.partition.str.endswith("~placebo")].drop(columns=["ae_sum", "n", "rep"], errors="ignore")


def summarize(E: pd.DataFrame) -> pd.DataFrame:
    nat = E[E.partition == "national"].set_index(["mo", "h", "origin"])["ae"]
    rows = []
    real = [p for p in E.partition.unique() if not p.endswith("~placebo") and p != "national"]
    for p in ["national"] + real:
        e = E[E.partition == p].set_index(["mo", "h", "origin"])["ae"]
        row = {"partition": p, "MAE": e.mean(), "vs_national_%": 100 * (e.mean() / nat.mean() - 1),
               "share_mo_better": float((e.groupby("mo").mean() < nat.groupby("mo").mean()).mean())}
        pl = E[E.partition == f"{p}~placebo"]
        if len(pl):
            g = pl.groupby("rep")[["ae_sum", "n"]].sum()
            pm = g["ae_sum"] / g["n"]
            row.update({"placebo_MAE_mean": pm.mean(), "placebo_MAE_p05": pm.quantile(0.05),
                        "p_vs_placebo": float((pm <= e.mean()).mean())})
            # средняя разница с национальным фактором по каждой точке прогноза
            d = (e - nat.reindex(e.index)).groupby(level="origin").mean()
            row["diff_by_origin"] = ";".join(f"{x:.0f}" for x in d)
        rows.append(row)
    return pd.DataFrame(rows)
