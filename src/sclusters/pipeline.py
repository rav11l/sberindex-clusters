"""Конвейер: снимки → графы → разбиения → метрики → устойчивость → внешняя проверка → динамика.

Каждая стадия пишет таблицы в output_dir; рисунки строит figures.py по этим таблицам.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cluster as C
from . import data as D
from . import dynamics as Y
from . import features as F
from . import graphs as G
from . import validate as V

log = logging.getLogger("sclusters")


def months(cfg) -> list[pd.Timestamp]:
    s = cfg["snapshots"]
    return list(pd.date_range(s["start"], s["end"], freq="MS"))


class Run:
    """Кэширует признаки и графы снимков, чтобы стадии не пересчитывали их."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out = Path(cfg["output_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.cube, self.passport = D.load(cfg)
        self.total = cfg["data"]["total"]
        self.cats = cfg["data"]["categories"]
        self._X, self._M, self._W, self._P = {}, {}, {}, {}

    # ---- признаки и графы ----
    def X(self, t):
        t = pd.Timestamp(t)
        if t not in self._X:
            self._X[t] = F.snapshot_features(self.cube, t, self.total, self.cats, self.cfg["snapshots"]["window"])
        return self._X[t]

    def Z(self, t, variant):
        X, blocks = self.X(t)
        return F.standardize(X, blocks, self.cfg["features"]["variants"][variant])

    def M(self, t):
        t = pd.Timestamp(t)
        if t not in self._M:
            self._M[t] = F.comove_matrix(self.cube, t, self.total, self.cats, self.cfg["snapshots"]["comove_window"])
        return self._M[t]

    def W(self, t, variant, rule):
        key = (pd.Timestamp(t), variant, rule)
        if key not in self._W:
            g = self.cfg["graph"]
            self._W[key] = G.build(rule, self.Z(t, variant), self.M(t), g["k"], g["hybrid_alpha"])
        return self._W[key]

    # ---- разбиение одной конфигурации на одном снимке ----
    def partition(self, t, variant, rule, method, seed=None, initial=None, resolution=None):
        key = (pd.Timestamp(t), variant, rule, method, seed, resolution)
        if initial is None and key in self._P:
            return self._P[key]
        lab = self._partition(t, variant, rule, method, seed, initial, resolution)
        if initial is None:
            self._P[key] = lab
        return lab

    def _partition(self, t, variant, rule, method, seed=None, initial=None, resolution=None):
        cfg = self.cfg
        if resolution is not None:
            cfg = {**cfg, "cluster": {**cfg["cluster"], "resolution": resolution}}
        Z = self.Z(t, variant)
        W = self.W(t, variant, rule)
        k = None
        if method != "leiden":
            k = len(np.unique(self.partition(t, variant, rule, "leiden", seed, resolution=resolution)))
        seed = cfg["seed"] if seed is None else seed
        return pd.Series(C.run(method, W, Z.to_numpy(), k, cfg, seed, initial), index=Z.index)


def stage_data(run: Run):
    p = run.passport
    p["sha256"] = D.sha256(Path(run.cfg["data"]["raw_dir"]) / run.cfg["data"]["file"])
    (run.out / "data_passport.json").write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def stage_snapshots(run: Run):
    """Признаки всех снимков (сырые значения) и сводка графов."""
    rows, feats = [], []
    for t in months(run.cfg):
        X, _ = run.X(t)
        feats.append(X.assign(month=str(t.date())))
        for v in run.cfg["features"]["variants"]:
            for r in run.cfg["graph"]["rules"]:
                rows.append({"month": str(t.date()), "variant": v, "rule": r, **G.summary(run.W(t, v, r))})
    pd.concat(feats).rename_axis("mo").reset_index().to_parquet(run.out / "features.parquet", index=False)
    S = pd.DataFrame(rows)
    S.to_csv(run.out / "graph_summary.csv", index=False)
    return S


def stage_compare(run: Run):
    """Все сочетания признаков × правила рёбер × метода на всех снимках: качество и согласие."""
    cfg = run.cfg
    labels, rows = {}, []
    for v in cfg["features"]["variants"]:
        for r in cfg["graph"]["rules"]:
            for m in cfg["cluster"]["methods"]:
                for t in months(cfg):
                    lab = run.partition(t, v, r, m)
                    labels[(v, r, m, str(t.date()))] = lab
                    q = C.quality(run.W(t, v, r), run.Z(t, v).to_numpy(), lab.to_numpy())
                    rows.append({"variant": v, "rule": r, "method": m, "month": str(t.date()), **q})
                log.info("compare %s %s %s", v, r, m)
    Q = pd.DataFrame(rows)
    # временная согласованность: ARI соседних снимков
    tc = []
    for (v, r, m) in Q[["variant", "rule", "method"]].drop_duplicates().itertuples(index=False):
        ms = [str(t.date()) for t in months(cfg)]
        a = [C.ari(labels[(v, r, m, x)], labels[(v, r, m, y)]) for x, y in zip(ms[:-1], ms[1:])]
        tc.append({"variant": v, "rule": r, "method": m, "temporal_ari": float(np.mean(a))})
    Q.to_csv(run.out / "quality_by_month.csv", index=False)
    pd.DataFrame(tc).to_csv(run.out / "temporal_consistency.csv", index=False)
    # согласие правил рёбер и методов между собой (Leiden, средний ARI по снимкам)
    agree = []
    keys = sorted({(v, r, m) for (v, r, m, _) in labels})
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] != b[0]:
                continue
            x = np.mean([C.ari(labels[(*a, str(t.date()))], labels[(*b, str(t.date()))]) for t in months(cfg)])
            agree.append({"variant": a[0], "a": f"{a[1]}/{a[2]}", "b": f"{b[1]}/{b[2]}", "ari": float(x)})
    pd.DataFrame(agree).to_csv(run.out / "agreement.csv", index=False)
    pd.to_pickle(labels, run.out / "labels_all.pkl")
    return Q


def _boot_once(run: Run, t, v, r, m, rng, frac, full: pd.Series, resolution=None):
    cfg = run.cfg
    Zf = run.Z(t, v)
    idx = np.sort(rng.choice(len(Zf), int(frac * len(Zf)), replace=False))
    Z = Zf.iloc[idx]
    g = cfg["graph"]
    W = G.build(r, Z, run.M(t), g["k"], g["hybrid_alpha"])
    c = {**cfg, "cluster": {**cfg["cluster"], "n_restarts": 3}}
    if resolution is not None:
        c["cluster"]["resolution"] = resolution
    seed = int(rng.integers(1 << 30))
    k = None
    if m != "leiden":
        k = len(np.unique(full))
    lab = C.run(m, W, Z.to_numpy(), k, c, seed)
    return C.ari(full.iloc[idx].to_numpy(), lab)


def stage_stability(run: Run):
    """Устойчивость к составу выборки: 90 % МО, граф строится заново, ARI с полным разбиением
    на общих МО. Плюс чувствительность Leiden к параметру разрешения."""
    cfg = run.cfg
    st = cfg["stability"]
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for v in cfg["features"]["variants"]:
        for r in cfg["graph"]["rules"]:
            for m in cfg["cluster"]["methods"]:
                for t in st["months"]:
                    full = run.partition(t, v, r, m)
                    for b in range(st["n_boot"]):
                        rows.append({"variant": v, "rule": r, "method": m, "month": t, "boot": b,
                                     "ari": _boot_once(run, t, v, r, m, rng, st["frac"], full)})
                log.info("stability %s %s %s", v, r, m)
    B = pd.DataFrame(rows)
    B.to_csv(run.out / "stability_boot.csv", index=False)
    # разрешение Leiden: k, модулярность, устойчивость — на основной конфигурации признаков
    v = cfg["features"]["main_variant"]
    res_rows = []
    for r in cfg["graph"]["rules"]:
        for gamma in st["resolution_grid"]:
            for t in st["months"]:
                full = run.partition(t, v, r, "leiden", resolution=gamma)
                W = run.W(t, v, r)
                q = C.quality(W, run.Z(t, v).to_numpy(), full.to_numpy())
                aris = [_boot_once(run, t, v, r, "leiden", rng, st["frac"], full, gamma) for _ in range(max(5, st["n_boot"] // 2))]
                res_rows.append({"rule": r, "resolution": gamma, "month": t, "k": q["k"],
                                 "modularity": q["modularity"], "silhouette": q.get("silhouette"),
                                 "stability_ari": float(np.mean(aris))})
        log.info("resolution %s", r)
    R = pd.DataFrame(res_rows)
    R.to_csv(run.out / "resolution_sensitivity.csv", index=False)
    return B, R


def stage_validate(run: Run):
    """Внешняя проверка: однороднее ли сезонный профиль внутри кластеров, чем внутри случайных
    групп тех же размеров, и помогает ли групповой фактор прогнозу против общего фактора страны."""
    cfg = run.cfg
    vc = cfg["validation"]
    wide = F.wide(run.cube, run.total)
    parts = {}
    for v in cfg["features"]["variants"]:
        for r in cfg["graph"]["rules"]:
            for m in cfg["cluster"]["methods"]:
                for o in vc["origins"]:
                    parts[(f"{v}/{r}/{m}", o)] = run.partition(o, v, r, m)
    for o in vc["origins"]:
        parts[("mo_type", o)] = pd.Series([D.mo_type(x) for x in wide.index], index=wide.index)
    out = {}
    for growth in ["national", "group"]:
        S, E = V.evaluate(wide, parts, vc["origins"], vc["horizons"], vc["factor_window"],
                          vc["n_placebo"], cfg["seed"], growth=growth)
        S.insert(0, "growth", growth)
        out[growth] = S
    S = pd.concat(out.values(), ignore_index=True)
    S.to_csv(run.out / "validation_forecast.csv", index=False)
    return S


def main_key(cfg):
    return cfg["features"]["main_variant"], cfg["graph"]["main_rule"], cfg["cluster"]["main_method"]


def stage_dynamics(run: Run):
    """Основная конфигурация: независимые и эволюционные разбиения, сквозные линии, события,
    профили кластеров, переходы, МО с паводком 2024 года."""
    cfg = run.cfg
    v, r, m = main_key(cfg)
    ms = months(cfg)
    indep = {t: run.partition(t, v, r, m) for t in ms}
    parts = {"independent": indep}
    if cfg["cluster"].get("evolutionary") and m == "leiden":
        evo, prev = {}, None
        for t in ms:
            init = None if prev is None else prev.reindex(run.Z(t, v).index).to_numpy()
            evo[t] = run.partition(t, v, r, m, initial=init)
            prev = evo[t]
        parts["evolutionary"] = evo
    rows = []
    for kind, P in parts.items():
        for t, lab in P.items():
            q = C.quality(run.W(t, v, r), run.Z(t, v).to_numpy(), lab.to_numpy())
            rows.append({"kind": kind, "month": str(t.date()), **q})
    Q = pd.DataFrame(rows)
    ch = []
    for kind, P in parts.items():
        L, ev = Y.track(P, cfg["dynamics"]["jaccard_continue"])
        c = Y.churn(L).assign(kind=kind)
        ch.append(c)
        L.to_csv(run.out / f"lines_{kind}.csv")
        ev.to_csv(run.out / f"events_{kind}.csv", index=False)
    Q.to_csv(run.out / "dynamics_quality.csv", index=False)
    pd.concat(ch).to_csv(run.out / "dynamics_churn.csv", index=False)
    kind = "evolutionary" if "evolutionary" in parts else "independent"
    L = pd.read_csv(run.out / f"lines_{kind}.csv", index_col=0)
    profiles(run, L)
    mob = Y.mo_mobility(L)
    mob["mo_type"] = [D.mo_type(x) for x in mob.index]
    mob.to_csv(run.out / "mo_mobility.csv")
    flood = [x for x in cfg["events"]["flood_2024"] if x in L.index]
    L.loc[flood].to_csv(run.out / "flood_2024_lines.csv")
    embedding(run, L)
    return Q


def profiles(run: Run, L: pd.DataFrame):
    """Профиль линии на каждом снимке: медианы сырых признаков, уровень в рублях, состав по типам МО."""
    rows, comp = [], []
    wide = F.wide(run.cube, run.total)
    for col in L.columns:
        t = pd.Timestamp(col)
        X, _ = run.X(t)
        lab = L[col].reindex(X.index)
        win = [c for c in wide.columns if t - pd.DateOffset(months=run.cfg["snapshots"]["window"] - 1) <= c <= t]
        rub = wide.loc[X.index, win].mean(axis=1)
        for c, idx in lab.groupby(lab).groups.items():
            med = X.loc[idx].median()
            rows.append({"month": col, "line": int(c), "n": len(idx), "rub_median": float(rub.loc[idx].median()),
                         **{k: float(x) for k, x in med.items()}})
            types = pd.Series([D.mo_type(x) for x in idx]).value_counts(normalize=True)
            comp.append({"month": col, "line": int(c), **types.to_dict()})
    pd.DataFrame(rows).to_csv(run.out / "profiles.csv", index=False)
    pd.DataFrame(comp).fillna(0).to_csv(run.out / "composition.csv", index=False)
    # состав линий на последнем снимке поимённо: крупнейшие по расходам МО каждой линии
    last = L.columns[-1]
    t = pd.Timestamp(last)
    rub = wide[t]
    ex = []
    for c, idx in L[last].groupby(L[last]).groups.items():
        top = rub.loc[idx].sort_values(ascending=False)
        ex.append({"line": int(c), "n": len(idx), "examples_top": "; ".join(top.index[:6]),
                   "examples_bottom": "; ".join(top.index[-3:])})
    pd.DataFrame(ex).to_csv(run.out / "examples.csv", index=False)


def embedding(run: Run, L: pd.DataFrame):
    """Двумерная проекция признакового пространства для «карты экономик»: в выгрузке нет
    координат и регионов МО, поэтому географической карты нет, есть карта сходства."""
    from sklearn.manifold import TSNE
    v = run.cfg["features"]["main_variant"]
    t = pd.Timestamp(L.columns[-1])
    Z = run.Z(t, v)
    xy = TSNE(n_components=2, perplexity=40, random_state=run.cfg["seed"], init="pca").fit_transform(Z.to_numpy())
    E = pd.DataFrame(xy, index=Z.index, columns=["x", "y"])
    E["line"] = L[L.columns[-1]].reindex(Z.index)
    E["mo_type"] = [D.mo_type(x) for x in Z.index]
    E["rub"] = F.wide(run.cube, run.total)[t].reindex(Z.index)
    E.to_csv(run.out / "embedding.csv")


def _haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def stage_geo(run: Run):
    """Регион и центроид МО из OSM; проверка, восстанавливают ли рёбра и кластеры географию.

    Ни регион, ни координаты в признаки и рёбра не входят. Если рёбра совместного движения
    соединяют МО одного региона чаще, чем случайные пары, сеть восстанавливает географию сама.
    """
    from sklearn.metrics import adjusted_mutual_info_score
    from . import geo as GEO
    cfg = run.cfg
    if not cfg.get("geo", {}).get("enabled"):
        return None
    try:
        root = GEO.fetch(cfg["geo"]["root"])
    except Exception as e:  # нет сети или git: стадия необязательная
        log.warning("geo пропущена: не удалось получить границы OSM (%s)", e)
        return None
    T = GEO.osm_table(root)
    names = run.Z(months(cfg)[0], cfg["features"]["main_variant"]).index
    Mg = GEO.match(names, T)
    Mg.to_csv(run.out / "geo_match.csv")
    ok = Mg[Mg.status == "ok"]
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    v = cfg["features"]["main_variant"]
    scopes = {"all": ok, "no_fed_cities": ok[ok.fd != "fed_city"]}
    for t in months(cfg):
        for r in cfg["graph"]["rules"]:
            W = run.W(t, v, r).tocoo()
            idx = run.Z(t, v).index
            up = W.row < W.col
            A0, B0 = idx[W.row[up]], idx[W.col[up]]
            for scope, S in scopes.items():
                m = A0.isin(S.index) & B0.isin(S.index)
                a, b = A0[m], B0[m]
                same = (S.loc[a, "region"].to_numpy() == S.loc[b, "region"].to_numpy())
                d = _haversine(S.loc[a, "lon"].to_numpy(), S.loc[a, "lat"].to_numpy(),
                               S.loc[b, "lon"].to_numpy(), S.loc[b, "lat"].to_numpy())
                # ожидание без географии: случайные пары из тех же МО
                pa, pb = rng.choice(S.index, 20000), rng.choice(S.index, 20000)
                keep = pa != pb
                pa, pb = pa[keep], pb[keep]
                same0 = (S.loc[pa, "region"].to_numpy() == S.loc[pb, "region"].to_numpy()).mean()
                d0 = np.median(_haversine(S.loc[pa, "lon"].to_numpy(), S.loc[pa, "lat"].to_numpy(),
                                          S.loc[pb, "lon"].to_numpy(), S.loc[pb, "lat"].to_numpy()))
                rows.append({"month": str(t.date()), "rule": r, "scope": scope, "edges": int(m.sum()),
                             "same_region_share": float(same.mean()), "same_region_random": float(same0),
                             "median_km": float(np.median(d)), "median_km_random": float(d0)})
    E = pd.DataFrame(rows)
    E.to_csv(run.out / "geo_edges.csv", index=False)
    # согласие кластеров с регионами (AMI) по всем конфигурациям
    labels = pd.read_pickle(run.out / "labels_all.pkl")
    am = []
    for (vv, r, m, t), lab in labels.items():
        l = lab.reindex(ok.index).dropna()
        am.append({"variant": vv, "rule": r, "method": m, "month": t,
                   "ami_region": float(adjusted_mutual_info_score(ok.loc[l.index, "region"], l)),
                   "ami_fd": float(adjusted_mutual_info_score(ok.loc[l.index, "fd"], l))})
    pd.DataFrame(am).to_csv(run.out / "geo_ami.csv", index=False)
    # состав кластеров по регионам на последнем снимке: совместное движение и основное правило
    last = str(months(cfg)[-1].date())
    reg = []
    for r in cfg["graph"]["rules"]:
        lab = labels[(v, r, "leiden", last)]
        l = lab.reindex(ok.index).dropna().astype(int)
        for c, idx in l.groupby(l).groups.items():
            vc = ok.loc[idx, "region"].value_counts()
            reg.append({"rule": r, "cluster": int(c), "n_matched": int(len(idx)), "n_regions": int(vc.size),
                        "top_regions": "; ".join(f"{a} {b}" for a, b in vc.head(8).items())})
    pd.DataFrame(reg).to_csv(run.out / "geo_cluster_regions.csv", index=False)
    # доля МО региона в его самом частом кластере совместного движения (для регионов с 15+ МО)
    lab = labels[(v, "comove", "leiden", last)].reindex(ok.index).dropna()
    conc = (ok.loc[lab.index].assign(c=lab).groupby("region")["c"]
            .agg(n="size", top_share=lambda x: x.value_counts(normalize=True).iloc[0],
                 top_cluster=lambda x: int(x.value_counts().index[0])))
    conc[conc.n >= 15].sort_values("top_share", ascending=False).to_csv(run.out / "geo_region_concentration.csv")
    return E


def stage_summary(run: Run):
    """Ключевые числа отчёта в одном файле: отчёт и страница берут цифры отсюда, тест сверяет."""
    cfg, out = run.cfg, run.out
    v, r, m = main_key(cfg)
    S = {"passport": json.loads((out / "data_passport.json").read_text(encoding="utf-8"))}
    A = pd.read_csv(out / "agreement.csv")
    A = A[A.variant == v].set_index(["a", "b"])["ari"]
    S["agreement_leiden"] = {f"{a}|{b}": round(float(x), 3) for (a, b), x in A.items()
                             if a.endswith("leiden") and b.endswith("leiden")}
    Q = pd.read_csv(out / "quality_by_month.csv")
    T = pd.read_csv(out / "temporal_consistency.csv")
    B = pd.read_csv(out / "stability_boot.csv")
    tab = (Q.groupby(["variant", "rule", "method"])[["k", "modularity", "silhouette", "largest_share"]].mean()
           .join(T.set_index(["variant", "rule", "method"]))
           .join(B.groupby(["variant", "rule", "method"])["ari"].mean().rename("bootstrap_ari")))
    if (out / "geo_ami.csv").exists():
        tab = tab.join(pd.read_csv(out / "geo_ami.csv").groupby(["variant", "rule", "method"])["ami_region"].mean())
    Vd = pd.read_csv(out / "validation_forecast.csv")
    Vn = Vd[(Vd.growth == "national") & (Vd.partition.str.count("/") == 2)].copy()
    Vn[["variant", "rule", "method"]] = Vn.partition.str.split("/", expand=True)
    Vn["vs_placebo_%"] = 100 * (Vn.MAE / Vn.placebo_MAE_mean - 1)
    tab = tab.join(Vn.set_index(["variant", "rule", "method"])[["MAE", "vs_national_%", "vs_placebo_%", "p_vs_placebo"]])
    tab.round(4).to_csv(out / "comparison_table.csv")
    S["main"] = {"variant": v, "rule": r, "method": m, "resolution": cfg["cluster"]["resolution"],
                 **{k: (None if pd.isna(x) else round(float(x), 4)) for k, x in tab.loc[(v, r, m)].items()}}
    S["national_MAE"] = round(float(Vd[(Vd.growth == "national") & (Vd.partition == "national")].MAE.iloc[0]), 1)
    if (out / "geo_edges.csv").exists():
        E = pd.read_csv(out / "geo_edges.csv").groupby(["scope", "rule"]).mean(numeric_only=True)
        S["geo_edges"] = {f"{a}|{b}": {k: round(float(x), 4) for k, x in row.items()} for (a, b), row in E.iterrows()}
        G = pd.read_csv(out / "geo_match.csv")
        S["geo_match"] = G.status.value_counts().to_dict()
    C = pd.read_csv(out / "dynamics_churn.csv")
    S["churn_mean"] = {k: round(float(x), 4) for k, x in C.groupby("kind").moved_share.mean().items()}
    S["churn_max"] = {k: round(float(x), 4) for k, x in C.groupby("kind").moved_share.max().items()}
    ev = pd.read_csv(out / "events_evolutionary.csv") if (out / "events_evolutionary.csv").exists() else pd.DataFrame()
    S["events_evolutionary"] = int(len(ev))
    P = pd.read_csv(out / "profiles.csv")
    last = P.month.max()
    S["profiles_last"] = P[P.month == last].round(4).to_dict(orient="records")
    first = P.month.min()
    S["sizes_first"] = P[P.month == first].set_index("line")["n"].astype(int).to_dict()
    mob = pd.read_csv(out / "mo_mobility.csv", index_col=0)
    S["mo_never_moved_share"] = round(float((mob.switches == 0).mean()), 4)
    fl = pd.read_csv(out / "flood_2024_lines.csv", index_col=0)
    S["flood_cities_constant"] = bool((fl.nunique(axis=1) == 1).all())
    S["n_flood_cities"] = int(len(fl))
    (out / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return S


STAGES = {"data": stage_data, "snapshots": stage_snapshots, "compare": stage_compare,
          "stability": stage_stability, "validate": stage_validate, "geo": stage_geo, "dynamics": stage_dynamics,
          "summary": stage_summary}


def run_all(cfg: dict, stages=None):
    run = Run(cfg)
    res = {}
    for name in stages or STAGES:
        t0 = time.time()
        res[name] = STAGES[name](run)
        log.info("стадия %s: %.0f с", name, time.time() - t0)
    return run, res
