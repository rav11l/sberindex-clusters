"""Рисунки по таблицам из output_dir. Палитра — восемь категориальных цветов в фиксированном
порядке; номер линии кластера закрепляет цвет за линией на всех рисунках."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY = "#b9b8b3"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10,
})


def color(line: int, names: dict | None = None) -> str:
    return PALETTE[int(line)] if 0 <= int(line) < len(PALETTE) else GREY


_CFG = {}


def _labels(out: Path) -> dict:
    return {int(k): v for k, v in _CFG.get("interpretation", {}).get("names", {}).items()}


def fig_method_matrix(out: Path, figs: Path, cfg):
    """Сравнение правил рёбер и методов: качество, устойчивость, согласованность во времени, плацебо."""
    Q = pd.read_csv(out / "quality_by_month.csv")
    T = pd.read_csv(out / "temporal_consistency.csv")
    B = pd.read_csv(out / "stability_boot.csv")
    Vd = pd.read_csv(out / "validation_forecast.csv")
    v = cfg["features"]["main_variant"]
    q = Q[Q.variant == v].groupby(["rule", "method"])[["k", "modularity", "silhouette"]].mean()
    q = q.join(T[T.variant == v].set_index(["rule", "method"])["temporal_ari"])
    q = q.join(B[B.variant == v].groupby(["rule", "method"])["ari"].mean().rename("bootstrap_ari"))
    vv = Vd[(Vd.growth == "national") & Vd.partition.str.startswith(f"{v}/")].copy()
    vv[["_v", "rule", "method"]] = vv.partition.str.split("/", expand=True)
    vv["placebo_gain"] = 100 * (vv["placebo_MAE_mean"] - vv["MAE"]) / vv["placebo_MAE_mean"]
    q = q.join(vv.set_index(["rule", "method"])["placebo_gain"])
    cols = {"modularity": "Модулярность", "silhouette": "Силуэт\n(признаки)", "bootstrap_ari": "Устойчивость\n(бутстрап ARI)",
            "temporal_ari": "Согласованность\nво времени (ARI)", "placebo_gain": "Однородность\nсезонности,\n% к плацебо"}
    M = q[list(cols)]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    norm = (M - M.min()) / (M.max() - M.min()).replace(0, 1)
    ax.imshow(norm.to_numpy(), cmap="Blues", aspect="auto", vmin=-0.15, vmax=1.15)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M.iat[i, j]
            ax.text(j, i, f"{val:.2f}" if j < 4 else f"{val:+.1f}", ha="center", va="center",
                    color="white" if norm.iat[i, j] > 0.6 else INK, fontsize=8.5)
    ax.set_xticks(range(len(cols)), list(cols.values()), fontsize=8)
    rn = {"attr": "признаки", "comove": "совместное движение", "hybrid": "гибрид"}
    ax.set_yticks(range(len(M)), [f"{rn[r]} · {m}" if m != "kmeans" else f"k-средних по признакам (k как у «{rn[r]}»)"
                                  for r, m in M.index], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Правило рёбер × метод: среднее по снимкам 2024 года (темнее — лучше в столбце)", loc="left")
    fig.savefig(figs / "fig1_methods.png")
    plt.close(fig)
    q.round(4).to_csv(out / "methods_summary.csv")


def fig_agreement(out: Path, figs: Path, cfg):
    A = pd.read_csv(out / "agreement.csv")
    v = cfg["features"]["main_variant"]
    A = A[A.variant == v]
    keys = sorted(set(A.a) | set(A.b), key=lambda s: (["attr", "comove", "hybrid"].index(s.split("/")[0]), s))
    M = pd.DataFrame(np.eye(len(keys)), index=keys, columns=keys)
    for r in A.itertuples():
        M.loc[r.a, r.b] = M.loc[r.b, r.a] = r.ari
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(M.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{M.iat[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if M.iat[i, j] > 0.6 else INK)
    rn = {"attr": "призн.", "comove": "движ.", "hybrid": "гибр."}
    lab = [f"{rn[k.split('/')[0]]}/{k.split('/')[1]}" if k.split('/')[1] != "kmeans" else f"k-ср. (k {rn[k.split('/')[0]]})" for k in keys]
    ax.set_xticks(range(len(keys)), lab, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(keys)), lab, fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Согласие разбиений (ARI, среднее по 12 снимкам)", loc="left")
    fig.savefig(figs / "fig2_agreement.png")
    plt.close(fig)


def fig_resolution(out: Path, figs: Path, cfg):
    R = pd.read_csv(out / "resolution_sensitivity.csv").groupby(["rule", "resolution"]).mean(numeric_only=True).reset_index()
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.8))
    rn = {"attr": "признаки", "comove": "совместное движение", "hybrid": "гибрид"}
    for i, r in enumerate(["attr", "comove", "hybrid"]):
        d = R[R.rule == r]
        for ax, col, t in zip(axs, ["k", "modularity", "stability_ari"], ["Число кластеров", "Модулярность", "Устойчивость (ARI)"]):
            ax.plot(d.resolution, d[col], marker="o", ms=4, lw=2, color=PALETTE[i], label=rn[r])
            ax.set_title(t, loc="left")
            ax.set_xlabel("разрешение γ")
    axs[0].legend(frameon=False, fontsize=8)
    g = cfg["cluster"]["resolution"]
    for ax in axs:
        ax.axvline(g, color=GREY, lw=1, ls="--")
    fig.tight_layout()
    fig.savefig(figs / "fig3_resolution.png")
    plt.close(fig)


def fig_sankey(out: Path, figs: Path, cfg, kind="evolutionary"):
    p = out / f"lines_{kind}.csv"
    if not p.exists():
        kind = "independent"
        p = out / f"lines_{kind}.csv"
    L = pd.read_csv(p, index_col=0)
    names = _labels(out)
    ms = [str(pd.Timestamp(m).date()) for m in cfg["dynamics"]["sankey_months"]]
    ms = [m for m in ms if m in L.columns]
    n = len(L)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    gap = 0.012 * n
    pos = {}
    for x, m in enumerate(ms):
        sizes = L[m].value_counts().sort_index()
        y = 0
        for c, s in sizes.items():
            pos[(m, c)] = [y, y + s, y]  # низ, верх, курсор для потоков
            ax.add_patch(plt.Rectangle((x - 0.04, y), 0.08, s, color=color(c), lw=0))
            y += s + gap
    for a, b in zip(ms[:-1], ms[1:]):
        ct = pd.crosstab(L[a], L[b])
        out_cur = {c: pos[(a, c)][0] for c in ct.index}
        in_cur = {c: pos[(b, c)][0] for c in ct.columns}
        xa, xb = ms.index(a) + 0.04, ms.index(b) - 0.04
        for i in ct.index:
            for j in ct.columns:
                f = ct.at[i, j]
                if f == 0:
                    continue
                y0a, y0b = out_cur[i], in_cur[j]
                xs = np.linspace(xa, xb, 50)
                s = (1 - np.cos(np.pi * (xs - xa) / (xb - xa))) / 2
                lo = y0a + (y0b - y0a) * s
                ax.fill_between(xs, lo, lo + f, color=color(i), alpha=0.35 if i == j else 0.55, lw=0)
                out_cur[i] += f
                in_cur[j] += f
    last = ms[-1]
    for c in sorted(L[last].unique()):
        lo, hi, _ = pos[(last, c)]
        ax.text(len(ms) - 1 + 0.08, (lo + hi) / 2, f"{c}. {names.get(c, '')}  ({int(round(hi - lo))})", va="center", fontsize=8, color=INK)
    first = ms[0]
    for c in sorted(L[first].unique()):
        lo, hi, _ = pos[(first, c)]
        ax.text(-0.08, (lo + hi) / 2, f"{int(round(hi - lo))}", va="center", ha="right", fontsize=8, color=INK2)
    mn = {1: "янв", 2: "фев", 3: "мар", 4: "апр", 5: "май", 6: "июн", 7: "июл", 8: "авг", 9: "сен", 10: "окт", 11: "ноя", 12: "дек"}
    ax.set_xticks(range(len(ms)), [f"{mn[pd.Timestamp(m).month]} 2024" for m in ms])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.5, len(ms) - 1 + 2.2)
    ax.invert_yaxis()
    ax.set_title("Переходы МО между типами локальных экономик в 2024 году", loc="left")
    fig.savefig(figs / "fig4_sankey.png")
    plt.close(fig)


def fig_profiles(out: Path, figs: Path, cfg):
    P = pd.read_csv(out / "profiles.csv")
    names = _labels(out)
    last = P.month.max()
    d = P[P.month == last].set_index("line").sort_index()
    cols = [c for c in d.columns if c.startswith(("level", "share:", "growth:"))]
    nice = {"level": "Уровень расходов\n(к медиане)"}
    for c in cols:
        if c.startswith("share:"):
            nice[c] = "Доля: " + c.split(":")[1].lower()
        elif c.startswith("growth:"):
            nice[c] = "Прирост г/г: " + c.split(":")[1].lower()
    # z к распределению признаков по всем МО на том же снимке
    F = pd.read_parquet(out / "features.parquet")
    F = F[F.month == last]
    Zc = (d[cols] - F[cols].median()) / ((F[cols].quantile(0.75) - F[cols].quantile(0.25)) / 1.349)
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(d) + 1.8))
    im = ax.imshow(Zc.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    for i in range(Zc.shape[0]):
        for j in range(Zc.shape[1]):
            ax.text(j, i, f"{Zc.iat[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(Zc.iat[i, j]) > 1.3 else INK)
    ax.set_xticks(range(len(cols)), [nice[c] for c in cols], rotation=55, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(d)), [f"{c}. {names.get(c, '')} · {int(n)} МО" for c, n in zip(d.index, d.n)], fontsize=8)
    for t, c in zip(ax.get_yticklabels(), d.index):
        t.set_color(color(c))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    used = run_blocks = _CFG.get("features", {}).get("variants", {}).get(_CFG.get("features", {}).get("main_variant"), {})
    g0 = next((j for j, c in enumerate(cols) if c.startswith("growth:")), None)
    title = "Профили типов на декабрь 2024 года"
    if g0 is not None and float(used.get("growth", 1)) == 0:
        ax.axvline(g0 - 0.5, color=INK, lw=1.5)
        ax.text(g0 - 0.4, -0.75, "в разбиении не участвуют →", fontsize=7.5, color=INK2, va="bottom")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("медиана кластера, робастный z к панели", fontsize=7.5)
    ax.set_title(title, loc="left", pad=16)
    fig.savefig(figs / "fig5_profiles.png")
    plt.close(fig)


def fig_embedding(out: Path, figs: Path, cfg):
    E = pd.read_csv(out / "embedding.csv", index_col=0)
    names = _labels(out)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    for c, d in E.groupby("line"):
        ax.scatter(d.x, d.y, s=9, color=color(c), alpha=0.75, lw=0)
        ax.text(d.x.median(), d.y.median(), f"{c}", fontsize=11, weight="bold", ha="center", va="center",
                color=INK, bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=color(c), lw=1.5))
    for name in ["городской округ город Казань", "городской округ город Уфа", "городской округ город Орск"]:
        if name in E.index:
            r = E.loc[name]
            ax.annotate(name.replace("городской округ город ", ""), (r.x, r.y), xytext=(8, 8),
                        textcoords="offset points", fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    leg = [plt.Line2D([], [], marker="o", ls="", color=color(c), label=f"{c}. {names.get(c, '')}") for c in sorted(E.line.unique())]
    ax.legend(handles=leg, frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("Карта сходства МО (t-SNE по признакам, декабрь 2024)", loc="left")
    fig.savefig(figs / "fig6_embedding.png")
    plt.close(fig)


def fig_churn(out: Path, figs: Path, cfg):
    C = pd.read_csv(out / "dynamics_churn.csv")
    fig, ax = plt.subplots(figsize=(7, 2.8))
    nm = {"independent": "независимо на каждом снимке", "evolutionary": "эволюционно (старт с прошлого месяца)"}
    for i, (k, d) in enumerate(C.groupby("kind")):
        x = pd.to_datetime(d["to"])
        ax.plot(x, 100 * d.moved_share, marker="o", ms=4, lw=2, color=PALETTE[i], label=nm.get(k, k))
    ax.set_ylabel("% МО, сменивших тип")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_title("Сколько МО меняют тип от месяца к месяцу", loc="left")
    fig.savefig(figs / "fig7_churn.png")
    plt.close(fig)


def fig_membership(out: Path, figs: Path, cfg, kind="evolutionary"):
    """Тепловая карта принадлежности: МО (строки, отсортированы по типу в декабре) × месяц."""
    p = out / f"lines_{kind}.csv"
    if not p.exists():
        p = out / "lines_independent.csv"
    L = pd.read_csv(p, index_col=0)
    L = L.sort_values(list(L.columns[::-1]))
    from matplotlib.colors import ListedColormap
    mx = int(L.max().max()) + 1
    cmap = ListedColormap([color(i) for i in range(mx)])
    fig, ax = plt.subplots(figsize=(5.5, 6))
    ax.imshow(L.to_numpy(), aspect="auto", cmap=cmap, vmin=-0.5, vmax=mx - 0.5, interpolation="nearest")
    ax.set_xticks(range(L.shape[1]), [c[5:7] for c in L.columns])
    ax.set_xlabel("месяц 2024")
    ax.set_yticks([])
    ax.set_title("Принадлежность МО к типам по месяцам", loc="left")
    fig.savefig(figs / "fig8_membership.png")
    plt.close(fig)


def albers(lon, lat, lon0=100.0, lat0=60.0, p1=52.0, p2=68.0):
    """Равновеликая коническая проекция Альберса для России; долготы за 180° переносятся."""
    lon = np.where(np.asarray(lon) < 0, np.asarray(lon) + 360, lon)
    lon, lat, lon0, lat0, p1, p2 = map(np.radians, (lon, lat, lon0, lat0, p1, p2))
    n = (np.sin(p1) + np.sin(p2)) / 2
    C = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho = np.sqrt(C - 2 * n * np.sin(lat)) / n
    rho0 = np.sqrt(C - 2 * n * np.sin(lat0)) / n
    th = n * (lon - lon0)
    return rho * np.sin(th), rho0 - rho * np.cos(th)


def _basemap(ax, cfg):
    import json
    root = Path(cfg.get("geo", {}).get("root", "")) / "GeoJson's" / "Countries" / "Russia_regions.geojson"
    if not root.exists():
        return
    for f in json.loads(root.read_text(encoding="utf-8"))["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for p in polys:
            ring = np.asarray(p[0], dtype=float)
            if len(ring) < 3:
                continue
            ring = ring[:: max(1, len(ring) // 3000)]
            x, y = albers(ring[:, 0], ring[:, 1])
            ax.fill(x, y, facecolor="#f3f2ef", edgecolor="#d6d5d0", lw=0.4, zorder=0)


def fig_map(out: Path, figs: Path, cfg, kind="evolutionary"):
    G = out / "geo_match.csv"
    if not G.exists():
        return
    Mg = pd.read_csv(G, index_col=0)
    Mg = Mg[Mg.status == "ok"]
    p = out / f"lines_{kind}.csv"
    if not p.exists():
        p = out / "lines_independent.csv"
    L = pd.read_csv(p, index_col=0)
    names = _labels(out)
    labels = pd.read_pickle(out / "labels_all.pkl")
    v = cfg["features"]["main_variant"]
    last = L.columns[-1]
    panels = [("Основное разбиение (гибрид), декабрь 2024", L[last], names),
              ("Только совместное движение расходов, декабрь 2024", labels[(v, "comove", "leiden", last)], {})]
    fig, axs = plt.subplots(2, 1, figsize=(10, 11))
    for ax, (title, lab, nm) in zip(axs, panels):
        _basemap(ax, cfg)
        d = Mg.join(lab.rename("c"), how="inner")
        x, y = albers(d.lon.to_numpy(), d.lat.to_numpy())
        for c in sorted(d.c.unique()):
            m = (d.c == c).to_numpy()
            ax.scatter(x[m], y[m], s=7, color=color(c), lw=0, alpha=0.9,
                       label=f"{int(c)}. {nm.get(int(c), '')}".rstrip(". ") + f" ({m.sum()})")
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, fontsize=7.5, loc="lower left", markerscale=2, ncol=2)
    fig.text(0.01, 0.005, f"Точки — центроиды {len(Mg)} МО, сопоставленных по названию с границами OpenStreetMap "
             "(© участники OpenStreetMap, ODbL). Москва и Санкт-Петербург на этом масштабе — по одной точке.",
             fontsize=7, color=INK2)
    fig.savefig(figs / "fig9_map.png")
    plt.close(fig)


def make_all(cfg):
    _CFG.clear()
    _CFG.update(cfg)
    out = Path(cfg["output_dir"])
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for f in [fig_method_matrix, fig_agreement, fig_resolution, fig_sankey, fig_profiles, fig_embedding,
              fig_churn, fig_membership, fig_map]:
        try:
            f(out, figs, cfg)
        except FileNotFoundError as e:
            print(f"{f.__name__}: нет входной таблицы ({e.filename}), пропущено")
