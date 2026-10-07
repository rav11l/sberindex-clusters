"""Модульные проверки на синтетических данных — без выгрузки СберИндекса."""
import numpy as np
import pandas as pd
import pytest

from sclusters import cluster as C
from sclusters import dynamics as Y
from sclusters import features as F
from sclusters import graphs as G
from sclusters import validate as V

CATS = ["Продовольствие", "Маркетплейсы", "Транспорт", "Здоровье", "Общественное питание"]
TOT = "Все категории"


def synthetic_cube(n_per=40, seed=0):
    """Три группы МО с разной структурой расходов и разным сезонным профилем."""
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
    groups = [
        dict(level=20000, shares=[0.45, 0.20, 0.04, 0.04, 0.02], amp=0.02, phase=0),
        dict(level=35000, shares=[0.38, 0.17, 0.06, 0.05, 0.04], amp=0.10, phase=6),
        dict(level=55000, shares=[0.30, 0.15, 0.07, 0.05, 0.07], amp=0.05, phase=3),
    ]
    rows = []
    for g, p in enumerate(groups):
        common = rng.normal(0, 0.03, len(ds))
        for i in range(n_per):
            lvl = p["level"] * np.exp(rng.normal(0, 0.05))
            season = p["amp"] * np.sin(2 * np.pi * (np.arange(len(ds)) + p["phase"]) / 12)
            tot = lvl * np.exp(season + common + rng.normal(0, 0.01, len(ds)))
            for t, d in enumerate(ds):
                row = {"mo": f"g{g}_mo{i}", "ds": d, TOT: tot[t]}
                for c, s in zip(CATS, p["shares"]):
                    row[c] = tot[t] * s * np.exp(rng.normal(0, 0.02))
                rows.append(row)
    return pd.DataFrame(rows).set_index(["mo", "ds"]).sort_index()


@pytest.fixture(scope="module")
def cube():
    return synthetic_cube()


def truth(index):
    return np.array([int(s[1]) for s in index])


def test_features_shapes_and_no_lookahead(cube):
    X, blocks = F.snapshot_features(cube, "2024-06-01", TOT, CATS, 3)
    assert set(blocks) == {"level", "structure", "growth"}
    assert len(X) == 120 and X.notna().all().all()
    # признаки снимка не зависят от данных после него
    cut = cube[cube.index.get_level_values("ds") <= "2024-06-01"]
    X2, _ = F.snapshot_features(cut, "2024-06-01", TOT, CATS, 3)
    pd.testing.assert_frame_equal(X.sort_index(), X2.sort_index())


def test_growth_absent_before_2024(cube):
    X, blocks = F.snapshot_features(cube, "2023-12-01", TOT, CATS, 3)
    assert blocks["growth"] == []


def test_standardize_drops_zero_weight_block(cube):
    X, blocks = F.snapshot_features(cube, "2024-06-01", TOT, CATS, 3)
    Z = F.standardize(X, blocks, {"level": 1, "structure": 1, "growth": 0})
    assert not any(c.startswith("growth") for c in Z.columns)


@pytest.mark.parametrize("rule", ["attr", "comove", "hybrid"])
def test_graph_symmetric_no_self_loops(cube, rule):
    X, blocks = F.snapshot_features(cube, "2024-06-01", TOT, CATS, 3)
    Z = F.standardize(X, blocks, {"level": 1, "structure": 1, "growth": 1})
    M = F.comove_matrix(cube, "2024-06-01", TOT, CATS, 12)
    W = G.build(rule, Z, M, 8)
    assert (abs(W - W.T) > 1e-12).nnz == 0
    assert W.diagonal().sum() == 0
    assert W.min() >= 0


@pytest.mark.parametrize("rule", ["attr", "comove", "hybrid"])
def test_recovers_planted_groups(cube, rule):
    X, blocks = F.snapshot_features(cube, "2024-12-01", TOT, CATS, 3)
    Z = F.standardize(X, blocks, {"level": 1, "structure": 1, "growth": 1})
    M = F.comove_matrix(cube, "2024-12-01", TOT, CATS, 12)
    W = G.build(rule, Z, M, 8)
    cfg = {"cluster": {"resolution": 0.5, "n_restarts": 3, "min_size": 5}}
    lab = C.run("leiden", W, Z.to_numpy(), None, cfg, 0)
    assert C.ari(lab, truth(Z.index)) > 0.9


def test_merge_small(cube):
    X, blocks = F.snapshot_features(cube, "2024-12-01", TOT, CATS, 3)
    Z = F.standardize(X, blocks, {"level": 1, "structure": 1, "growth": 1})
    W = G.knn_attr(Z, 8)
    lab = truth(Z.index).copy()
    lab[:3] = 9
    out = C.merge_small(lab, W, Z.to_numpy(), 5)
    assert np.bincount(out).min() >= 5


def test_track_keeps_identity_under_relabel():
    idx = [f"m{i}" for i in range(30)]
    a = pd.Series([0] * 10 + [1] * 10 + [2] * 10, index=idx)
    b = a.map({0: 2, 1: 0, 2: 1})
    L, ev = Y.track({pd.Timestamp("2024-01-01"): a, pd.Timestamp("2024-02-01"): b})
    assert (L.iloc[:, 0] == L.iloc[:, 1]).all()
    assert ev.empty


def test_track_detects_merge():
    idx = [f"m{i}" for i in range(30)]
    a = pd.Series([0] * 10 + [1] * 10 + [2] * 10, index=idx)
    b = pd.Series([0] * 20 + [1] * 10, index=idx)
    _, ev = Y.track({pd.Timestamp("2024-01-01"): a, pd.Timestamp("2024-02-01"): b})
    assert "слияние" in set(ev.event)


def test_group_factor_single_group_equals_national(cube):
    wide = F.wide(cube, TOT)
    o = pd.Timestamp("2024-06-01")
    one = pd.Series(0, index=wide.index)
    p1 = V.group_factor_forecast(wide, one, o, [1, 2, 3], 3, "group")
    p2 = V.group_factor_forecast(wide, one, o, [1, 2, 3], 3, "national")
    np.testing.assert_allclose(p1.to_numpy(), p2.to_numpy(), rtol=1e-12)


def test_group_factor_uses_only_past(cube):
    wide = F.wide(cube, TOT)
    o = pd.Timestamp("2024-06-01")
    lab = pd.Series(truth(wide.index), index=wide.index)
    p1 = V.group_factor_forecast(wide, lab, o, [1, 2, 3], 3)
    w2 = wide.copy()
    w2.loc[:, w2.columns > o] *= 10
    p2 = V.group_factor_forecast(w2, lab, o, [1, 2, 3], 3)
    np.testing.assert_allclose(p1.to_numpy(), p2.to_numpy())


def test_true_groups_beat_placebo(cube):
    wide = F.wide(cube, TOT)
    lab = pd.Series(truth(wide.index), index=wide.index)
    parts = {("truth", o): lab for o in ["2024-06-01", "2024-07-01"]}
    S, _ = V.evaluate(wide, parts, ["2024-06-01", "2024-07-01"], [1, 2, 3], 3, 10, 0, "national")
    r = S.set_index("partition").loc["truth"]
    assert r["MAE"] < r["placebo_MAE_mean"]
