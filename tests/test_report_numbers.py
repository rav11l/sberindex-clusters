"""Сверка чисел отчёта с прогоном. Пропускается, если outputs/summary.json нет (прогон не делался)."""
import json
from pathlib import Path

import pytest

S_PATH = Path("outputs/summary.json")
pytestmark = pytest.mark.skipif(not S_PATH.exists(), reason="нет outputs/summary.json: запустите python -m sclusters.cli all")


@pytest.fixture(scope="module")
def S():
    return json.loads(S_PATH.read_text(encoding="utf-8"))


def test_passport(S):
    p = S["passport"]
    assert p["mo_kept"] == 1904 and p["ambiguous_names"] == 49 and p["mo_dropped_incomplete"] == 165
    assert p["sha256"].startswith("dbe07d6a")


def test_rules_disagree(S):
    assert S["agreement_leiden"]["attr/leiden|comove/leiden"] == pytest.approx(0.16, abs=0.01)


def test_main_partition(S):
    m = S["main"]
    assert (m["variant"], m["rule"], m["method"], m["resolution"]) == ("no_growth", "hybrid", "leiden", 0.5)
    assert m["bootstrap_ari"] == pytest.approx(0.88, abs=0.01)
    assert m["temporal_ari"] == pytest.approx(0.86, abs=0.01)
    assert m["vs_national_%"] == pytest.approx(3.9, abs=0.1)


def test_dynamics(S):
    assert S["churn_mean"]["evolutionary"] == pytest.approx(0.019, abs=0.001)
    assert S["churn_mean"]["independent"] == pytest.approx(0.059, abs=0.001)
    assert S["mo_never_moved_share"] == pytest.approx(0.902, abs=0.001)
    assert S["events_evolutionary"] == 0
    assert S["flood_cities_constant"] and S["n_flood_cities"] == 5
    assert S["seed_noise_moved_mean"] == pytest.approx(0.017, abs=0.001)


def test_geography(S):
    e = S["geo_edges"]["no_fed_cities|comove"]
    assert e["same_region_share"] == pytest.approx(0.214, abs=0.002)
    assert e["median_km"] == pytest.approx(611, abs=2)
    assert S["geo_match"]["ok"] == 1744
    assert e["same_region_random_same_maintype"] == pytest.approx(0.082, abs=0.002)
    assert e["same_region_random_same_motype"] == pytest.approx(0.057, abs=0.002)


def test_profiles(S):
    P = {r["line"]: r for r in S["profiles_last"]}
    assert [P[i]["n"] for i in range(4)] == [1066, 302, 346, 190]
    assert round(P[1]["share:Общественное питание"], 3) == 0.072
    assert S["national_MAE"] == pytest.approx(954.5, abs=0.1)
