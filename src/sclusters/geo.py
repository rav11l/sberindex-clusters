"""Привязка МО к региону и координатам — внешний источник, в признаки и рёбра не входит.

В выгрузке СберИндекса нет ни региона, ни кода ОКТМО: только название МО. Регион нужен для двух
вещей: проверить, восстанавливает ли сеть географию, которой в данных нет, и нарисовать карту.
Источник — границы муниципальных образований из OpenStreetMap (admin_level = 6), собранные
в репозитории github.com/timurkanaz/Russia_geojson_OSM (данные © участники OpenStreetMap, ODbL).
Версия закреплена коммитом, геометрия в репозиторий проекта не копируется.

Сопоставление по «ядру» названия: из названия убираются типовые слова («муниципальный район»,
«городской округ», «город», «район» и т. п.), сравниваются остатки. Ядро, встречающееся в OSM
в нескольких регионах, считается неоднозначным, и МО остаётся без региона.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "https://github.com/timurkanaz/Russia_geojson_OSM.git"
COMMIT = "5af46d6dab9a502713611846892e4185f157dab7"

STOP = [
    "внутригородская территория города федерального значения", "внутригородская территория",
    "закрытое административно-территориальное образование", "зато",
    "муниципальное образование", "муниципальный район", "муниципальный округ", "городской округ",
    "сельское поселение", "городское поселение", "рабочий поселок", "рабочий посёлок", "поселение",
    "поселок", "посёлок", "город-курорт", "город", "район", "округ", "наукоград", "г.", "пгт",
]

FED_CITIES = {"Москва_Moscow": "г. Москва", "Санкт-Петербург_Saint-Petersburg": "г. Санкт-Петербург",
              "Севастополь_Sevastopol": "г. Севастополь", "Казань_Kazan": None}


def core(name: str) -> str:
    s = name.lower().replace("ё", "е")
    s = re.sub(r"[«»\"()]", " ", s)
    for w in STOP:
        s = re.sub(rf"(?<![а-яa-z]){re.escape(w.replace('ё', 'е'))}(?![а-яa-z])", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -")
    return s


def fetch(dest: str | Path) -> Path:
    dest = Path(dest)
    if not (dest / ".git").exists():
        subprocess.run(["git", "clone", "-q", REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", COMMIT], check=True)
    return dest


def _ring_centroid(ring: np.ndarray) -> tuple[float, float, float]:
    x, y = ring[:, 0], ring[:, 1]
    a = x[:-1] * y[1:] - x[1:] * y[:-1]
    A = a.sum() / 2
    if abs(A) < 1e-12:
        return float(x.mean()), float(y.mean()), 0.0
    cx = ((x[:-1] + x[1:]) * a).sum() / (6 * A)
    cy = ((y[:-1] + y[1:]) * a).sum() / (6 * A)
    return float(cx), float(cy), float(abs(A))


def centroid(geom: dict) -> tuple[float, float]:
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    pts = [_ring_centroid(np.asarray(p[0], dtype=float)) for p in polys if len(p) and len(p[0]) > 2]
    w = np.array([p[2] for p in pts])
    if w.sum() == 0:
        return pts[0][0], pts[0][1]
    return float(np.average([p[0] for p in pts], weights=w)), float(np.average([p[1] for p in pts], weights=w))


def osm_table(root: str | Path) -> pd.DataFrame:
    """Все МО из OSM: регион, федеральный округ, название, ядро, центроид."""
    root = Path(root) / "GeoJson's"
    rows = []
    for f in sorted((root / "Regions").glob("*/*.geojson")):
        region = f.stem.split("_")[0]
        for x in json.loads(f.read_text(encoding="utf-8"))["features"]:
            lon, lat = centroid(x["geometry"])
            rows.append({"region": region, "fd": f.parent.name, "osm_name": x["properties"]["district"],
                         "lon": lon, "lat": lat})
    for f in sorted((root / "Cities").glob("*.geojson")):
        region = FED_CITIES.get(f.stem)
        if region is None:
            continue
        for x in json.loads(f.read_text(encoding="utf-8"))["features"]:
            lon, lat = centroid(x["geometry"])
            rows.append({"region": region, "fd": "fed_city", "osm_name": x["properties"]["district"],
                         "lon": lon, "lat": lat})
    T = pd.DataFrame(rows)
    T["core"] = T["osm_name"].map(core)
    return T


def match(mo_names, T: pd.DataFrame) -> pd.DataFrame:
    """Название МО СберИндекса -> регион и центроид. Неоднозначные и ненайденные — NaN."""
    is_city = {n: n.startswith("внутригородская территория") for n in mo_names}
    out = []
    city_T = T[T.fd == "fed_city"]
    reg_T = T[T.fd != "fed_city"]
    for n in mo_names:
        c = core(n)
        cand = (city_T if is_city[n] else reg_T)
        hit = cand[cand.core == c]
        status = "ok"
        if hit["region"].nunique() > 1:
            status, hit = "ambiguous", hit.iloc[0:0]
        elif len(hit) == 0:
            status = "not_found"
        r = hit.iloc[0] if len(hit) else None
        out.append({"mo": n, "core": c, "status": status,
                    "region": r["region"] if r is not None else np.nan,
                    "fd": r["fd"] if r is not None else np.nan,
                    "lon": r["lon"] if r is not None else np.nan,
                    "lat": r["lat"] if r is not None else np.nan})
    M = pd.DataFrame(out).set_index("mo")
    # внутригородские территории без пары в OSM: регион по признакам названия не восстановить,
    # но федеральный город известен, если ядро нашлось хотя бы в одном городе
    return M
