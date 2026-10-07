"""Загрузка выгрузки СберИндекса и сборка панели МО × категория × месяц.

Чтение и восстановление рядов повторяют sberindex-shocks (src/sshocks/data.py): в выгрузке нет
идентификатора ряда, ряд — это сплошной блок строк одной категории и одного МО. Для одного
ряда этого достаточно. Здесь же ряды шести категорий сшиваются по названию МО, и для
одноимённых МО разных регионов сшивка не проверяема: номер блока в одной категории не обязан
соответствовать номеру блока в другой. Такие МО по умолчанию исключаются (data.drop_ambiguous_names).
"""
from __future__ import annotations

import hashlib
import io
import json
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij"
API_BASE = "https://sberindex.ru/api/dataset/v1"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes(raw_dir: str | Path, hashes_file: str | Path) -> dict:
    raw_dir = Path(raw_dir)
    recorded = json.loads(Path(hashes_file).read_text(encoding="utf-8"))
    status = {}
    for name, meta in recorded["files"].items():
        p = raw_dir / name
        status[name] = "missing" if not p.exists() else ("ok" if sha256(p) == meta["sha256"] else "mismatch")
    return status


def download(raw_dir: str | Path, fmt: str = "parquet") -> Path:
    """Выгрузка через публичный API sberindex.ru (нужен заголовок RqUID).
    Сайт подписан корневым сертификатом Минцифры: truststore берёт сертификаты из хранилища ОС."""
    import urllib.request
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = f"{API_BASE}/download/{DATASET_ID}/{fmt}"
    req = urllib.request.Request(url, headers={"RqUID": uuid.uuid4().hex, "User-Agent": "sberindex-clusters"})
    out = raw_dir / ("sberindex_mo_spending.parquet" if fmt == "parquet" else "sberindex_mo_spending.csv.zip")
    with urllib.request.urlopen(req, timeout=120) as r:
        out.write_bytes(r.read())
    return out


def read_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.name.endswith(".csv.zip"):
        with zipfile.ZipFile(path) as z:
            df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), sep=";", dtype=str)
        df["value"] = pd.to_numeric(df["value"])
    else:
        raise ValueError(f"Неизвестный формат: {path}")
    df["period"] = pd.to_datetime(df["period"])
    return df


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Сырые строки -> длинная таблица; ряд = сплошной блок строк одной категории и МО."""
    key = df["category_15"] + "|" + df["mo"]
    new_block = (key != key.shift()) | (df["period"] <= df["period"].shift())
    df = df.assign(block=new_block.cumsum())
    first = df.groupby("block", sort=False)[["category_15", "mo"]].first()
    first["dup_rank"] = first.groupby(["category_15", "mo"]).cumcount()
    first["n_same_name"] = first.groupby(["category_15", "mo"])["dup_rank"].transform("size")
    out = df.merge(first[["dup_rank", "n_same_name"]], left_on="block", right_index=True)
    out = out.rename(columns={"category_15": "category", "period": "ds", "value": "y"})
    return out[["category", "mo", "dup_rank", "n_same_name", "ds", "y"]].reset_index(drop=True)


def mo_type(name: str) -> str:
    """Тип МО по названию — внешний описатель для интерпретации кластеров, в признаки не входит."""
    if name.startswith("внутригородская территория"):
        return "внутригородская территория"
    if name.startswith("городской округ город") or name.startswith("городской округ город-"):
        return "город (ГО)"
    if "городской округ" in name:
        return "городской округ"
    if "муниципальный округ" in name:
        return "муниципальный округ"
    if "муниципальный район" in name:
        return "муниципальный район"
    return "прочее"


def build_cube(long: pd.DataFrame, total: str, categories: list[str], drop_ambiguous: bool = True):
    """Возвращает (cube, passport).

    cube — DataFrame с MultiIndex (mo, ds) и столбцами total + categories; только МО с полной
    историей во всех шести рядах и положительными значениями.
    """
    passport = {"rows": int(len(long)), "mo_names": int(long["mo"].nunique())}
    amb = long.loc[long["n_same_name"] > 1, "mo"].unique()
    passport["ambiguous_names"] = int(len(amb))
    if drop_ambiguous:
        long = long[~long["mo"].isin(amb)]
    else:
        long = long.assign(mo=long["mo"] + np.where(long["n_same_name"] > 1, " #" + long["dup_rank"].astype(str), ""))
    cols = [total] + list(categories)
    cube = long[long["category"].isin(cols)].pivot_table(index=["mo", "ds"], columns="category", values="y")[cols]
    months = cube.index.get_level_values("ds").unique().sort_values()
    ok = cube.notna().all(axis=1) & (cube > 0).all(axis=1)
    n_ok = ok.groupby(level="mo").sum()
    keep = n_ok[n_ok == len(months)].index
    cube = cube.loc[cube.index.get_level_values("mo").isin(keep)].sort_index()
    passport.update({
        "months": [str(months.min().date()), str(months.max().date())],
        "n_months": int(len(months)),
        "mo_unique_names": int(long["mo"].nunique()),
        "mo_kept": int(len(keep)),
        "mo_dropped_incomplete": int(long["mo"].nunique() - len(keep)),
    })
    return cube, passport


def load(cfg: dict):
    d = cfg["data"]
    raw = Path(d["raw_dir"]) / d["file"]
    if not raw.exists():
        raise FileNotFoundError(f"{raw} не найден. Запустите: python -m sclusters.cli data --download")
    long = to_long(read_raw(raw))
    return build_cube(long, d["total"], d["categories"], d.get("drop_ambiguous_names", True))
