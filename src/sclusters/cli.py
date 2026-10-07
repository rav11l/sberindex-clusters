"""Командная строка.

python -m sclusters.cli data --download      # выгрузка с sberindex.ru и сверка sha256
python -m sclusters.cli all                  # все стадии и рисунки
python -m sclusters.cli dynamics figures     # отдельные стадии
python -m sclusters.cli all --set graph.k=15 cluster.resolution=0.75
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import yaml


def _set(cfg: dict, assignments: list[str]) -> dict:
    for a in assignments or []:
        key, val = a.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sclusters")
    ap.add_argument("stages", nargs="+", help="data | snapshots | compare | stability | validate | dynamics | figures | all")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = _set(yaml.safe_load(Path(a.config).read_text(encoding="utf-8")), a.set)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    from . import data as D

    if "data" in a.stages and a.download:
        D.download(cfg["data"]["raw_dir"], "parquet")
    if "data" in a.stages or "all" in a.stages:
        status = D.verify_hashes(cfg["data"]["raw_dir"], cfg["data"]["hashes"])
        print(json.dumps(status, ensure_ascii=False))
        if status.get(cfg["data"]["file"]) == "mismatch":
            print("Контрольная сумма не совпала: выгрузка отличается от проверенной.", file=sys.stderr)
    from . import pipeline as P
    from . import figures as Fg

    stages = list(P.STAGES) if "all" in a.stages else [s for s in a.stages if s in P.STAGES]
    if stages:
        P.run_all(cfg, stages)
    if "all" in a.stages or "figures" in a.stages:
        Fg.make_all(cfg)


if __name__ == "__main__":
    main()
