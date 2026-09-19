"""Result records with a fixed schema + automatic sanity validation."""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .metrics.registry import get_spec


@dataclass
class ResultRecord:
    model: str                     # model name (e.g. "UniTraj")
    run_tag: str                   # checkpoint / training seed tag; aggregated over in reports
    task: str                      # e.g. "recovery/block@0.5"
    metric: str                    # registered metric name, e.g. "ade_m"
    value: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    n: int = 0
    protocol: str = "native"       # native | linear_probe | ...
    baseline: Optional[str] = None
    baseline_value: Optional[float] = None
    skill: Optional[float] = None
    skill_ci_low: Optional[float] = None
    skill_ci_high: Optional[float] = None
    eval_seed: int = 0
    dataset: str = ""
    split: str = "test"
    unit: str = ""
    higher_is_better: Optional[bool] = None
    family: str = ""
    flags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        spec = get_spec(self.metric)
        self.unit, self.higher_is_better, self.family = spec.unit, spec.higher_is_better, spec.family
        self.value = None if self.value is None else float(self.value)
        self.validate(spec)

    def validate(self, spec):
        v = self.value
        if v is None or not np.isfinite(v):
            self.flags.append("non-finite value")
            return
        lo, hi = spec.valid_range
        if not (lo - 1e-9 <= v <= hi + 1e-9):
            self.flags.append(f"outside valid range [{lo}, {hi}] - check scale (e.g. % vs fraction)")
        if spec.plausible_max is not None and v > spec.plausible_max:
            self.flags.append(f"implausibly large (> {spec.plausible_max} {spec.unit}) - check units/aggregation")


class ResultStore:
    def __init__(self):
        self.records: List[ResultRecord] = []

    def add(self, rec: ResultRecord):
        self.records.append(rec)

    def extend(self, recs):
        self.records.extend(recs)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(r) for r in self.records])

    def save(self, path_jsonl: str):
        with open(path_jsonl, "w") as f:
            for r in self.records:
                f.write(json.dumps(dataclasses.asdict(r), default=float) + "\n")

    @classmethod
    def load(cls, path_jsonl: str) -> "ResultStore":
        s = cls()
        with open(path_jsonl) as f:
            for line in f:
                d = json.loads(line)
                flags = d.pop("flags", [])
                for k in ("unit", "higher_is_better", "family"):
                    d.pop(k, None)
                r = ResultRecord(**d)
                r.flags = flags
                s.add(r)
        return s
