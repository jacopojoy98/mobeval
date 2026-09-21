"""Result records with a fixed schema + automatic sanity validation."""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

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
    task_unit: str = ""            # the pipeline task that produced it; the unit a resumed run skips
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
    """Holds result records, and can keep a file on disk in step with them.

    A long evaluation is dozens of (model, task) pairs, any one of which can take minutes.
    Writing `results.jsonl` only at the end meant a crash in the last task threw away every
    number computed before it. With `bind(path)`, the file is rewritten after every completed
    task, so what is on disk is always exactly the set of finished tasks - never a half-written
    one - and `--resume` can pick up from there.

    The file is rewritten in full rather than appended to. It is a few hundred kilobytes at
    most, and a whole-file atomic replace cannot leave a partially committed task behind,
    which an append can.
    """

    def __init__(self, path: Optional[str] = None, resume: bool = False):
        self.records: List[ResultRecord] = []
        self.path: Optional[Path] = None
        self.resumed = 0
        self.on_persist = None                 # set by the runner to mirror to durable storage
        if path:
            self.bind(path, resume=resume)

    # --------------------------------------------------------------- persistence
    def bind(self, path, resume: bool = False):
        """Keep `path` in step with this store. With resume=True, adopt what is already there."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if resume:
                self.records.extend(_read_jsonl(path))
                self.resumed = len(self.records)
            else:
                path.replace(path.with_name(path.name + ".previous"))
        self.path = path
        self.persist()
        return self

    def persist(self) -> Optional[Path]:
        """Atomically rewrite the bound file from the records held now."""
        if self.path is None:
            return None
        tmp = self.path.with_name(self.path.name + f".tmp{os.getpid()}")
        with open(tmp, "w") as f:
            for r in self.records:
                f.write(json.dumps(dataclasses.asdict(r), default=float) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())           # survive the node going away, not just the process
            except OSError:
                pass
        os.replace(tmp, self.path)
        if self.on_persist is not None:
            self.on_persist(self.path)
        return self.path

    def done_tasks(self) -> Set[tuple]:
        """(model, run_tag, task unit) triples already recorded - what a resumed run skips."""
        return {(r.model, r.run_tag, r.task_unit) for r in self.records if r.task_unit}

    # --------------------------------------------------------------- records
    def add(self, rec: ResultRecord):
        self.records.append(rec)

    def extend(self, recs):
        self.records.extend(recs)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(r) for r in self.records])

    def save(self, path_jsonl: str):
        if self.path is not None and Path(path_jsonl) == self.path:
            self.persist()
            return
        with open(path_jsonl, "w") as f:
            for r in self.records:
                f.write(json.dumps(dataclasses.asdict(r), default=float) + "\n")

    @classmethod
    def load(cls, path_jsonl: str) -> "ResultStore":
        s = cls()
        s.records.extend(_read_jsonl(path_jsonl))
        return s


def _read_jsonl(path) -> List[ResultRecord]:
    """Read result records, tolerating a truncated last line.

    A journal written by a job that was killed mid-write can end in a partial line. That is
    one lost record, not a corrupt file, so it is dropped with a warning rather than raising.
    """
    out, bad = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            flags = d.pop("flags", [])
            for k in ("unit", "higher_is_better", "family"):
                d.pop(k, None)
            try:
                r = ResultRecord(**d)
            except TypeError:
                bad += 1
                continue
            r.flags = flags
            out.append(r)
    if bad:
        logging.getLogger("mobeval").warning(
            f"{path}: skipped {bad} unreadable line(s) - the run was probably interrupted while writing")
    return out
