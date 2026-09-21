"""Orchestration: prepare the shared context once, run every applicable task for
every adapter, collect records. One model failing never aborts the others."""
from __future__ import annotations

import logging
import time
import traceback
from typing import Optional, Sequence

from . import progress
from .adapters.base import MobilityModelAdapter
from .context import EvalConfig, EvalContext
from .data import MobilityDataset
from .results import ResultStore
from .tasks import Task, default_tasks

log = logging.getLogger("mobeval")


def _identity(rec):
    """What makes a result row unique. Two rows with the same identity are the same measurement."""
    return (rec.model, rec.run_tag, rec.task, rec.metric, rec.eval_seed, rec.protocol)


class EvaluationPipeline:
    def __init__(self, config: Optional[EvalConfig] = None, tasks: Optional[Sequence[Task]] = None,
                 strict: bool = False):
        self.cfg = config or EvalConfig()
        self.tasks = list(tasks) if tasks is not None else default_tasks(self.cfg)
        self.strict = strict

    def prepare(self, dataset: MobilityDataset) -> EvalContext:
        rep = progress.get()
        with rep.stage("prepare", f"preparing {dataset.name}: splits, windows, staypoints"):
            ctx = EvalContext(dataset, self.cfg)
        log.info("\n" + ctx.summary())
        rep.event("prepared", message=f"prepared {dataset.name} ({ctx.fingerprint})",
                  fingerprint=ctx.fingerprint,
                  sizes={k: len(v.points) for k, v in ctx.splits.items()})
        return ctx

    def run(self, adapters: Sequence[MobilityModelAdapter], ctx: EvalContext,
            store: Optional[ResultStore] = None) -> ResultStore:
        store = store or ResultStore()
        rep = progress.get()
        already = store.done_tasks()
        # Baseline rows are emitted once per task by whichever adapter reaches it first, tracked
        # in `ctx.emitted_baselines`, which a resumed process starts with empty. The tasks use
        # more than one key shape for that set, so rather than reconstructing each of them, a
        # resumed run drops any record whose identity is already in the kept results. That also
        # covers whatever a future task invents.
        seen = {_identity(r) for r in store.records} if already else None
        ctx.resumed_units = already
        if already:
            log.info(f"resuming: {len(already)} (model, task) pair(s) already have results")
        plan = [(a, t) for a in adapters for t in self.tasks if t.applicable(a)]
        rep.set_plan(rep.state.get("steps_total", 0) + len(plan) if rep.active else len(plan))
        for adapter in adapters:
            key = f"{adapter.name}@{adapter.run_tag}"
            n_tasks = sum(1 for a, t in plan if a is adapter)
            rep.model(adapter.name, state="evaluating", detail=f"0/{n_tasks} tasks")
            done = 0
            with rep.scoped(adapter.name):
                for task in self.tasks:
                    if not task.applicable(adapter):
                        ctx.skipped.append((key, task.name, "capability not declared"))
                        rep.count(skipped=1)
                        continue
                    if (adapter.name, adapter.run_tag, task.name) in already:
                        log.info(f"[{key}] {task.name}: already done, keeping the earlier result")
                        done += 1
                        rep.step_done()
                        rep.model(adapter.name, detail=f"{done}/{n_tasks} tasks")
                        continue
                    log.info(f"[{key}] {task.name}")
                    t0 = time.time()
                    rep.set_phase(f"{adapter.name}: {task.name}")
                    rep.event("task_start", model=adapter.name, task=task.name)
                    try:
                        recs = task.run(adapter, ctx)
                    except progress.Interrupted:
                        # A walltime kill is not a task failure, and there is no point starting
                        # the next task: SIGKILL follows shortly. Everything finished is on disk.
                        rep.model(adapter.name, state="interrupted", detail=f"stopped during {task.name}")
                        raise
                    except Exception as e:                     # noqa: BLE001
                        if self.strict:
                            rep.error(f"{key} {task.name}: {e!r}", model=adapter.name, task=task.name)
                            raise
                        ctx.errors.append((key, task.name, repr(e), traceback.format_exc()))
                        log.error(f"[{adapter.name}] {task.name} failed: {e!r}")
                        rep.error(f"{key} {task.name}: {e!r}", model=adapter.name, task=task.name)
                    else:
                        for r in recs:
                            r.task_unit = task.name     # what a resumed run checks off
                        if seen is not None:
                            recs = [r for r in recs if _identity(r) not in seen]
                            seen.update(_identity(r) for r in recs)
                        store.extend(recs)
                        store.persist()                 # this task's numbers are now safe on disk
                        rep.count(records=len(recs))
                        rep.event("task_end", model=adapter.name, task=task.name, records=len(recs),
                                  seconds=time.time() - t0,
                                  message=f"{adapter.name} · {task.name}: {len(recs)} records "
                                          f"({time.time() - t0:.0f}s)")
                    done += 1
                    rep.step_done()
                    rep.model(adapter.name, detail=f"{done}/{n_tasks} tasks")
            rep.model(adapter.name, state="evaluated", detail=f"{done}/{n_tasks} tasks")
        return store
