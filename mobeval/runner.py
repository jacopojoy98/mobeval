"""Orchestration: prepare the shared context once, run every applicable task for
every adapter, collect records. One model failing never aborts the others."""
from __future__ import annotations

import logging
import traceback
from typing import Optional, Sequence

from .adapters.base import MobilityModelAdapter
from .context import EvalConfig, EvalContext
from .data import MobilityDataset
from .results import ResultStore
from .tasks import Task, default_tasks

log = logging.getLogger("mobeval")


class EvaluationPipeline:
    def __init__(self, config: Optional[EvalConfig] = None, tasks: Optional[Sequence[Task]] = None,
                 strict: bool = False):
        self.cfg = config or EvalConfig()
        self.tasks = list(tasks) if tasks is not None else default_tasks(self.cfg)
        self.strict = strict

    def prepare(self, dataset: MobilityDataset) -> EvalContext:
        ctx = EvalContext(dataset, self.cfg)
        log.info("\n" + ctx.summary())
        return ctx

    def run(self, adapters: Sequence[MobilityModelAdapter], ctx: EvalContext,
            store: Optional[ResultStore] = None) -> ResultStore:
        store = store or ResultStore()
        for adapter in adapters:
            for task in self.tasks:
                if not task.applicable(adapter):
                    ctx.skipped.append((f"{adapter.name}@{adapter.run_tag}", task.name, "capability not declared"))
                    continue
                log.info(f"[{adapter.name}@{adapter.run_tag}] {task.name}")
                try:
                    store.extend(task.run(adapter, ctx))
                except Exception as e:                         # noqa: BLE001
                    if self.strict:
                        raise
                    ctx.errors.append((f"{adapter.name}@{adapter.run_tag}", task.name, repr(e),
                                       traceback.format_exc()))
                    log.error(f"[{adapter.name}] {task.name} failed: {e!r}")
        return store
