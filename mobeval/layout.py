"""Where a run's files live, and how they survive the job dying.

Three separate ideas, deliberately kept apart:

*run directory* - every invocation writes its report, leaderboard and results into
    ``<output_dir>/runs/<run_id>/``, so two jobs sharing an output directory never
    overwrite each other's numbers. ``<output_dir>/latest`` points at the newest one.

*checkpoint directory* - ``<output_dir>/checkpoints/`` is deliberately NOT per-run.
    Training is the expensive part; a second run must be able to reuse the first
    run's weights, so the checkpoints live in one stable place.

*persist directory* - the cluster makes jobs compute in node-local scratch, which is
    wiped when the job ends and is lost entirely if the job is killed. Anything
    durable is therefore copied into ``persist_dir`` (a home/project folder) the
    moment it is written, rather than at the end of the job: a checkpoint lands
    there as soon as its epoch improves, and results land there after every task.
    The mirror keeps the same relative layout, so ``persist_dir`` ends up looking
    exactly like ``output_dir``.

A single Layout is made active for the process (like ``progress``), because the
mirroring hook has to be reachable from deep inside the training loop.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("mobeval.layout")

RUNS = "runs"
CHECKPOINTS = "checkpoints"
LATEST = "latest"


def make_run_id(command: str = "run") -> str:
    """Same shape as a progress run id, so a run directory and its progress files match."""
    job = os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID") or ""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{job.split('.')[0] or os.getpid()}-{command}"


@dataclass
class Layout:
    output_dir: Path
    run_id: str
    persist_dir: Optional[Path] = None
    run_dirs: bool = True
    checkpoint_dir_override: Optional[Path] = None
    mirrored: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.persist_dir is not None:
            self.persist_dir = Path(self.persist_dir)
            # Mirroring a directory onto itself would be a no-op at best and a self-copy at worst.
            if self.persist_dir.resolve() == self.output_dir.resolve():
                self.persist_dir = None
        if self.checkpoint_dir_override is not None:
            self.checkpoint_dir_override = Path(self.checkpoint_dir_override)

    # ------------------------------------------------------------------ places
    @property
    def run_dir(self) -> Path:
        return self.output_dir / RUNS / self.run_id if self.run_dirs else self.output_dir

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoint_dir_override or self.output_dir / CHECKPOINTS

    def checkpoint(self, name: str) -> Path:
        return self.checkpoint_dir / f"{name}.pt"

    def create(self) -> "Layout":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.run_dirs:
            _point_latest(self.output_dir, self.run_id)
        if self.persist_dir is not None:
            try:
                (self.persist_dir / CHECKPOINTS).mkdir(parents=True, exist_ok=True)
                if self.run_dirs:
                    (self.persist_dir / RUNS / self.run_id).mkdir(parents=True, exist_ok=True)
                    _point_latest(self.persist_dir, self.run_id)
            except OSError as e:
                # Worth shouting about - the whole point of persist_dir is surviving the job -
                # but not worth refusing to compute over.
                log.error(f"persist_dir {self.persist_dir} is not usable ({e}); results will only be "
                          f"written to {self.output_dir}, which may be wiped when the job ends")
        return self

    # ------------------------------------------------------------------ mirroring
    def _twin(self, path: Path) -> Optional[Path]:
        """Where `path` belongs inside persist_dir, or None if it is not ours to mirror."""
        if self.persist_dir is None:
            return None
        try:
            rel = Path(path).resolve().relative_to(self.output_dir.resolve())
        except ValueError:
            return None                          # outside output_dir (an explicit `checkpoint:` path)
        return self.persist_dir / rel

    def mirror(self, path) -> Optional[Path]:
        """Copy a finished file into the durable directory. Best effort: a full or unwritable
        persist directory must never take down a training run that is otherwise going fine."""
        path = Path(path)
        dest = self._twin(path)
        if dest is None or not path.exists():
            return None
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + f".tmp{os.getpid()}")
            shutil.copyfile(path, tmp)
            os.replace(tmp, dest)                # readers never see a half-copied file
        except OSError as e:
            log.warning(f"could not mirror {path.name} to {dest.parent}: {e}")
            return None
        if str(dest) not in self.mirrored:
            self.mirrored.append(str(dest))
        return dest

    def restore_checkpoints(self) -> List[str]:
        """Bring back checkpoints an earlier job mirrored, so this one does not retrain them."""
        if self.persist_dir is None:
            return []
        src = self.persist_dir / CHECKPOINTS
        if not src.is_dir():
            return []
        restored = []
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            if f.suffix not in (".pt", ".json") or not f.is_file():
                continue
            dest = self.checkpoint_dir / f.name
            if dest.exists() and dest.stat().st_mtime >= f.stat().st_mtime:
                continue
            try:
                shutil.copyfile(f, dest)
            except OSError as e:
                log.warning(f"could not restore {f.name}: {e}")
                continue
            if f.suffix == ".pt":
                restored.append(f.stem)
        if restored:
            log.info(f"restored checkpoints from {src}: {', '.join(restored)}")
        return restored

    def restore_run(self) -> List[str]:
        """Bring this run id's files back from durable storage.

        The case this exists for: a job dies, and the next job starts on a different node with
        an empty scratch. The results of the finished tasks are only in `persist_dir`, so
        `--resume` would otherwise find nothing to resume from.
        """
        if self.persist_dir is None or not self.run_dirs:
            return []
        src = self.persist_dir / RUNS / self.run_id
        if not src.is_dir():
            return []
        restored = []
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            if not f.is_file():
                continue
            dest = self.run_dir / f.name
            if dest.exists() and dest.stat().st_mtime >= f.stat().st_mtime:
                continue
            try:
                shutil.copyfile(f, dest)
            except OSError as e:
                log.warning(f"could not restore {f.name}: {e}")
                continue
            restored.append(f.name)
        if restored:
            log.info(f"restored run {self.run_id} from {src}: {', '.join(restored)}")
        return restored

    def previous_runs(self) -> List[Path]:
        """Existing run directories, newest last, from scratch and from durable storage both.
        Used by --resume; a run that only survives in persist_dir still counts."""
        seen = {}
        for base in ([self.persist_dir / RUNS] if self.persist_dir else []) + [self.output_dir / RUNS]:
            if base.is_dir():
                for d in base.iterdir():
                    if d.is_dir():
                        seen[d.name] = d                # the local copy wins if both exist
        return [seen[k] for k in sorted(seen)]

    def describe(self) -> str:
        lines = [f"run id:      {self.run_id}",
                 f"run dir:     {self.run_dir}",
                 f"checkpoints: {self.checkpoint_dir}  (shared by all runs)"]
        lines.append(f"persisted:   {self.persist_dir}" if self.persist_dir
                     else "persisted:   (none - set persist_dir to keep results if the job dies)")
        return "\n".join(lines)


def _point_latest(parent: Path, run_id: str):
    """`<parent>/latest` -> `runs/<run_id>`, as a relative symlink so it survives being copied."""
    link = parent / LATEST
    try:
        target = Path(RUNS) / run_id
        tmp = parent / f".{LATEST}.tmp{os.getpid()}"
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(target, target_is_directory=True)
        os.replace(tmp, link)
    except (OSError, NotImplementedError):
        # Filesystems without symlinks (or a `latest` that is a real directory): write a pointer file.
        try:
            (parent / "latest.txt").write_text(f"{RUNS}/{run_id}\n")
        except OSError:
            pass


# ---------------------------------------------------------------- process-wide handle
_active: Optional[Layout] = None


def activate(layout: Optional[Layout]) -> Optional[Layout]:
    global _active
    _active = layout
    return layout


def get() -> Optional[Layout]:
    return _active


def mirror(path) -> Optional[Path]:
    """Module-level hook, so a save deep inside the training loop reaches durable storage
    without every adapter having to know a Layout exists."""
    return _active.mirror(path) if _active is not None else None


def from_config(cfg: dict, command: str = "run", run_id: Optional[str] = None,
                persist_dir=None, resume_dir=None) -> Layout:
    """Build the layout from a config dictionary plus command-line/environment overrides."""
    persist = persist_dir or os.environ.get("MOBEVAL_PERSIST_DIR") or cfg.get("persist_dir")
    lay = Layout(output_dir=Path(cfg["output_dir"]),
                 run_id=run_id or make_run_id(command),
                 persist_dir=Path(persist) if persist else None,
                 run_dirs=bool(cfg.get("run_dirs", True)),
                 checkpoint_dir_override=Path(cfg["checkpoint_dir"]) if cfg.get("checkpoint_dir") else None)
    if resume_dir is not None:
        lay.run_id = Path(resume_dir).name
    return lay
