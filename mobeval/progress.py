"""Live progress for long runs, readable from somewhere else while the job is running.

A run appends events to `<progress_dir>/<run_id>.events.jsonl` and keeps a rolled-up snapshot in
`<progress_dir>/<run_id>.status.json`, rewritten atomically so a reader never sees a half-written
file. `mobeval status` formats those files; nothing here imports torch, so the status command works
in a plain shell on a login node.

On a cluster the progress directory MUST be on a filesystem the submitting machine can see (a home
or project folder), not the node-local /scratch the job computes in. Set it with
`MOBEVAL_PROGRESS_DIR`, `--progress-dir`, or `progress_dir:` in the config.

Several runs may share a directory (one job per model, say); each writes its own pair of files and
`mobeval status` shows them together.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

HEARTBEAT_S = 30.0
RECENT_EVENTS = 12
STALE_AFTER_S = 180.0        # no update for this long while "running" -> flagged as stale


class Interrupted(RuntimeError):
    """The scheduler asked the job to stop - SIGTERM, which on PBS means the walltime expired.

    Defined here rather than in the CLI so the runner and the registry can let it through
    instead of filing it as a model or task failure. It is not an error in the work; it means
    the work should be continued with --resume.
    """


def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Reporter:
    """Writes progress for one run. Thread-safe; every method is best-effort and never raises."""

    active = True
    scope: Optional[str] = None        # model currently being worked on, for nested call sites

    def __init__(self, progress_dir, command: str = "run", run_id: Optional[str] = None,
                 heartbeat_s: float = HEARTBEAT_S, meta: Optional[dict] = None):
        self.dir = Path(progress_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        job = os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID") or ""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = run_id or f"{stamp}-{job.split('.')[0] or os.getpid()}-{command}"
        self.events_path = self.dir / f"{self.run_id}.events.jsonl"
        self.status_path = self.dir / f"{self.run_id}.status.json"
        self._lock = threading.RLock()
        self._recent: List[str] = []
        self.state = {
            "run_id": self.run_id, "command": command, "state": "running",
            "host": socket.gethostname(), "job_id": job, "pid": os.getpid(),
            "started": time.time(), "updated": time.time(),
            "phase": "starting", "steps_done": 0, "steps_total": 0,
            "models": {}, "counters": {"records": 0, "errors": 0, "skipped": 0},
            "errors": [], "recent": [], **(meta or {}),
        }
        self.event("run_start", command=command, host=self.state["host"], job_id=job)
        self._stop = threading.Event()
        self._beat = None
        if heartbeat_s:
            self._beat = threading.Thread(target=self._heartbeat, args=(heartbeat_s,), daemon=True)
            self._beat.start()

    # ------------------------------------------------------------------ writing
    def _heartbeat(self, every: float):
        while not self._stop.wait(every):
            with self._lock:
                self._flush()

    def _flush(self):
        self.state["updated"] = time.time()
        self.state["elapsed_s"] = self.state["updated"] - self.state["started"]
        self.state["recent"] = self._recent[-RECENT_EVENTS:]
        try:
            _atomic_write(self.status_path, json.dumps(self.state, default=str, indent=1))
        except OSError:
            pass

    def event(self, event: str, message: Optional[str] = None, **fields):
        with self._lock:
            rec = {"ts": time.time(), "event": event, **fields}
            if message:
                rec["message"] = message
                self._recent.append(f"{datetime.now():%H:%M:%S}  {message}")
            try:
                with open(self.events_path, "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            except OSError:
                pass
            self._flush()

    # ------------------------------------------------------------------ structure
    def set_plan(self, steps_total: int, phase: Optional[str] = None):
        with self._lock:
            self.state["steps_total"] = int(steps_total)
            if phase:
                self.state["phase"] = phase
            self._flush()

    def step_done(self, n: int = 1):
        with self._lock:
            self.state["steps_done"] += n
            self._flush()

    def set_phase(self, phase: str):
        with self._lock:
            self.state["phase"] = phase
            self._flush()

    def model(self, name: str, **fields):
        with self._lock:
            self.state["models"].setdefault(name, {}).update(fields)
            self._flush()

    def count(self, **fields):
        with self._lock:
            for k, v in fields.items():
                self.state["counters"][k] = self.state["counters"].get(k, 0) + v
            self._flush()

    def error(self, message: str, **fields):
        with self._lock:
            self.state["errors"].append({"ts": time.time(), "message": message, **fields})
            self.count(errors=1)
        self.event("error", message=message, **fields)

    @contextmanager
    def scoped(self, name: str):
        """Mark the model that nested code (the training loop, a task) is working on."""
        prev, self.scope = self.scope, name
        try:
            yield self
        finally:
            self.scope = prev

    @contextmanager
    def stage(self, name: str, phase: Optional[str] = None, step: bool = False):
        t0 = time.time()
        self.set_phase(phase or name)
        self.event("stage_start", stage=name, message=phase or name)
        try:
            yield self
        except Exception as e:                                          # noqa: BLE001
            self.event("stage_end", stage=name, seconds=time.time() - t0, failed=True, message=f"{name}: {e!r}")
            raise
        else:
            self.event("stage_end", stage=name, seconds=time.time() - t0)
            if step:
                self.step_done()

    def close(self, state: str = "done", **fields):
        self._stop.set()
        with self._lock:
            self.state["state"] = state
            self.state.update(fields)
            if state == "done":
                self.state["phase"] = "finished"
            self._flush()
        self.event("run_end", state=state, seconds=time.time() - self.state["started"], **fields)


class NullReporter:
    """Used when no progress directory is configured: every call is a no-op."""

    active = False
    run_id = None
    state: Dict = {}

    def event(self, *a, **k):
        pass

    set_plan = step_done = set_phase = model = count = error = event

    @contextmanager
    def scoped(self, *a, **k):
        yield self

    @contextmanager
    def stage(self, *a, **k):
        yield self

    def close(self, *a, **k):
        pass


_current: object = NullReporter()


def get() -> Reporter:
    """The active reporter, or a no-op one. Call sites never need to check."""
    return _current                                                     # type: ignore[return-value]


def start(progress_dir, command: str = "run", **kw) -> Reporter:
    global _current
    _current = Reporter(progress_dir, command, **kw) if progress_dir else NullReporter()
    return _current                                                     # type: ignore[return-value]


def stop(state: str = "done", **fields):
    global _current
    _current.close(state, **fields)                                     # type: ignore[attr-defined]
    _current = NullReporter()


# --------------------------------------------------------------------------- reading
def load_runs(progress_dir, limit: Optional[int] = None) -> List[dict]:
    """Every run's snapshot in a directory, newest first, with staleness filled in."""
    d = Path(progress_dir)
    if not d.is_dir():
        return []
    runs = []
    now = time.time()
    for f in sorted(d.glob("*.status.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            r = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue                                                    # being rewritten right now
        r["age_s"] = now - float(r.get("updated", now))
        if r.get("state") == "running" and r["age_s"] > STALE_AFTER_S:
            r["stale"] = True
        r["_events"] = str(f).replace(".status.json", ".events.jsonl")
        runs.append(r)
        if limit and len(runs) >= limit:
            break
    return runs


SKIP_FIELDS = {"ts", "event", "message"}


def event_line(e: dict) -> str:
    """One readable line per event, falling back to its fields when it carries no message."""
    stamp = datetime.fromtimestamp(e.get("ts", 0)).strftime("%H:%M:%S")
    text = e.get("message")
    if not text:
        fields = {k: v for k, v in e.items() if k not in SKIP_FIELDS and not isinstance(v, (dict, list))}
        text = "  ".join(f"{k}={v}" for k, v in list(fields.items())[:5])
    return f"{stamp}  {e.get('event', ''):<12} {text}"


def tail_events(events_path, n: int = 20) -> List[dict]:
    try:
        lines = Path(events_path).read_text().splitlines()[-n:]
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


# --------------------------------------------------------------------------- formatting
def human_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    s = int(max(seconds, 0))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def bar(done: int, total: int, width: int = 24, ascii_only: bool = False) -> str:
    if total <= 0:
        return ""
    filled = int(round(width * min(done / total, 1.0)))
    full, empty = ("#", ".") if ascii_only else ("█", "░")
    return f"[{full * filled}{empty * (width - filled)}] {done}/{total}"


MARK = {"running": "*", "done": "+", "failed": "!", "stale": "?", "interrupted": "~"}


def format_runs(runs: List[dict], verbose: bool = False, ascii_only: bool = False) -> str:
    if not runs:
        return ("No runs found. The job writes progress only if a progress directory is set "
                "(MOBEVAL_PROGRESS_DIR, --progress-dir, or progress_dir: in the config).")
    out: List[str] = []
    for r in runs:
        state = "stale" if r.get("stale") else r.get("state", "?")
        out.append(f"{MARK.get(state, ' ')} {r['run_id']}  [{state}]"
                   + (f"  job {r['job_id']}" if r.get("job_id") else "")
                   + f" on {r.get('host', '?')}")
        started = datetime.fromtimestamp(r["started"]).strftime("%a %H:%M:%S") if r.get("started") else "?"
        out.append(f"  started {started}   elapsed {human_time(r.get('elapsed_s'))}"
                   f"   last update {human_time(r.get('age_s'))} ago")
        out.append(f"  phase: {r.get('phase', '?')}")
        if r.get("persist_dir"):
            out.append(f"  keeping results in: {r['persist_dir']}")
        if r.get("steps_total"):
            eta = ""
            done, total = r["steps_done"], r["steps_total"]
            if done and r.get("state") == "running":
                eta = f"   eta ~{human_time(r['elapsed_s'] / done * (total - done))}"
            out.append(f"  steps: {bar(done, total, ascii_only=ascii_only)}{eta}")
        for name, m in (r.get("models") or {}).items():
            trained = f"   trained in {human_time(m['train_seconds'])}" if m.get("train_seconds") else ""
            out.append(f"    {name:<22} {m.get('state', '?'):<10} {m.get('detail', '')}{trained}")
        c = r.get("counters") or {}
        if any(c.values()):
            out.append("  " + "   ".join(f"{k}: {v}" for k, v in c.items() if v))
        if r.get("failure"):
            if r.get("state") == "interrupted":
                # Not a failure: the scheduler stopped it. Everything finished is already saved.
                out.append(f"  STOPPED {r['failure']} - finished work was written as it went; "
                           f"re-submit with --resume to continue")
                if r.get("run_dir"):
                    out.append(f"  results so far: {r['run_dir']}")
            else:
                out.append(f"  FAILED {r['failure']}")
        for e in (r.get("errors") or [])[-3:]:
            out.append(f"  ERROR {e.get('message', '')}")
        if r.get("stale"):
            out.append("  ! no update for a while - the job may have been killed "
                       "(check with qstat, and see the job's .o/.e files)")
        if verbose:
            out += ["  recent:"] + [f"    {line}" for line in (r.get("recent") or [])]
        out.append("")
    return "\n".join(out).rstrip()
