"""
A durable record of every run, so a crashed client can still find its box.

Each run writes one JSON file under ``~/.mbo/cloud/runs/<run_id>.json`` as soon
as the instance is requested - before it exists, on purpose. That file is what
``imgui-cloud ls`` reads and what ``imgui-cloud down`` uses to clean up
instances left behind by a client that died mid-run.
"""

from __future__ import annotations


import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PHASE_PENDING = "pending"
PHASE_UPLOADING = "uploading"
PHASE_PROVISIONING = "provisioning"
PHASE_RUNNING = "running"
PHASE_DOWNLOADING = "downloading"
PHASE_TEARDOWN = "teardown"
PHASE_DONE = "done"
PHASE_FAILED = "failed"
PHASE_CANCELLED = "cancelled"

# A run whose process is gone keeps its last phase on disk; after this
# long with no update it is reported as stalled rather than as running.
SECONDS_STALE = 900.0

PHASES_TERMINAL = (PHASE_DONE, PHASE_FAILED, PHASE_CANCELLED)


def dir_runs() -> Path:
    """Directory holding run records, honoring ``MBO_DIR`` / ``MBO_USER``."""
    from imgui_cloud.credentials import dir_settings

    path = dir_settings().parent / "cloud" / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class RunRecord:
    """Everything needed to follow, resume reporting on, or tear down a run."""

    run_id: str
    name: str = ""
    profile_name: str = "default"
    project_id: str = ""
    zone: str = ""
    bucket: str = ""
    prefix: str = ""
    instance_name: str = ""
    pipeline: str = ""
    machine_type: str = ""
    spot: bool = True
    dir_input: str = ""
    dir_output: str = ""
    user_email: str = ""
    phase: str = PHASE_PENDING
    message: str = ""
    error: str = ""
    time_created: float = field(default_factory=time.time)
    time_updated: float = 0.0
    time_started: float = 0.0
    time_finished: float = 0.0
    cost_per_hour_estimate: float = 0.0
    bytes_uploaded: int = 0
    bytes_downloaded: int = 0

    @property
    def stalled(self) -> bool:
        """
        Whether a live-looking run has stopped reporting.

        A closed window or a killed process leaves its last phase written to
        disk forever, so a run that still says "provisioning" hours later is
        not provisioning - nobody is watching it any more.
        """
        if self.phase in PHASES_TERMINAL:
            return False
        return time.time() - (self.time_updated or self.time_created) > SECONDS_STALE

    @property
    def uri_run(self) -> str:
        """``gs://`` staging directory for this run."""
        return f"gs://{self.bucket}/{self.prefix}/{self.run_id}"

    @property
    def is_terminal(self) -> bool:
        """Whether the run reached a state that will not change on its own."""
        return self.phase in PHASES_TERMINAL

    @property
    def duration_s(self) -> float:
        """Seconds from instance request to finish (or to now, if running)."""
        if not self.time_started:
            return 0.0
        end = self.time_finished or time.time()
        return end - self.time_started

    def cost_estimate(self) -> float:
        """Indicative USD spent so far, from the machine rate and elapsed time."""
        return self.cost_per_hour_estimate * self.duration_s / 3600.0


def filepath_record(run_id: str) -> str:
    """Path of one run's record."""
    return str(dir_runs() / f"{run_id}.json")


def save(record: RunRecord) -> None:
    """Write (or overwrite) a run record, stamping when it was last alive."""
    record.time_updated = time.time()
    Path(filepath_record(record.run_id)).write_text(
        json.dumps(asdict(record), indent=2)
    )


def load(run_id: str) -> RunRecord | None:
    """Load one run record, or None when there is no such run."""
    path = Path(filepath_record(run_id))
    if not path.exists():
        return None
    body = json.loads(path.read_text())
    names_valid = {f.name for f in fields(RunRecord)}
    return RunRecord(**{k: v for k, v in body.items() if k in names_valid})


def load_all(limit: int = 0, phase: str = "") -> list:
    """
    Run records, newest first.

    Parameters
    ----------
    limit : int
        Maximum number returned; 0 means all.
    phase : str
        Keep only this phase; empty keeps every phase.
    """
    records = []
    for path in dir_runs().glob("*.json"):
        try:
            body = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        names_valid = {f.name for f in fields(RunRecord)}
        records.append(RunRecord(**{k: v for k, v in body.items() if k in names_valid}))
    records.sort(key=lambda r: r.time_created, reverse=True)
    if phase:
        records = [r for r in records if r.phase == phase]
    return records[:limit] if limit else records


def load_live() -> list:
    """Runs that have not reached a terminal phase - candidates for teardown."""
    return [r for r in load_all() if not r.is_terminal]


def delete(run_id: str) -> bool:
    """Remove a run record. Returns False when there was nothing to remove."""
    path = Path(filepath_record(run_id))
    if not path.exists():
        return False
    path.unlink()
    return True
