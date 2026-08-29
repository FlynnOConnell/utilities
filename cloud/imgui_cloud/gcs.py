"""
The staging layer: a GCS bucket is the only thing the laptop and the box share.

Inputs go up to ``gs://<bucket>/<prefix>/<run_id>/input``, the worker pulls them
onto its scratch disk, writes results to ``.../output``, and the laptop pulls
those down. Nothing is transferred over SSH, so a run survives the laptop
sleeping, and a preempted spot VM loses no uploaded bytes.

Layout under one run::

    input/            what was uploaded from the local input dir
    output/           what the worker produced
    logs/worker.log   streamed while the job runs
    status/state.txt  one word: STARTED | RUNNING | DONE | FAILED
    job.json          the resolved pipeline call
    worker_main.py    the script the VM executes
"""

from __future__ import annotations

from collections.abc import Callable

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path

STATE_STARTED = "STARTED"
STATE_RUNNING = "RUNNING"
STATE_DONE = "DONE"
STATE_FAILED = "FAILED"


@dataclass
class TransferProgress:
    """Snapshot handed to a transfer's progress callback."""

    files_done: int = 0
    files_total: int = 0
    bytes_done: int = 0
    bytes_total: int = 0
    current: str = ""

    @property
    def fraction(self) -> float:
        """Completed fraction by bytes, 0.0 when nothing is known yet."""
        if self.bytes_total <= 0:
            return 0.0
        return min(1.0, self.bytes_done / self.bytes_total)


ProgressCallback = Callable[[TransferProgress], None]


def client_for(profile, credentials=None, project: str = ""):
    """A :class:`google.cloud.storage.Client` for ``profile``."""
    from google.cloud import storage

    from imgui_cloud.credentials import credentials_for

    if credentials is None:
        credentials, project_resolved = credentials_for(profile)
        project = project or project_resolved
    return storage.Client(
        project=project or profile.project_id, credentials=credentials
    )


def matching_files(dir_local: str, patterns: list | None = None) -> list:
    """
    Files under ``dir_local`` matching any glob in ``patterns`` (default all).

    Patterns are matched against the path relative to ``dir_local``, so
    ``["*.tif", "metadata/*"]`` works the way it reads.
    """
    root = Path(dir_local)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"no such directory: {root}")
    patterns = patterns or ["*"]
    out = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(
            fnmatch.fnmatch(relative, p) or fnmatch.fnmatch(path.name, p)
            for p in patterns
        ):
            out.append(path)
    return out


def upload_tree(
    client,
    bucket_name: str,
    prefix: str,
    dir_local: str,
    patterns: list | None = None,
    on_progress: ProgressCallback | None = None,
) -> TransferProgress:
    """
    Upload a local directory (or single file) under ``gs://bucket/prefix/``.

    Returns
    -------
    TransferProgress
        Final counts, so a caller can report totals without re-walking.
    """
    bucket = client.bucket(bucket_name)
    root = Path(dir_local)
    filepaths = matching_files(dir_local, patterns)
    progress = TransferProgress(
        files_total=len(filepaths),
        bytes_total=sum(p.stat().st_size for p in filepaths),
    )
    for path in filepaths:
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        progress.current = relative
        if on_progress is not None:
            on_progress(progress)
        bucket.blob(f"{prefix.rstrip('/')}/{relative}").upload_from_filename(str(path))
        progress.files_done += 1
        progress.bytes_done += path.stat().st_size
        if on_progress is not None:
            on_progress(progress)
    return progress


def download_tree(
    client,
    bucket_name: str,
    prefix: str,
    dir_local: str,
    patterns: list | None = None,
    on_progress: ProgressCallback | None = None,
) -> TransferProgress:
    """Mirror ``gs://bucket/prefix/`` into ``dir_local``, creating it if needed."""
    bucket = client.bucket(bucket_name)
    prefix = prefix.rstrip("/") + "/"
    blobs = [
        b for b in client.list_blobs(bucket, prefix=prefix) if not b.name.endswith("/")
    ]
    if patterns:
        blobs = [
            b
            for b in blobs
            if any(
                fnmatch.fnmatch(b.name[len(prefix) :], p)
                or fnmatch.fnmatch(os.path.basename(b.name), p)
                for p in patterns
            )
        ]
    progress = TransferProgress(
        files_total=len(blobs), bytes_total=sum(b.size or 0 for b in blobs)
    )
    root = Path(dir_local)
    root.mkdir(parents=True, exist_ok=True)
    for blob in blobs:
        relative = blob.name[len(prefix) :]
        progress.current = relative
        if on_progress is not None:
            on_progress(progress)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))
        progress.files_done += 1
        progress.bytes_done += blob.size or 0
        if on_progress is not None:
            on_progress(progress)
    return progress


def upload_text(client, bucket_name: str, name: str, text: str) -> None:
    """Write ``text`` to ``gs://bucket/name``."""
    client.bucket(bucket_name).blob(name).upload_from_string(text)


def read_text(client, bucket_name: str, name: str) -> str | None:
    """Contents of ``gs://bucket/name``, or None when it does not exist."""
    blob = client.bucket(bucket_name).get_blob(name)
    if blob is None:
        return None
    return blob.download_as_text()


def exists(client, bucket_name: str, name: str) -> bool:
    """Whether ``gs://bucket/name`` exists."""
    return client.bucket(bucket_name).get_blob(name) is not None


def delete_prefix(client, bucket_name: str, prefix: str) -> int:
    """Delete every object under ``prefix``. Returns how many were removed."""
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket, prefix=prefix.rstrip("/") + "/"))
    for blob in blobs:
        blob.delete()
    return len(blobs)


def total_size(client, bucket_name: str, prefix: str) -> int:
    """Summed size in bytes of everything under ``prefix``."""
    return sum(
        b.size or 0
        for b in client.list_blobs(
            client.bucket(bucket_name), prefix=prefix.rstrip("/") + "/"
        )
    )
