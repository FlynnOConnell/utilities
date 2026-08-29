"""
The orchestrator: one object that owns a run from upload to teardown.

:class:`CloudRun` runs the whole lifecycle on a background thread and publishes
a snapshot of where it is, so a GUI can poll it once a frame and a CLI can print
it once a second, without either knowing anything about Compute Engine.

The phase sequence is::

    pending -> uploading -> provisioning -> running -> downloading -> teardown
                                                                   -> done
                                                                   -> failed
                                                                   -> cancelled

Teardown is not a phase you can skip. Every exit path - success, failure,
cancellation, an exception nobody expected - goes through :meth:`CloudRun.
_teardown`, which deletes the instance unless the config asked to keep it.
"""

from __future__ import annotations

from collections.abc import Callable

import secrets
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from imgui_cloud import config as config_module
from imgui_cloud import credentials as credentials_module
from imgui_cloud import gcs, history, instance as instance_module, pipelines, startup

POLL_INTERVAL_S = 15.0


def make_run_id(name: str) -> str:
    """A sortable, unique, DNS-safe run id like ``run-20260829-142233-1f3a``."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = instance_module.sanitize_label(name or "run") or "run"
    return f"{slug}-{stamp}-{secrets.token_hex(4)}"


@dataclass
class RunState:
    """A consistent snapshot of a run, safe to read from another thread."""

    phase: str = history.PHASE_PENDING
    message: str = ""
    fraction: float = 0.0
    log_tail: str = ""
    error: str = ""
    instance_status: str = ""
    record: history.RunRecord | None = None
    events: list = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        """Whether the run has stopped for good."""
        return self.phase in history.PHASES_TERMINAL


class CloudRun:
    """
    One cloud run: upload, provision, watch, retrieve, tear down.

    Parameters
    ----------
    config : imgui_cloud.config.CloudConfig
        What to compute and on what hardware.
    profile : imgui_cloud.credentials.CloudProfile, optional
        Connection settings; the active stored profile when omitted.
    on_event : callable, optional
        Called with this :class:`CloudRun` on every phase or message change.
        Runs on the worker thread, so a GUI callback must not touch imgui - poll
        :attr:`state` from the draw loop instead.
    """

    def __init__(
        self,
        config: config_module.CloudConfig,
        profile: credentials_module.CloudProfile | None = None,
        on_event: Callable | None = None,
    ):
        self.config = config
        self.profile = (
            profile if profile is not None else credentials_module.load_profile()
        )
        self.on_event = on_event
        self.run_id = make_run_id(config.io.name)
        self.state = RunState()
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._credentials = None
        self._project = ""
        self._client_storage = None
        self._instance_name = instance_module.instance_name(self.run_id)
        self._dir_output_local = ""

        self.record = history.RunRecord(
            run_id=self.run_id,
            name=config.io.name,
            profile_name=self.profile.name,
            project_id=self.profile.project_id,
            zone=self.profile.zone,
            bucket=self.profile.bucket,
            prefix=self.profile.prefix,
            instance_name=self._instance_name,
            pipeline=config.job.pipeline,
            machine_type=config.machine.machine_type,
            spot=config.machine.spot,
            dir_input=config.io.input,
            user_email=self.profile.user_email,
            cost_per_hour_estimate=config.machine.cost_per_hour_estimate(),
        )
        self.state.record = self.record

    @property
    def prefix_run(self) -> str:
        """Object prefix of this run inside the bucket."""
        return f"{self.profile.prefix}/{self.run_id}"

    @property
    def uri_run(self) -> str:
        """``gs://`` staging directory for this run."""
        return f"gs://{self.profile.bucket}/{self.prefix_run}"

    def _publish(
        self,
        phase: str | None = None,
        message: str = "",
        fraction: float | None = None,
        error: str = "",
    ) -> None:
        """Update the snapshot, persist the record, and notify the listener."""
        with self._lock:
            if phase is not None:
                self.state.phase = phase
                self.record.phase = phase
            if message:
                self.state.message = message
                self.record.message = message
                self.state.events.append((time.time(), message))
            if fraction is not None:
                self.state.fraction = fraction
            if error:
                self.state.error = error
                self.record.error = error
        history.save(self.record)
        if self.on_event is not None:
            self.on_event(self)

    def start(self) -> None:
        """Begin the run on a background thread. Returns immediately."""
        if self._thread is not None:
            raise RuntimeError(f"run {self.run_id} was already started")
        self._thread = threading.Thread(
            target=self._execute, name=f"cloud-run-{self.run_id}", daemon=True
        )
        self._thread.start()

    def cancel(self) -> None:
        """Ask the run to stop; the instance is deleted on the way out."""
        self._cancel.set()
        self._publish(message="cancellation requested")

    def is_alive(self) -> bool:
        """Whether the run's worker thread is still going."""
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout: float | None = None) -> RunState:
        """Block until the run finishes (or ``timeout`` elapses) and return the state."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        return self.state

    def _execute(self) -> None:
        """Whole lifecycle, with teardown guaranteed on every exit path."""
        try:
            self._prepare()
            self._upload()
            if self._cancel.is_set():
                raise RuntimeError("cancelled before provisioning")
            self._provision()
            self._watch()
            if self.state.phase != history.PHASE_FAILED:
                self._download()
        except Exception as e:
            self._publish(
                phase=history.PHASE_FAILED,
                message=str(e),
                error=traceback.format_exc(),
            )
        finally:
            self._teardown()

    def _prepare(self) -> None:
        """Validate, authenticate, and open the storage client."""
        problems = self.config.validate()
        if problems:
            raise ValueError("; ".join(problems))
        if not self.profile.bucket:
            raise ValueError("profile has no bucket; sign in first")

        self._publish(phase=history.PHASE_UPLOADING, message="authenticating")
        self._credentials, self._project = credentials_module.credentials_for(
            self.profile
        )
        self._project = self.profile.project_id or self._project
        if not self._project:
            raise ValueError("no project id in the profile or the credentials")
        self.record.project_id = self._project
        self._client_storage = gcs.client_for(
            self.profile, credentials=self._credentials, project=self._project
        )
        self._dir_output_local = str(config_module.resolve_output_dir(self.config))
        self.record.dir_output = self._dir_output_local

    def _upload(self) -> None:
        """Stage the inputs, the worker script, and the resolved job."""
        spec = pipelines.get(self.config.job.pipeline)
        job = pipelines.build_job(
            spec,
            dir_input_remote=f"{startup.DIR_DATA}/input",
            dir_output_remote=f"{startup.DIR_DATA}/output",
            params=self.config.job.params,
        )

        import json

        gcs.upload_text(
            self._client_storage,
            self.profile.bucket,
            f"{self.prefix_run}/job.json",
            json.dumps(job, indent=2),
        )
        gcs.upload_text(
            self._client_storage,
            self.profile.bucket,
            f"{self.prefix_run}/worker_main.py",
            Path(__file__).with_name("worker_main.py").read_text(),
        )

        self._publish(message=f"uploading {self.config.io.input}", fraction=0.0)
        progress = gcs.upload_tree(
            self._client_storage,
            self.profile.bucket,
            f"{self.prefix_run}/input",
            self.config.io.input,
            patterns=self.config.job.upload_patterns,
            on_progress=self._on_transfer,
        )
        self.record.bytes_uploaded = progress.bytes_done
        self._publish(
            message=f"uploaded {progress.files_done} files "
            f"({progress.bytes_done / 1e9:.2f} GB)",
            fraction=1.0,
        )

    def _on_transfer(self, progress: gcs.TransferProgress) -> None:
        """Progress callback shared by the upload and download phases."""
        with self._lock:
            self.state.fraction = progress.fraction
            self.state.message = (
                f"{progress.files_done}/{progress.files_total}  {progress.current}"
            )
        if self.on_event is not None:
            self.on_event(self)

    def _provision(self) -> None:
        """Create the worker instance with its startup script."""
        self._publish(
            phase=history.PHASE_PROVISIONING,
            message=f"creating {self.config.machine.machine_type} in {self.profile.zone}",
            fraction=0.0,
        )
        spec = pipelines.get(self.config.job.pipeline)
        script = startup.build_startup_script(
            uri_run=self.uri_run,
            run_id=self.run_id,
            requirements=pipelines.requirements(spec, self.config.job.pip),
            env=self.config.job.env,
            python=self.config.job.python,
            torch_backend=self.config.job.torch_backend,
            self_delete=not self.config.machine.keep_instance,
        )
        image = instance_module.resolve_image(
            self._credentials,
            self.config.machine.image_project,
            self.config.machine.image_family,
        )
        resource = instance_module.build_instance(
            machine=self.config.machine,
            zone=self.profile.zone,
            name=self._instance_name,
            image_self_link=image,
            startup_script=script,
            service_account_email=self.profile.service_account_email,
            labels={
                "run-id": self.run_id,
                "user": self.profile.user_email or "unknown",
            },
        )
        self.record.time_started = time.time()
        instance_module.create_worker(
            self._credentials, self._project, self.profile.zone, resource
        )
        self._publish(message=f"instance {self._instance_name} created")

    def _watch(self) -> None:
        """Poll the run's GCS state marker and log until the worker finishes."""
        self._publish(
            phase=history.PHASE_RUNNING, message="worker booting", fraction=0.0
        )
        name_state = f"{self.prefix_run}/status/state.txt"
        name_log = f"{self.prefix_run}/logs/worker.log"
        seen_running = False

        while not self._cancel.is_set():
            state_remote = gcs.read_text(
                self._client_storage, self.profile.bucket, name_state
            )
            log_text = gcs.read_text(
                self._client_storage, self.profile.bucket, name_log
            )
            if log_text:
                with self._lock:
                    self.state.log_tail = "\n".join(log_text.splitlines()[-200:])

            status = instance_module.worker_status(
                self._credentials, self._project, self.profile.zone, self._instance_name
            )
            with self._lock:
                self.state.instance_status = status

            if state_remote == gcs.STATE_DONE:
                self._publish(message="worker finished")
                return
            if state_remote == gcs.STATE_FAILED:
                self._publish(
                    phase=history.PHASE_FAILED,
                    message="worker reported failure; see the log",
                    error="worker reported failure; see the log",
                )
                return
            if state_remote == gcs.STATE_RUNNING and not seen_running:
                seen_running = True
                self._publish(message="pipeline running on the A100")
            if (
                status in ("DELETED", "TERMINATED", "STOPPED")
                and state_remote != gcs.STATE_DONE
            ):
                reason = (
                    "spot instance was preempted"
                    if self.config.machine.spot
                    else "instance stopped before the worker reported DONE"
                )
                self._publish(phase=history.PHASE_FAILED, message=reason, error=reason)
                return

            self._cancel.wait(POLL_INTERVAL_S)

        self._publish(phase=history.PHASE_CANCELLED, message="cancelled")

    def _download(self) -> None:
        """Pull the worker's outputs into the local output directory."""
        if self.state.phase in (history.PHASE_FAILED, history.PHASE_CANCELLED):
            return
        self._publish(
            phase=history.PHASE_DOWNLOADING,
            message=f"downloading results to {self._dir_output_local}",
            fraction=0.0,
        )
        progress = gcs.download_tree(
            self._client_storage,
            self.profile.bucket,
            f"{self.prefix_run}/output",
            self._dir_output_local,
            patterns=self.config.job.download_patterns,
            on_progress=self._on_transfer,
        )
        self.record.bytes_downloaded = progress.bytes_done
        self._publish(
            phase=history.PHASE_DONE,
            message=f"{progress.files_done} files "
            f"({progress.bytes_done / 1e9:.2f} GB) in {self._dir_output_local}",
            fraction=1.0,
        )

    def _teardown(self) -> None:
        """Delete the instance unless the config asked to keep it."""
        self.record.time_finished = time.time()
        if self.config.machine.keep_instance:
            self._publish(
                message=f"instance {self._instance_name} left running "
                f"(machine.keep_instance = true)"
            )
            history.save(self.record)
            return
        if self._credentials is None or not self._project:
            history.save(self.record)
            return

        phase_final = self.state.phase
        self._publish(phase=history.PHASE_TEARDOWN, message="deleting the instance")
        try:
            deleted = instance_module.delete_worker(
                self._credentials, self._project, self.profile.zone, self._instance_name
            )
        except Exception as e:
            self._publish(message=f"could not delete {self._instance_name}: {e}")
            deleted = False
        message = (
            f"instance deleted; ~${self.record.cost_estimate():.2f} of compute"
            if deleted
            else "instance was already gone"
        )
        phase_final = (
            phase_final
            if phase_final in history.PHASES_TERMINAL
            else history.PHASE_DONE
        )
        self._publish(phase=phase_final, message=message)


def teardown_orphans(profile: credentials_module.CloudProfile | None = None) -> list:
    """
    Delete every worker this package left behind in the profile's zone.

    The safety valve for a client that died mid-run: instances are matched by
    the ``managed-by=imgui-cloud`` label, so nothing else is ever touched.

    Returns
    -------
    list of str
        Names of the instances deleted.
    """
    profile = profile if profile is not None else credentials_module.load_profile()
    creds, project = credentials_module.credentials_for(profile)
    project = profile.project_id or project
    deleted = []
    for worker in instance_module.list_workers(creds, project, profile.zone):
        instance_module.delete_worker(creds, project, profile.zone, worker.name)
        deleted.append(worker.name)
    for record in history.load_live():
        if record.instance_name in deleted or record.zone != profile.zone:
            record.phase = history.PHASE_CANCELLED
            record.message = "torn down by imgui-cloud down"
            history.save(record)
    return deleted


def execute(
    config: config_module.CloudConfig,
    profile: credentials_module.CloudProfile | None = None,
    on_event: Callable | None = None,
) -> CloudRun:
    """Start a run and block until it finishes. Returns the finished run."""
    run = CloudRun(config, profile=profile, on_event=on_event)
    run.start()
    run.wait()
    return run
