"""The orchestrator's phase machine, and the promise that the box always dies."""

from __future__ import annotations

import pytest

from imgui_cloud import config as config_module
from imgui_cloud import credentials as credentials_module
from imgui_cloud import gcs, history
from imgui_cloud import run as run_module


class FakeCloud:
    """Records what the orchestrator did, and scripts what GCS reports back."""

    def __init__(self, states):
        self.states = list(states)
        self.uploaded = []
        self.downloaded = []
        self.created = []
        self.deleted = []
        self.instance_status = "RUNNING"

    def read_text(self, client, bucket, name):
        if name.endswith("state.txt"):
            return self.states.pop(0) if self.states else None
        if name.endswith("worker.log"):
            return "line one\nline two"
        return None


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A CloudRun whose every cloud call is replaced by an in-memory fake."""
    monkeypatch.setenv("MBO_DIR", str(tmp_path / "mbo"))
    dir_input = tmp_path / "raw"
    dir_input.mkdir()
    (dir_input / "a.tif").write_bytes(b"0123456789")

    config = config_module.default_config(
        dir_input=str(dir_input), dir_output=str(tmp_path / "out"), name="unit"
    )
    profile = credentials_module.CloudProfile(
        project_id="proj", bucket="bkt", zone="us-central1-a"
    )

    def make(states=("STARTED", "RUNNING", "DONE")):
        fake = FakeCloud(states)
        monkeypatch.setattr(
            run_module.credentials_module,
            "credentials_for",
            lambda p: ("creds", "proj"),
        )
        monkeypatch.setattr(run_module.gcs, "client_for", lambda *a, **k: "client")
        monkeypatch.setattr(run_module.gcs, "upload_text", lambda *a, **k: None)
        monkeypatch.setattr(
            run_module.gcs,
            "upload_tree",
            lambda *a, **k: (
                fake.uploaded.append(a)
                or gcs.TransferProgress(
                    files_done=1, files_total=1, bytes_done=10, bytes_total=10
                )
            ),
        )
        monkeypatch.setattr(
            run_module.gcs,
            "download_tree",
            lambda *a, **k: (
                fake.downloaded.append(a)
                or gcs.TransferProgress(
                    files_done=2, files_total=2, bytes_done=20, bytes_total=20
                )
            ),
        )
        monkeypatch.setattr(run_module.gcs, "read_text", fake.read_text)
        monkeypatch.setattr(
            run_module.instance_module, "resolve_image", lambda *a, **k: "image-link"
        )
        monkeypatch.setattr(
            run_module.instance_module, "build_instance", lambda **kwargs: kwargs
        )
        monkeypatch.setattr(
            run_module.instance_module,
            "create_worker",
            lambda creds, project, zone, resource: fake.created.append(resource),
        )
        monkeypatch.setattr(
            run_module.instance_module,
            "worker_status",
            lambda *a, **k: fake.instance_status,
        )
        monkeypatch.setattr(
            run_module.instance_module,
            "delete_worker",
            lambda creds, project, zone, name, **k: (
                bool(fake.deleted.append(name)) or True
            ),
        )
        monkeypatch.setattr(run_module, "POLL_INTERVAL_S", 0.01)
        return fake, run_module.CloudRun(config, profile=profile)

    return make


def test_a_successful_run_walks_every_phase_and_tears_down(wired):
    fake, run = wired()
    phases = []
    run.on_event = lambda r: phases.append(r.state.phase)
    run.start()
    state = run.wait(timeout=30)

    assert state.phase == history.PHASE_DONE
    assert "uploading" in phases and "provisioning" in phases and "running" in phases
    assert fake.uploaded and fake.downloaded
    assert fake.deleted == [run._instance_name]
    assert run.record.bytes_uploaded == 10
    assert run.record.bytes_downloaded == 20


def test_a_failing_worker_still_deletes_the_instance(wired):
    fake, run = wired(states=("STARTED", "FAILED"))
    run.start()
    state = run.wait(timeout=30)
    assert state.phase == history.PHASE_FAILED
    assert fake.deleted == [run._instance_name]
    assert not fake.downloaded


def test_preemption_is_reported_as_such(wired):
    fake, run = wired(states=("STARTED", "RUNNING", None, None))
    fake.instance_status = "TERMINATED"
    run.start()
    state = run.wait(timeout=30)
    assert state.phase == history.PHASE_FAILED
    assert "preempted" in state.error


def test_a_bad_config_fails_before_anything_is_created(wired):
    fake, run = wired()
    run.config.io.input = "/definitely/not/here"
    run.start()
    state = run.wait(timeout=30)
    assert state.phase == history.PHASE_FAILED
    assert not fake.created
    assert not fake.deleted


def test_keep_instance_leaves_the_box_up(wired):
    fake, run = wired()
    run.config.machine.keep_instance = True
    run.start()
    run.wait(timeout=30)
    assert fake.deleted == []


def test_the_record_is_written_before_the_instance_exists(wired):
    fake, run = wired()
    run.start()
    run.wait(timeout=30)
    record = history.load(run.run_id)
    assert record is not None
    assert record.instance_name == run._instance_name
    assert record.phase == history.PHASE_DONE
    assert record.pipeline == "masknmf"


def test_run_ids_are_unique_and_dns_safe():
    ids = {run_module.make_run_id("MK 355") for _ in range(50)}
    assert len(ids) == 50
    assert all(i.startswith("mk-355-") for i in ids)
    assert all(c.isalnum() or c == "-" for i in ids for c in i)


def test_starting_twice_is_refused(wired):
    fake, run = wired()
    run.start()
    with pytest.raises(RuntimeError, match="already started"):
        run.start()
    run.wait(timeout=30)
