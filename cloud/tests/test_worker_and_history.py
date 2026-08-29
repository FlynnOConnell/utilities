"""The worker's entry-point resolution, and the durable run record."""

from __future__ import annotations

import json

import pytest

from imgui_cloud import history, worker_main


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MBO_DIR", str(tmp_path / "mbo"))


def test_resolve_entry_imports_module_and_function():
    function = worker_main.resolve_entry("json:dumps")
    assert function is json.dumps


@pytest.mark.parametrize("bad", ["json", "json:", ":dumps", ""])
def test_resolve_entry_rejects_malformed_specs(bad):
    with pytest.raises(ValueError, match="module:function"):
        worker_main.resolve_entry(bad)


def test_worker_runs_a_job_file_and_reports_success(tmp_path, monkeypatch, capsys):
    module = tmp_path / "fake_pipeline.py"
    module.write_text(
        "CALLED = {}\n"
        "def run_volume(input_data, save_path, settings=None):\n"
        "    CALLED['args'] = (input_data, save_path, settings)\n"
        "    open(save_path, 'w').write('done')\n"
        "    return save_path\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    filepath_job = tmp_path / "job.json"
    filepath_out = tmp_path / "out.txt"
    filepath_job.write_text(
        json.dumps(
            {
                "entry": "fake_pipeline:run_volume",
                "kwargs": {
                    "input_data": "/mnt/data/input",
                    "save_path": str(filepath_out),
                    "settings": {"a": 1},
                },
            }
        )
    )
    assert worker_main.main([str(filepath_job)]) == 0
    assert filepath_out.read_text() == "done"
    assert "OK in" in capsys.readouterr().out


def test_worker_returns_nonzero_when_the_pipeline_raises(tmp_path, monkeypatch, capsys):
    filepath_job = tmp_path / "job.json"
    filepath_job.write_text(
        json.dumps({"entry": "json:loads", "kwargs": {"s": "not json"}})
    )
    assert worker_main.main([str(filepath_job)]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_records_round_trip_and_sort_newest_first():
    for index, name in enumerate(["a", "b", "c"]):
        history.save(history.RunRecord(run_id=name, name=name, time_created=index))
    records = history.load_all()
    assert [r.run_id for r in records] == ["c", "b", "a"]
    assert history.load("b").name == "b"
    assert history.load("nope") is None


def test_live_runs_exclude_terminal_phases():
    history.save(history.RunRecord(run_id="live", phase=history.PHASE_RUNNING))
    history.save(history.RunRecord(run_id="done", phase=history.PHASE_DONE))
    assert [r.run_id for r in history.load_live()] == ["live"]


def test_cost_estimate_tracks_elapsed_time():
    record = history.RunRecord(
        run_id="r",
        cost_per_hour_estimate=2.0,
        time_started=1000.0,
        time_finished=2800.0,
    )
    assert record.cost_estimate() == pytest.approx(1.0)


def test_delete_is_idempotent():
    history.save(history.RunRecord(run_id="gone"))
    assert history.delete("gone") is True
    assert history.delete("gone") is False


def test_a_record_from_a_newer_version_still_loads():
    history.save(history.RunRecord(run_id="r"))
    path = history.filepath_record("r")
    body = json.loads(open(path).read())
    body["field_from_the_future"] = 1
    open(path, "w").write(json.dumps(body))
    assert history.load("r").run_id == "r"


def test_describe_gpus_never_raises():
    assert isinstance(worker_main.describe_gpus(), str)
