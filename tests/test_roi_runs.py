"""gui.roi_runs: run orchestration and derived-set state, pure-unit.

RoiRunManager is exercised against a stub process manager (real LocalJob
handles, fake spawn/get_running/kill); the overlay, pick-map, scan,
worker-args and sidecar helpers need only numpy and a tmp dir. Nothing
here imports masknmf or opens a figure.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mbo_utilities import roi_workflow as rw
from mbo_utilities.gui import roi_runs as rr
from mbo_utilities.gui.widgets.process_manager import LocalJob


# ---- fixtures ---------------------------------------------------------------


class _StubPM:
    """start_job/spawn/get_running/kill lookalike; runs nothing."""

    def __init__(self):
        self.jobs: list[LocalJob] = []
        self.spawns: list[dict] = []
        self.killed: list[int] = []
        self.running: list[SimpleNamespace] = []
        self.spawn_fails = False
        self._next_pid = 1000

    def start_job(self, task_type, description):
        job = LocalJob(
            job_id=len(self.jobs) + 1,
            task_type=task_type,
            description=description,
            start_time=time.time(),
        )
        self.jobs.append(job)
        return job

    def spawn(self, task_type, args, description, output_path=None):
        self.spawns.append(
            {"task_type": task_type, "args": args, "description": description,
             "output_path": output_path}
        )
        if self.spawn_fails:
            return None
        self._next_pid += 1
        self.running.append(
            SimpleNamespace(pid=self._next_pid, status="running", status_message="")
        )
        return self._next_pid

    def get_running(self):
        return list(self.running)

    def kill(self, pid):
        self.killed.append(pid)
        self.running = [p for p in self.running if p.pid != pid]
        return True


@pytest.fixture
def pm():
    return _StubPM()


def _drain(mgr, pm, n=1, timeout=5.0):
    end = time.time() + timeout
    out = []
    while time.time() < end and len(out) < n:
        out += mgr.poll(pm)
        time.sleep(0.005)
    return out


def _until(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def _row(ypix, xpix, lam):
    ypix, xpix = np.asarray(ypix, np.int32), np.asarray(xpix, np.int32)
    lam = np.asarray(lam, np.float32)
    return {"ypix": ypix, "xpix": xpix, "lam": lam,
            "med": (float(ypix.mean()), float(xpix.mean())), "npix": int(ypix.size)}


def _result(rows, shape=(6, 6), z=0, kind="discover"):
    return rw.RunResult(
        path=Path("run"), kind=kind, z=z, shape=shape,
        stat=np.array(rows, dtype=object), F=None, Fneu=None, norm=None,
        iscell=None, uids=None, store_indices=None,
    )


# ---- manager: in-process runs -----------------------------------------------


def test_submit_success_and_single_poll(pm):
    mgr = rr.RoiRunManager(pm)
    run = rr.RoiRun(kind="extract", tag="a", description="extract rois_a")
    seen = []
    mgr.submit(run, lambda job: seen.append(job) or ["out"])
    assert mgr.busy and run in mgr.active
    done = _drain(mgr, pm)
    assert done == [(run, ["out"])]
    assert run.finished and run.error is None
    assert seen == [run.job] and run.job.status == "completed"
    assert pm.jobs[0].task_type == "extract"
    assert not mgr.busy and mgr.poll(pm) == []


def test_submit_failure_lands_on_run_and_job(pm):
    mgr = rr.RoiRunManager(pm)
    run = rr.RoiRun(kind="demix", tag="b", description="demix rois_b")

    def boom(job):
        raise ValueError("boom")

    mgr.submit(run, boom)
    (got, payload), = _drain(mgr, pm)
    assert got is run and payload is None
    # the message names the frame that raised, so a failure deep in a
    # pipeline is diagnosable without opening the log
    assert run.error.startswith("ValueError: boom (test_roi_runs.py:")
    assert run.job.status == "error" and run.job.error_details == run.error


def test_heavy_runs_serialize_on_the_gpu_lock(pm):
    mgr = rr.RoiRunManager(pm)
    gate = threading.Event()
    order = []

    def slow(job):
        order.append("a")
        assert gate.wait(5)
        order.append("a-done")

    ra = mgr.submit(rr.RoiRun(kind="demix", tag="a", description="demix a"), slow, heavy=True)
    assert _until(lambda: order == ["a"])
    rb = mgr.submit(rr.RoiRun(kind="discover", tag="b", description="find b"),
                    lambda job: order.append("b"), heavy=True)
    assert _until(lambda: rb.job.status_message == "waiting for gpu")
    time.sleep(0.05)
    assert order == ["a"]  # b is queued behind a, not running
    gate.set()
    assert len(_drain(mgr, pm, n=2)) == 2
    assert order == ["a", "a-done", "b"]
    assert ra.finished and rb.finished and not mgr.busy


# ---- manager: spawned runs --------------------------------------------------


def test_spawn_records_pid_and_out_root(pm, tmp_path):
    mgr = rr.RoiRunManager(pm)
    run = rr.RoiRun(kind="suite2p", tag="plane01", description="suite2p plane01")
    mgr.spawn(run, "suite2p", {"output_dir": str(tmp_path), "planes": [1]})
    assert run.pid == pm.running[0].pid and run.out_root == tmp_path
    assert pm.spawns[0]["task_type"] == "suite2p"
    assert pm.spawns[0]["output_path"] == str(tmp_path)
    assert mgr.poll(pm) == [] and mgr.busy
    pm.running[0].status = "completed"
    assert mgr.poll(pm) == [(run, None)]
    assert run.finished and run.error is None
    assert mgr.poll(pm) == []


def test_spawn_error_and_disappearance(pm, tmp_path):
    mgr = rr.RoiRunManager(pm)
    bad = rr.RoiRun(kind="masknmf", tag="p2", description="masknmf plane02")
    mgr.spawn(bad, "masknmf", {"output_dir": str(tmp_path)})
    pm.running[0].status = "error"
    pm.running[0].status_message = "cuda oom"
    assert mgr.poll(pm) == [(bad, None)] and bad.error == "cuda oom"

    gone = rr.RoiRun(kind="suite2p", tag="p3", description="suite2p plane03")
    mgr.spawn(gone, "suite2p", {"output_dir": str(tmp_path)})
    pm.running.clear()
    assert mgr.poll(pm) == [(gone, None)] and gone.error == "process stopped"


def test_spawn_start_failure_surfaces_through_poll(pm, tmp_path):
    mgr = rr.RoiRunManager(pm)
    pm.spawn_fails = True
    run = rr.RoiRun(kind="suite2p", tag="p1", description="suite2p plane01")
    mgr.spawn(run, "suite2p", {"output_dir": str(tmp_path)})
    assert run.pid is None
    assert mgr.poll(pm) == [(run, None)]
    assert run.finished and run.error == "failed to start worker"


def _fake_run_dir(root, name, n_rois=0):
    d = root / name
    d.mkdir(parents=True)
    np.save(d / "stat.npy", np.empty(n_rois, object))
    np.save(d / "ops.npy", {"Ly": 4, "Lx": 4}, allow_pickle=True)
    return d


def test_finished_dirs_prefix_matches_suffixed_plane_dirs(tmp_path):
    a = _fake_run_dir(tmp_path, "zplane01_tp00001-00400")
    b = _fake_run_dir(tmp_path, "zplane02")
    partial = tmp_path / "zplane03"
    partial.mkdir()
    np.save(partial / "ops.npy", {}, allow_pickle=True)  # no stat: still writing
    assert rr.finished_dirs(tmp_path, [1]) == [a]
    assert rr.finished_dirs(tmp_path, [3]) == []
    assert rr.finished_dirs(tmp_path) == [a, b]
    assert rr.finished_dirs(a) == [a]


def test_scan_lists_the_opened_dir_and_vanilla_suite2p(tmp_path):
    np.save(tmp_path / "stat.npy", np.empty(0, object))
    np.save(tmp_path / "ops.npy", {"Ly": 4, "Lx": 4}, allow_pickle=True)
    _fake_run_dir(tmp_path, "suite2p/plane0")
    _fake_run_dir(tmp_path, "plane2")
    paths = {str(r["path"]) for r in rr.scan_run_dirs(tmp_path / "data.bin")}
    assert str(tmp_path) in paths
    assert str(tmp_path / "suite2p" / "plane0") in paths
    assert str(tmp_path / "plane2") in paths


def test_spawn_disappearance_with_outputs_is_success(pm, tmp_path):
    # the console prunes completed entries after a few minutes; outputs on
    # disk mean the run finished, not that it crashed
    _fake_run_dir(tmp_path, "zplane01_tp00001-00400")
    mgr = rr.RoiRunManager(pm)
    run = rr.RoiRun(kind="suite2p", tag="p1", description="suite2p plane01", planes=[1])
    mgr.spawn(run, "suite2p", {"output_dir": str(tmp_path)})
    pm.running.clear()
    assert mgr.poll(pm) == [(run, None)]
    assert run.finished and run.error is None


def test_stop_kills_spawned_only(pm, tmp_path):
    mgr = rr.RoiRunManager(pm)
    spawned = rr.RoiRun(kind="suite2p", tag="p1", description="suite2p plane01")
    mgr.spawn(spawned, "suite2p", {"output_dir": str(tmp_path)})
    mgr.stop(spawned, pm)
    assert pm.killed == [spawned.pid]
    assert mgr.poll(pm) == [(spawned, None)] and spawned.error == "process stopped"

    gate = threading.Event()
    local = rr.RoiRun(kind="extract", tag="a", description="extract rois_a")
    mgr.submit(local, lambda job: gate.wait(5))
    mgr.stop(local, pm)  # in-process: no cancel
    assert pm.killed == [spawned.pid]
    gate.set()
    _drain(mgr, pm)


# ---- derived sets -----------------------------------------------------------


def test_pick_map_strongest_lam_wins():
    res = _result([
        _row([0, 1], [0, 1], [5.0, 5.0]),
        _row([1, 2], [1, 2], [2.0, 2.0]),
    ])
    pick = rr.build_pick_map(res.stat, res.shape)
    assert pick[0, 0] == 0 and pick[2, 2] == 1
    assert pick[1, 1] == 0  # contested pixel goes to the stronger component
    assert pick[3, 3] == -1
    s = rr.DerivedSet(res, "a", rr.set_color(0))
    np.testing.assert_array_equal(s.pick_map, pick)


def test_derived_rgba_lam_weights():
    s = rr.DerivedSet(_result([_row([1, 2], [1, 2], [1.0, 2.0])]), "a", (1.0, 0.0, 0.0))
    img = rr.derived_rgba((6, 6), [s], 0.5)
    assert img.shape == (6, 6, 4) and img.dtype == np.uint8
    c = np.rint(np.asarray(rr.component_color(s, 0)) * 255).astype(int).tolist()
    assert img[1, 1].tolist() == [*c, 64]  # lam/peak = 0.5 -> 0.25 alpha
    assert img[2, 2].tolist() == [*c, 128]
    assert img[0, 0, 3] == 0


def test_derived_rgba_overlap_winner_takes_color():
    a = rr.DerivedSet(_result([_row([1, 2], [1, 2], [0.5, 1.0])]), "a", (1.0, 0.0, 0.0))
    b = rr.DerivedSet(_result([_row([1], [1], [1.0])]), "b", (0.0, 0.0, 1.0))
    img = rr.derived_rgba((6, 6), [a, b], 0.6)
    ca = np.rint(np.asarray(rr.component_color(a, 0)) * 255).astype(int).tolist()
    cb = np.rint(np.asarray(rr.component_color(b, 0)) * 255).astype(int).tolist()
    assert img[2, 2].tolist() == [*ca, 153]  # a's peak pixel, 0.6 alpha
    assert img[1, 1].tolist() == [*cb, 153]  # b's 0.6 beats a's 0.3


def test_derived_rgba_selection_fill_and_rim():
    yy, xx = np.mgrid[1:4, 1:4]
    s = rr.DerivedSet(
        _result([_row(yy.ravel(), xx.ravel(), np.ones(9))]), "a", (1.0, 0.0, 0.0)
    )
    img = rr.derived_rgba((6, 6), [s], 0.3, selected=(s, 0))
    fill = round(rr.SELECTED_ALPHA * 255)
    c = np.rint(np.asarray(rr.component_color(s, 0)) * 255).astype(int).tolist()
    assert img[2, 2].tolist() == [*c, fill]  # interior keeps its own color
    assert img[1, 1].tolist() == [255, 255, 255, fill]  # border is the white rim
    assert img[1, 2].tolist() == [255, 255, 255, fill]


def test_derived_rgba_skips_discarded_and_invisible():
    s = rr.DerivedSet(_result([
        _row([1], [1], [1.0]),
        _row([3], [3], [1.0]),
    ]), "a", (1.0, 0.0, 0.0), discarded={0})
    img = rr.derived_rgba((6, 6), [s], 0.5)
    assert img[1, 1, 3] == 0 and img[3, 3, 3] == 128
    s.visible = False
    assert not rr.derived_rgba((6, 6), [s], 0.5).any()
    # a discarded selection is not highlighted either
    assert not rr.derived_rgba((6, 6), [s], 0.5, selected=(s, 0)).any()


def test_component_color_is_class_color_when_labeled():
    from mbo_utilities.annotation import class_color

    s = rr.DerivedSet(_result([_row([1], [1], [1.0])]), "a", (1.0, 0.0, 0.0))
    hue = rr.component_color(s, 0)
    s.classes[0] = 1
    assert rr.component_color(s, 0) == class_color(1) != hue


def test_display_trace_matches_the_lsp_recipe():
    F = np.array([10.0, 10.0, 30.0, 10.0], np.float32)
    Fneu = np.array([2.0, 2.0, 2.0, 2.0], np.float32)
    corr = F - 0.7 * Fneu
    f0 = float(np.percentile(corr, 20))
    expected = (corr - f0) / f0 * 100.0
    got = rr.display_trace({"F": F, "Fneu": Fneu})
    np.testing.assert_allclose(got, expected, rtol=1e-5)
    # correction off: raw F baseline
    f0 = float(np.percentile(F, 20))
    np.testing.assert_allclose(
        rr.display_trace({"F": F, "Fneu": Fneu}, correct_neuropil=False),
        (F - f0) / f0 * 100.0, rtol=1e-5,
    )
    # the run's own norm_traces win outright
    norm = np.array([0.0, 5.0, 50.0, 0.0], np.float32)
    np.testing.assert_array_equal(rr.display_trace({"F": F, "norm": norm}), norm)
    # neuropil rides the same percent scale; absent -> None
    assert rr.display_fneu({"F": F}) is None
    yneu = rr.display_fneu({"F": F, "Fneu": Fneu})
    np.testing.assert_allclose(yneu, np.zeros(4), atol=1e-3)


def test_trace_set_prune_keeps_live_uids():
    ts = rr.TraceSet("quick", "extract", {3: {"F": np.ones(4)}, 7: {"F": np.zeros(4)}})
    ts.prune([7, 9])
    assert list(ts.data) == [7]
    ts.prune([])
    assert ts.data == {}


# ---- disk helpers -----------------------------------------------------------


def _run_dir(path, kind=None, n_stat=1, n_rois=None, pipeline=None):
    path.mkdir(parents=True, exist_ok=True)
    ops = {"Ly": 8, "Lx": 9}
    if kind is not None:
        ops["roi_workflow"] = {"process": kind}
    if n_rois is not None:
        ops["n_rois"] = n_rois
    if pipeline is not None:
        ops["pipeline"] = pipeline
    np.save(path / "ops.npy", ops, allow_pickle=True)
    if n_stat is not None:
        np.save(path / "stat.npy", np.array([_row([0], [0], [1.0])] * n_stat, object))
    return path


def test_scan_run_dirs(tmp_path):
    _run_dir(tmp_path / "rois_a", kind="extract", n_rois=2, n_stat=2)
    _run_dir(tmp_path / "rois_m" / "z01", kind="demix", pipeline="masknmf")
    _run_dir(tmp_path / "rois_m" / "z02", kind="demix", pipeline="masknmf")
    _run_dir(tmp_path / "rois_partial", kind="extract", n_stat=None)  # still writing
    _run_dir(tmp_path / "zplane01", n_stat=3)  # full suite2p plane
    _run_dir(tmp_path / "zplane01" / "rois_d", kind="discover")
    _run_dir(tmp_path / "zplane02", n_stat=None)  # registered only, no detection

    rows = rr.scan_run_dirs(tmp_path)
    by_path = {r["path"]: r for r in rows}
    assert set(by_path) == {
        tmp_path / "rois_a", tmp_path / "rois_m" / "z01", tmp_path / "rois_m" / "z02",
        tmp_path / "zplane01", tmp_path / "zplane01" / "rois_d",
    }
    assert by_path[tmp_path / "rois_a"]["kind"] == "extract"
    assert by_path[tmp_path / "rois_a"]["n_rois"] == 2
    assert by_path[tmp_path / "rois_m" / "z01"]["kind"] == "demix"
    assert by_path[tmp_path / "rois_m" / "z01"]["n_rois"] == 1  # from stat.npy
    assert by_path[tmp_path / "zplane01"]["kind"] == "suite2p"
    assert by_path[tmp_path / "zplane01"]["n_rois"] == 3
    assert all(r["mtime"] > 0 for r in rows)
    assert [r["mtime"] for r in rows] == sorted((r["mtime"] for r in rows), reverse=True)
    # a data file scans the dir beside it; nothing on disk is fine
    assert {r["path"] for r in rr.scan_run_dirs(tmp_path / "raw.tif")} == set(by_path)
    assert rr.scan_run_dirs(tmp_path / "missing" / "raw.tif") == []


def test_full_plane_args_minimal_payloads(tmp_path):
    from mbo_utilities.masknmf.params import MasknmfSettings

    fpath = tmp_path / "raw.tif"
    s2p = rr.full_plane_args("suite2p", fpath, 3, None)
    assert set(s2p) == {"input_path", "output_dir", "planes", "reader_kwargs"}
    assert s2p["input_path"] == str(fpath) and s2p["output_dir"] == str(tmp_path)
    assert s2p["planes"] == [3] and s2p["reader_kwargs"] == {}

    mnmf = rr.full_plane_args("masknmf", fpath, 1, None)
    assert set(mnmf) == {"input_path", "output_dir", "planes", "reader_kwargs", "settings"}
    assert mnmf["settings"] == MasknmfSettings().to_dict()

    with pytest.raises(ValueError, match="pipeline"):
        rr.full_plane_args("bogus", fpath, 1, None)
    with pytest.raises(ValueError, match="data path"):
        rr.full_plane_args("suite2p", None, 1, None)


def test_run_registry_round_trip(tmp_path):
    target = rr.registry_path(tmp_path / "raw.tif")
    assert target == tmp_path / rr.REGISTRY_NAME
    assert rr.load_run_registry(target) == []

    entries = [
        {"path": tmp_path / "rois_a", "kind": "extract", "discarded": {3, 1},
         "classes": {2: 0}},
        {"path": tmp_path / "gone" / "rois_b", "kind": "demix"},  # dir never existed
    ]
    rr.save_run_registry(target, entries)
    got = rr.load_run_registry(target)
    assert got == [
        {"path": str(tmp_path / "rois_a"), "kind": "extract",
         "discarded": [1, 3], "classes": {2: 0}},
        {"path": str(tmp_path / "gone" / "rois_b"), "kind": "demix",
         "discarded": [], "classes": {}},
    ]
    assert json.loads(target.read_text())["runs"][0]["discarded"] == [1, 3]

    target.write_text("not json {")
    assert rr.load_run_registry(target) == []
    nested = tmp_path / "new" / "deep" / rr.REGISTRY_NAME
    rr.save_run_registry(nested, [])
    assert rr.load_run_registry(nested) == []
