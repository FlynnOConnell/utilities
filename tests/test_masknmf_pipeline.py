"""masknmf integration tests: params round-trip, stage gating, suite2p-shaped
output conversion. No masknmf install required — the compute stages are only
exercised when the package is present (none here yet)."""

import json

import numpy as np
import pytest

from mbo_utilities.masknmf.params import (
    STAGE_FORCE,
    STAGE_RUN,
    STAGE_SKIP,
    MasknmfSettings,
    stage_action,
)
from mbo_utilities.masknmf import outputs


# ---- params -----------------------------------------------------------------


def test_settings_roundtrip_json():
    s = MasknmfSettings()
    s.registration.strategy = "pwrigid"
    s.registration.max_shifts = (7, 9)
    s.demixing.do_demixing = STAGE_FORCE
    d = json.loads(json.dumps(s.to_dict()))
    restored = MasknmfSettings.from_dict(d)
    assert restored.registration.strategy == "pwrigid"
    assert restored.registration.max_shifts == (7, 9)
    assert restored.demixing.do_demixing == STAGE_FORCE


def test_settings_from_dict_ignores_unknown_keys():
    restored = MasknmfSettings.from_dict(
        {"registration": {"strategy": "rigid", "bogus": 1}, "extra_section": {}}
    )
    assert restored.registration.strategy == "rigid"


def test_strategy_kwargs_by_strategy():
    s = MasknmfSettings()
    assert set(s.registration.strategy_kwargs()) == {"max_shifts"}
    s.registration.strategy = "pwrigid"
    assert "num_blocks" in s.registration.strategy_kwargs()
    assert "max_rigid_shifts" in s.registration.strategy_kwargs()


def test_nmf_kwargs_detrender_never_tuple():
    # upstream NMFConfig has a buggy `None,` detrender default; ours must
    # always pass a real None through
    kw = MasknmfSettings().demixing.nmf_kwargs(0.8, ring=True)
    assert kw["detrender"] is None
    assert kw["ring_model_start_pt"] == 0
    # matches the reference NMFConfig default, not demix()'s False default
    assert kw["reassign_background"] is True
    kw = MasknmfSettings().demixing.nmf_kwargs(0.8, ring=False)
    assert kw["ring_model_start_pt"] is None


# ---- stage gating -----------------------------------------------------------


@pytest.mark.parametrize(
    "tri,cached,expected",
    [
        (STAGE_SKIP, False, "skip"),
        (STAGE_SKIP, True, "skip"),
        (STAGE_RUN, False, "compute"),
        (STAGE_RUN, True, "reuse"),
        (STAGE_FORCE, False, "compute"),
        (STAGE_FORCE, True, "compute"),
    ],
)
def test_stage_action(tri, cached, expected):
    assert stage_action(tri, cached) == expected


# ---- suite2p-shaped outputs -------------------------------------------------


def _toy_footprints():
    # two ROIs on a 4x5 grid: roi0 = pixels {(0,0),(0,1)}, roi1 = {(2,3)}
    shape = (4, 5)
    pix = np.array([0, 1, 13])
    roi = np.array([0, 0, 1])
    vals = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    return np.stack([pix, roi]), vals, shape


def test_split_sparse_footprints():
    indices, values, _ = _toy_footprints()
    per_roi = outputs.split_sparse_footprints(indices, values, 2)
    assert len(per_roi) == 2
    np.testing.assert_array_equal(per_roi[0][0], [0, 1])
    np.testing.assert_array_equal(per_roi[1][0], [13])
    np.testing.assert_allclose(per_roi[1][1], [2.0])


def test_roi_stat_coordinates():
    indices, values, shape = _toy_footprints()
    per_roi = outputs.split_sparse_footprints(indices, values, 2)
    stat = outputs.roi_stat(*per_roi[1], shape)
    # flat index 13 in C-order on (4,5) -> y=2, x=3
    np.testing.assert_array_equal(stat["ypix"], [2])
    np.testing.assert_array_equal(stat["xpix"], [3])
    assert stat["npix"] == 1
    assert stat["med"] == (2.0, 3.0)


def test_roi_baselines_weighted():
    indices, values, shape = _toy_footprints()
    per_roi = outputs.split_sparse_footprints(indices, values, 2)
    baseline = np.arange(20, dtype=np.float32)  # flat (pixels,)
    b = outputs.roi_baselines(per_roi, baseline)
    # roi0: (0*0.5 + 1*1.0) / 1.5
    np.testing.assert_allclose(b[0], 1.0 / 1.5, rtol=1e-6)
    np.testing.assert_allclose(b[1], 13.0)


def test_write_plane_outputs(tmp_path):
    indices, values, shape = _toy_footprints()
    c = np.vstack([np.sin(np.linspace(0, 6, 50)), np.ones(50)]).T.astype(np.float32)
    info = outputs.write_plane_outputs(
        tmp_path, indices=indices, values=values, c=c, shape=shape, baseline=None
    )
    assert info["n_rois"] == 2
    stat = np.load(tmp_path / "stat.npy", allow_pickle=True)
    iscell = np.load(tmp_path / "iscell.npy")
    F = np.load(tmp_path / "F.npy")
    assert len(stat) == 2
    assert iscell.shape == (2, 2)
    assert (iscell[:, 0] == 1).all()
    assert F.shape == (2, 50)
    for name in ("Fneu.npy", "spks.npy", "norm_traces.npy"):
        assert (tmp_path / name).exists()


def test_merge_ops_roundtrip(tmp_path):
    ops = outputs.merge_ops(tmp_path, {"Ly": 4, "Lx": 5})
    assert ops["Ly"] == 4
    ops = outputs.merge_ops(tmp_path, {"nframes": 50})
    assert ops["Ly"] == 4 and ops["nframes"] == 50
    assert ops["save_path"] == str(tmp_path)
    loaded = np.load(tmp_path / "ops.npy", allow_pickle=True).item()
    assert loaded["nframes"] == 50


# ---- task registration ------------------------------------------------------


def test_task_registered():
    pytest.importorskip("imgui_bundle")
    from mbo_utilities.gui.tasks import TASKS

    assert "masknmf" in TASKS


def test_plane_dirname():
    from mbo_utilities.masknmf.runner import generate_plane_dirname

    assert generate_plane_dirname(3) == "zplane03"
    assert generate_plane_dirname(3, [0, 4999]) == "zplane03_tp00001-05000"


def test_widget_modified_params():
    pytest.importorskip("imgui_bundle")
    from mbo_utilities.gui.widgets.pipelines.masknmf import (
        _collect_modified,
        _is_default,
    )

    s = MasknmfSettings()
    assert _collect_modified(s) == []
    assert _is_default(s.registration, "max_shifts")
    s.registration.max_shifts = (7, 9)
    s.demixing.maxiter = 55
    s.demixing.do_demixing = STAGE_FORCE  # tri-states excluded from the table
    rows = _collect_modified(s)
    assert [r[0] for r in rows] == ["reg.max_shifts", "demix.maxiter"]
    assert rows[1][1] == 55 and rows[1][2] == 40
    # float32 truncation from imgui must not read as modified
    s2 = MasknmfSettings()
    import numpy as np

    s2.demixing.filter_sigma = float(np.float32(s2.demixing.filter_sigma))
    assert _is_default(s2.demixing, "filter_sigma")


def test_find_masknmf_run(tmp_path):
    pytest.importorskip("imgui_bundle")
    from mbo_utilities.gui.widgets.pipelines.masknmf import find_masknmf_run

    # not a run
    assert find_masknmf_run(tmp_path) == (None, None)
    assert find_masknmf_run(None) == (None, None)

    # failed run: stage cache exists, ops.npy never stamped -> outdir only
    plane = tmp_path / "zplane01"
    plane.mkdir()
    (plane / "motion_correction.hdf5").touch()
    params, outdir = find_masknmf_run(tmp_path)
    assert params is None and outdir == str(tmp_path)

    # partial run whose ops.npy predates the masknmf stamp
    np.save(plane / "ops.npy", {"Ly": 4, "Lx": 5})
    params, outdir = find_masknmf_run(plane)
    assert params is None and outdir == str(tmp_path)

    # completed run: parameters come back, from any entry point
    saved = MasknmfSettings()
    saved.demixing.maxiter = 55
    np.save(plane / "ops.npy", {"pipeline": "masknmf", "masknmf": saved.to_dict()})
    for entry in (tmp_path, plane, plane / "ops.npy"):
        params, outdir = find_masknmf_run(entry)
        assert outdir == str(tmp_path)
        assert MasknmfSettings.from_dict(params).demixing.maxiter == 55
