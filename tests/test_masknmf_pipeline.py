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


def test_roi_calibration_unweighted_over_support():
    """masknmf's convention: mean over the support, not a lam-weighted mean."""
    indices, values, shape = _toy_footprints()
    per_roi = outputs.split_sparse_footprints(indices, values, 2)
    n_pix = shape[0] * shape[1]
    var_img = np.full(n_pix, 2.0, dtype=np.float32)
    mean_img = np.arange(n_pix, dtype=np.float32)
    baseline = np.ones(n_pix, dtype=np.float32)

    gain, f0 = outputs.roi_calibration(
        per_roi, var_img=var_img, mean_img=mean_img, baseline=baseline
    )
    # roi0 lam = [0.5, 1.0] on pixels [0, 1]; gain = mean(lam * 2) = 1.5
    np.testing.assert_allclose(gain[0], 1.5, rtol=1e-6)
    # roi1 lam = [2.0] on pixel 13; gain = 2.0 * 2 = 4.0
    np.testing.assert_allclose(gain[1], 4.0, rtol=1e-6)
    # F0 = mean_support(b * var_img + mean_img); roi0 -> mean([2+0, 2+1]) = 2.5
    np.testing.assert_allclose(f0[0], 2.5, rtol=1e-6)
    np.testing.assert_allclose(f0[1], 2.0 + 13.0, rtol=1e-6)


def test_roi_calibration_defaults_are_inert():
    """No PMD images -> gain is the mean lam and F0 is 0 (uncalibrated)."""
    indices, values, shape = _toy_footprints()
    per_roi = outputs.split_sparse_footprints(indices, values, 2)
    gain, f0 = outputs.roi_calibration(per_roi)
    np.testing.assert_allclose(gain[0], 0.75, rtol=1e-6)  # mean([0.5, 1.0])
    np.testing.assert_allclose(gain[1], 2.0, rtol=1e-6)
    assert (f0 == 0).all()


def test_calibrated_traces_zeroes_uncalibrated_rois():
    c = np.ones((10, 2), dtype=np.float32)
    gain = np.array([2.0, 3.0], dtype=np.float32)
    f0 = np.array([50.0, 0.0], dtype=np.float32)  # roi1 has no usable F0
    F, dff = outputs.calibrated_traces(c, gain, f0)
    np.testing.assert_allclose(F[0], 52.0, rtol=1e-6)
    np.testing.assert_allclose(F[1], 3.0, rtol=1e-6)
    np.testing.assert_allclose(dff[0], 4.0, rtol=1e-6)  # 2/50 -> 4%
    assert (dff[1] == 0).all()


def test_write_plane_outputs(tmp_path):
    indices, values, shape = _toy_footprints()
    n_pix = shape[0] * shape[1]
    # real demixed c is nonnegative (c_nonneg=True) with the baseline already
    # factored out into b, which is what makes dF/F well posed here
    c = np.vstack(
        [np.abs(np.sin(np.linspace(0, 6, 50))), np.ones(50)]
    ).T.astype(np.float32)
    info = outputs.write_plane_outputs(
        tmp_path,
        indices=indices,
        values=values,
        c=c,
        shape=shape,
        baseline=np.ones(n_pix, dtype=np.float32),
        var_img=np.full(n_pix, 2.0, dtype=np.float32),
        mean_img=np.full(n_pix, 100.0, dtype=np.float32),
    )
    assert info["n_rois"] == 2
    assert info["n_calibrated"] == 2
    stat = np.load(tmp_path / "stat.npy", allow_pickle=True)
    iscell = np.load(tmp_path / "iscell.npy")
    F = np.load(tmp_path / "F.npy")
    norm = np.load(tmp_path / "norm_traces.npy")
    assert len(stat) == 2
    assert iscell.shape == (2, 2)
    assert (iscell[:, 0] == 1).all()
    assert F.shape == (2, 50)
    # F is calibrated into movie units, so it sits on the F0 = 102 baseline
    # rather than hovering around 0 the way standardised c does.
    assert F.min() > 90.0
    # norm_traces is dF/F in percent against that F0, not a z-score: a
    # nonnegative trace stays nonnegative and is not centred on 0.
    assert norm.shape == (2, 50)
    assert norm.min() >= 0.0
    np.testing.assert_allclose(norm, (F - 102.0) / 102.0 * 100.0, atol=1e-4)
    for name in ("Fneu.npy", "spks.npy"):
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
