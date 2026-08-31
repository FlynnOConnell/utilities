"""roi_workflow: ROI selection, registered-plane opening, mask extraction,
and the register=none -> extract chain. suite2p / masknmf paths are
exercised only when those packages import."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mbo_utilities.annotation.ngff import LabelsZarr
from mbo_utilities.annotation.store import RoiLabelStore
from mbo_utilities import roi_workflow as rw


def _importable(name: str) -> bool:
    # importorskip only catches ImportError; a half-merged editable install
    # raises SyntaxError and would fail the test instead of skipping it
    try:
        __import__(name)
    except Exception:
        return False
    return True


# ---- fixtures ---------------------------------------------------------------


T, LY, LX = 60, 48, 40


def _disc(cy, cx, r, shape=(LY, LX)):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


@pytest.fixture
def store():
    s = RoiLabelStore(nz=2, ny=LY, nx=LX, label_names=("soma", "dendrite"))
    s.add_roi(0, _disc(10, 10, 4))  # 0: z0 soma
    s.add_roi(0, _disc(30, 25, 5))  # 1: z0 dendrite
    s.add_roi(1, _disc(20, 20, 4))  # 2: z1 soma
    s.add_roi(1, _disc(40, 30, 3))  # 3: z1 unlabeled
    s.set_class(0, 0)
    s.set_class(1, 1)
    s.set_class(2, 0)
    return s


def _write_plane(tmp_path: Path, name: str, plane: int, store: RoiLabelStore, nframes: int = T) -> Path:
    """suite2p-shaped plane dir whose ROI pixels carry a known signal."""
    rng = np.random.default_rng(plane)
    mov = rng.integers(90, 110, size=(nframes, LY, LX)).astype(np.int16)
    z = plane - 1
    for i, rec in enumerate(store.rois):
        if rec.z != z:
            continue
        mask = store.labels[z] == (i + 1)
        # ROI i lights up with amplitude 100*(i+1) on every 10th frame
        mov[::10][:, mask] += np.int16(100 * (i + 1))
    d = tmp_path / name
    d.mkdir()
    mov.tofile(d / "data.bin")
    ops = {"Ly": LY, "Lx": LX, "nframes": nframes, "plane": plane, "fs": 10.0,
           "meanImg": mov.mean(0), "processing_history": []}
    np.save(d / "ops.npy", ops, allow_pickle=True)
    return d


@pytest.fixture
def planes(tmp_path, store):
    return [_write_plane(tmp_path, "zplane01", 1, store), _write_plane(tmp_path, "zplane02", 2, store)]


# ---- selection --------------------------------------------------------------


def test_select_all_by_default(store):
    assert rw.select_rois(store) == [0, 1, 2, 3]


def test_select_intersects_criteria(store):
    assert rw.select_rois(store, planes=[0]) == [0, 1]
    assert rw.select_rois(store, labels=["soma"]) == [0, 2]
    assert rw.select_rois(store, planes=[1], labels=["soma"]) == [2]
    assert rw.select_rois(store, indices=[3, 1]) == [1, 3]
    assert rw.select_rois(store, indices=[3, 1], planes=[0]) == [1]


def test_select_rejects_bad_inputs(store):
    with pytest.raises(KeyError):
        rw.select_rois(store, labels=["axon"])
    with pytest.raises(IndexError):
        rw.select_rois(store, indices=[9])


def test_plane_masks_renumbers_in_order(store):
    img, kept = rw.plane_masks(store, 1, [3, 2, 0])
    assert kept == [3, 2]
    assert set(np.unique(img)) == {0, 1, 2}
    assert (img == 1).sum() == store.rois[3].area
    assert (img == 2).sum() == store.rois[2].area
    img0, kept0 = rw.plane_masks(store, 0, [2, 3])
    assert kept0 == [] and img0.max() == 0


def test_roi_selection_dict_roundtrip():
    sel = rw.RoiSelection(planes=[0], labels=["soma"])
    assert rw.RoiSelection(**sel.to_dict()) == sel


# ---- plane dirs -------------------------------------------------------------


def test_open_registered_and_plane_index(planes):
    mov, ops = rw.open_registered(planes[1])
    assert mov.shape == (T, LY, LX) and mov.dtype == np.int16
    assert rw.plane_index(planes[1], ops) == 1
    assert rw.plane_index(Path("x/zplane07_tp00001-00100")) == 6
    assert rw.plane_index(Path("x/anything")) == 0


def test_open_registered_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        rw.open_registered(tmp_path)
    d = tmp_path / "p"
    d.mkdir()
    np.save(d / "ops.npy", {"Ly": 4, "Lx": 4}, allow_pickle=True)
    with pytest.raises(FileNotFoundError):
        rw.open_registered(d)


def test_load_rois_from_zarr_and_sibling(tmp_path, store):
    fake_data = tmp_path / "raw.tif"
    fake_data.write_bytes(b"")
    LabelsZarr(rw.labels_path(fake_data)).save(store, source_path=fake_data)
    a = rw.load_rois(None, source=fake_data)
    b = rw.load_rois(rw.labels_path(fake_data))
    assert len(a.rois) == len(b.rois) == 4
    assert a.label_names == ("soma", "dendrite")
    assert rw.load_rois(store) is store
    with pytest.raises(ValueError):
        rw.load_rois(None)


# ---- the movie contract -----------------------------------------------------


class _SpatialOnly:
    """Minimal 5-D array that honours y/x keys - the whole contract."""

    def __init__(self, data):
        self.data = data  # (T, C, Z, Y, X)
        self.shape = data.shape
        self.ndim = 5
        self.dtype = data.dtype
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return self.data[key]


def test_plane_movie_forwards_spatial_keys_and_normalizes_shape():
    data = np.arange(3 * 2 * 2 * 5 * 6, dtype=np.int16).reshape(3, 2, 2, 5, 6)
    arr = _SpatialOnly(data)
    m = rw.as_movie(arr, z=1, c=1)
    assert m.shape == (3, 5, 6) and m.ndim == 3 and len(m) == 3
    np.testing.assert_array_equal(m[1:3, 2:4, 1:5], data[1:3, 1, 1, 2:4, 1:5])
    assert arr.reads[-1] == (slice(1, 3), 1, 1, slice(2, 4), slice(1, 5))
    assert m[2].shape == (5, 6)
    assert m[:, 3, 4].shape == (3,)
    assert m[[0, 2], 1, 1:3].shape == (2, 2)
    assert m.frames(0, 2).shape == (2, 5, 6)
    with pytest.raises(IndexError):
        rw.as_movie(arr, z=5)


def test_plane_movie_crop_shifts_keys():
    data = np.arange(4 * 1 * 1 * 10 * 12, dtype=np.int16).reshape(4, 1, 1, 10, 12)
    arr = _SpatialOnly(data)
    m = rw.as_movie(arr)
    assert m.box is None
    c = m.crop(2, 8, 3, 11)
    assert c.shape == (4, 6, 8) and c.box == (2, 8, 3, 11)
    assert c[0, 0, 0] == data[0, 0, 0, 2, 3]
    assert c[0, -1, -1] == data[0, 0, 0, 7, 10]
    np.testing.assert_array_equal(c[1], data[1, 0, 0, 2:8, 3:11])
    np.testing.assert_array_equal(c[:, 1:3, 2:5], data[:, 0, 0, 3:5, 5:8])
    assert arr.reads[-1][3] == slice(3, 5, 1) and arr.reads[-1][4] == slice(5, 8, 1)
    np.testing.assert_array_equal(c[0, [0, 2], 1], data[0, 0, 0, [2, 4], 4])
    # crops of crops compose
    cc = c.crop(1, 4, 2, 6)
    assert cc.box == (3, 6, 5, 9)
    np.testing.assert_array_equal(cc[2], data[2, 0, 0, 3:6, 5:9])
    assert cc.frames(0, 2).shape == (2, 3, 4)
    # a crop covering the whole frame is not a crop
    assert m.crop(0, 10, 0, 12).box is None
    with pytest.raises(IndexError):
        c.crop(0, 7, 0, 1)
    with pytest.raises(IndexError):
        c[0, 6, 0]


def test_feather_mask_and_weighted_trace():
    mask = np.zeros((9, 9), bool)
    mask[2:7, 2:7] = True
    w = rw.feather_mask(mask, edge_width=2)
    assert w[4, 4] == 1.0 and 0 < w[2, 2] < 1.0
    assert w[~mask].max() == 0.0
    mov = np.zeros((3, 1, 1, 9, 9), np.float32)
    mov[:, 0, 0, 4, 4] = 90.0  # centre-only signal
    plain = rw.roi_trace(mov, mask)
    weighted = rw.roi_trace(mov, mask, weights=w)
    assert weighted[0] > plain[0]  # the feather emphasises the centre


def test_shift_matches_numpy_on_a_crop():
    # incl. empty negative-step slices, whose indices() clamp start to -1;
    # the shifted slice must stay empty, not become a full reverse read
    src = np.arange(10)
    keys = (
        slice(2, None, -1), slice(None, None, -1), slice(-10, -8, -1),
        slice(4, 1, -1), slice(1, 4), slice(0, 0), slice(3, 3, -1),
    )
    for off in (0, 3):
        crop = src[off:off + 5]
        for key in keys:
            np.testing.assert_array_equal(
                src[rw._shift(key, off, 5)], crop[key], err_msg=f"{off=} {key=}"
            )


def test_movie_fingerprint_box_sensitivity(tmp_path):
    data = np.zeros((3, 1, 1, 10, 12), np.int16)
    m = rw.as_movie(_SpatialOnly(data))
    assert rw._movie_fingerprint(m, None) == f"array:{m.shape}"
    fp = rw._movie_fingerprint(m.crop(2, 8, 3, 11), None)
    assert fp.startswith("array:") and fp.endswith(":2:8:3:11")
    src = tmp_path / "plane"
    full = rw._movie_fingerprint(m, src)
    assert full == f"{src}:{m.shape}:{m.z}:{m.c}"
    assert rw._movie_fingerprint(m.crop(2, 8, 3, 11), src) != full
    assert rw._movie_fingerprint(m.crop(0, 10, 0, 12), src) == full


def test_plane_movie_lower_ranks():
    d4 = np.zeros((4, 2, 3, 3), np.float32)
    assert rw.as_movie(d4, z=1).shape == (4, 3, 3)
    d3 = np.zeros((4, 3, 3), np.float32)
    assert rw.as_movie(d3).shape == (4, 3, 3)
    d2 = np.ones((3, 3), np.float32)
    m = rw.as_movie(d2)
    assert m.shape == (1, 3, 3) and m[0, 1, 1] == 1.0
    assert rw.as_movie(m) is m


def test_roi_and_pixel_traces_read_only_the_bbox():
    T = 40
    data = np.zeros((T, 1, 1, 20, 30), np.float32)
    data[:, 0, 0, 5:8, 10:14] = np.arange(T)[:, None, None]
    data[:, 0, 0, 6, 11] += 100
    arr = _SpatialOnly(data)
    mask = np.zeros((20, 30), bool)
    mask[5:8, 10:14] = True
    tr = rw.roi_trace(arr, mask, batch=7)
    expected = data[:, 0, 0][:, mask].mean(axis=1)
    np.testing.assert_allclose(tr, expected)
    assert all(k[3] == slice(5, 8) and k[4] == slice(10, 14) for k in arr.reads)
    px = rw.pixel_trace(arr, 6, 11, t=slice(0, 10, 2))
    np.testing.assert_allclose(px, np.arange(0, 10, 2) + 100)
    assert arr.reads[-1][3] == 6 and arr.reads[-1][4] == 11
    with pytest.raises(IndexError):
        rw.pixel_trace(arr, 50, 0)


def test_extract_from_in_memory_array(tmp_path, store):
    rng = np.random.default_rng(3)
    data = rng.integers(90, 110, size=(T, 1, 2, LY, LX)).astype(np.int16)
    mask = store.labels[1] == 3
    data[::10, 0, 1][:, mask] += 300
    arr = _SpatialOnly(data)
    with pytest.raises(ValueError, match="out_dir"):
        rw.extract_rois(arr, store, [2], z=1)
    out = rw.extract_rois(arr, store, [2], z=1, out_dir=tmp_path / "o")
    F = np.load(out / "F.npy")
    assert F.shape == (1, T)
    assert abs((F[0, ::10].mean() - F[0, 1::10].mean()) - 300) < 5
    # only the ROI's (and ring's) bounding box was read, never the full frame
    assert all(k[3] != slice(None) for k in arr.reads if isinstance(k, tuple))
    ops = np.load(out / "ops.npy", allow_pickle=True).item()
    assert ops["Ly"] == LY and ops["source"] is None and "reg_file" not in ops


# ---- extraction (mean engine) ----------------------------------------------


def test_extract_mean_recovers_signal(planes, store):
    out = rw.extract_rois(planes[0], store, [0, 1], engine="mean", tag="t")
    assert out == planes[0] / "rois_t"
    F = np.load(out / "F.npy")
    Fneu = np.load(out / "Fneu.npy")
    assert F.shape == Fneu.shape == (2, T)
    # every 10th frame carries +100*(i+1) inside ROI i, background does not
    for row, i in enumerate([0, 1]):
        bump = F[row, ::10].mean() - F[row, 1::10].mean()
        assert abs(bump - 100 * (i + 1)) < 5, (i, bump)
        assert abs(Fneu[row, ::10].mean() - Fneu[row, 1::10].mean()) < 5
    assert np.load(out / "roi_indices.npy").tolist() == [0, 1]
    stat = np.load(out / "stat.npy", allow_pickle=True)
    assert len(stat) == 2 and stat[0]["npix"] == store.rois[0].area
    assert np.load(out / "iscell.npy").shape == (2, 2)
    recs = json.loads((out / "rois.json").read_text())
    assert [r["label"] for r in recs] == ["soma", "dendrite"]
    ops = np.load(out / "ops.npy", allow_pickle=True).item()
    assert ops["roi_workflow"]["process"] == "extract"
    assert Path(ops["reg_file"]) == planes[0] / "data.bin"
    assert ops["processing_history"][-1]["step"] == "roi_extract"


def test_extract_skips_plane_without_selected_rois(planes, store):
    assert rw.extract_rois(planes[1], store, [0, 1]) is None
    assert not (planes[1] / "rois_manual").exists()


def test_extract_no_neuropil(planes, store):
    out = rw.extract_rois(planes[1], store, None, neuropil=False)
    Fneu = np.load(out / "Fneu.npy")
    assert Fneu.shape == (2, T) and not Fneu.any()


def test_extract_shape_mismatch(planes):
    bad = RoiLabelStore(nz=1, ny=LY + 1, nx=LX)
    bad.add_roi(0, _disc(5, 5, 2, (LY + 1, LX)))
    with pytest.raises(ValueError, match="same field of view"):
        rw.extract_rois(planes[0], bad, [0])


def test_extract_batches_match_single_pass(planes, store):
    a = np.load(rw.extract_rois(planes[0], store, [0, 1], batch_size=7, tag="b7") / "F.npy")
    b = np.load(rw.extract_rois(planes[0], store, [0, 1], batch_size=1000, tag="b1k") / "F.npy")
    np.testing.assert_allclose(a, b, rtol=1e-5)


# ---- run dirs ----------------------------------------------------------------


def _stat_row(y=0, x=0, npix=1):
    return {
        "ypix": np.full(npix, y, np.int32), "xpix": np.full(npix, x, np.int32),
        "lam": np.ones(npix, np.float32), "med": (float(y), float(x)), "npix": npix,
    }


def test_rois_json_uid_from_extract_writer(planes, store):
    out = rw.extract_rois(planes[0], store, [0, 1], tag="uid")
    recs = json.loads((out / "rois.json").read_text())
    assert [r["uid"] for r in recs] == [store.rois[0].uid, store.rois[1].uid]
    assert all(r["uid"] > 0 for r in recs)


def test_load_run_dir_extract(planes, store):
    out = rw.extract_rois(planes[0], store, [0, 1], tag="lr")
    res = rw.load_run_dir(out)
    assert res.kind == "extract" and res.z == 0
    assert res.path == out and res.shape == (LY, LX)
    assert len(res.stat) == 2 and res.F.shape == (2, T) and res.Fneu.shape == (2, T)
    assert res.norm is None
    assert res.uids.tolist() == [store.rois[0].uid, store.rois[1].uid]
    assert res.store_indices.tolist() == [0, 1]
    # stat stays full-FOV, untouched by the loader
    for s, i in zip(res.stat, [0, 1]):
        mask = store.labels[0] == (i + 1)
        assert np.array_equal(np.sort(s["ypix"]), np.sort(np.nonzero(mask)[0]))


def test_load_run_dir_demix_mismatch(tmp_path):
    # masknmf merged 2 seeds into 1 component: row maps must null, not IndexError
    d = tmp_path / "rois_dm"
    d.mkdir()
    np.save(d / "stat.npy", np.array([_stat_row()], dtype=object))
    np.save(d / "F.npy", np.zeros((1, 5), np.float32))
    np.save(d / "Fneu.npy", np.zeros((1, 5), np.float32))
    np.save(d / "iscell.npy", np.ones((1, 2), np.float32))
    np.save(d / "norm_traces.npy", np.zeros((1, 5), np.float32))
    np.save(d / "roi_indices.npy", np.array([0, 1]))
    (d / "rois.json").write_text(json.dumps(
        [{"index": 0, "uid": 3, "z": 0}, {"index": 1, "uid": 4, "z": 0}]
    ))
    np.save(d / "ops.npy", {"Ly": 8, "Lx": 9, "pipeline": "masknmf",
                            "roi_workflow": {"process": "demix", "plane": 1}}, allow_pickle=True)
    res = rw.load_run_dir(d)
    assert res.kind == "demix" and res.z == 1 and res.shape == (8, 9)
    assert res.uids is None and res.store_indices is None
    assert res.F.shape == (1, 5) and res.norm.shape == (1, 5)


def test_load_run_dir_drops_stale_traces(tmp_path):
    # a reused tag dir can hold trace files from a previous, larger run
    d = tmp_path / "rois_stale"
    d.mkdir()
    np.save(d / "stat.npy", np.array([_stat_row(), _stat_row(y=3)], dtype=object))
    np.save(d / "F.npy", np.zeros((3, 5), np.float32))
    np.save(d / "Fneu.npy", np.zeros((2, 5), np.float32))
    np.save(d / "norm_traces.npy", np.zeros((3, 5), np.float32))
    np.save(d / "ops.npy", {"Ly": 8, "Lx": 9,
                            "roi_workflow": {"process": "extract", "plane": 0}}, allow_pickle=True)
    res = rw.load_run_dir(d)
    assert res.F is None and res.norm is None
    assert res.Fneu.shape == (2, 5)


def test_load_run_dir_suite2p_iscell_filter(tmp_path):
    d = tmp_path / "z02"
    d.mkdir()
    np.save(d / "stat.npy", np.array([_stat_row(y=i) for i in range(3)], dtype=object))
    np.save(d / "F.npy", np.arange(15, dtype=np.float32).reshape(3, 5))
    np.save(d / "iscell.npy", np.array([[1, 0.5], [0, 0.5], [1, 0.5]], np.float32))
    np.save(d / "ops.npy", {"Ly": 8, "Lx": 9}, allow_pickle=True)
    res = rw.load_run_dir(d)
    assert res.kind == "suite2p" and res.z == 1
    assert len(res.stat) == 2 and res.Fneu is None
    np.testing.assert_array_equal(res.F, np.arange(15, dtype=np.float32).reshape(3, 5)[[0, 2]])
    assert len(rw.load_run_dir(d, iscell_only=False).stat) == 3


def test_load_run_dir_vanilla_plane_names(tmp_path):
    # vanilla suite2p writes suite2p/plane0, plane1... 0-based
    d = tmp_path / "plane1"
    d.mkdir()
    np.save(d / "stat.npy", np.array([_stat_row()], dtype=object))
    np.save(d / "ops.npy", {"Ly": 8, "Lx": 9}, allow_pickle=True)
    res = rw.load_run_dir(d)
    assert res.kind == "suite2p" and res.z == 1


def test_shift_stat(tmp_path):
    stat = np.array([
        {"ypix": np.array([0, 1], np.int32), "xpix": np.array([2, 3], np.int32),
         "lam": np.ones(2, np.float32), "med": (1.0, 2.0), "npix": 2},
        {"ypix": np.array([5], np.int32), "xpix": np.array([7], np.int32),
         "lam": np.ones(1, np.float32), "med": (5.0, 7.0), "npix": 1},
    ], dtype=object)
    np.save(tmp_path / "stat.npy", stat)
    out = rw._shift_stat(tmp_path, 5, 7, (10, 12))
    saved = np.load(tmp_path / "stat.npy", allow_pickle=True)
    for got in (out, saved):
        assert got[0]["ypix"].tolist() == [5, 6] and got[0]["xpix"].tolist() == [9, 10]
        assert got[0]["med"] == (6.0, 9.0)
        # pixels past the frame edge clip
        assert got[1]["ypix"].tolist() == [9] and got[1]["xpix"].tolist() == [11]


def test_write_discovery_outputs(tmp_path, planes):
    stat = np.array([_stat_row(3, 4, 2), _stat_row(5, 6, 3)], dtype=object)
    F = np.ones((2, T), np.float32)
    info = {"process": "discover", "engine": "masknmf", "box": [2, 8, 3, 11],
            "plane": 0, "channel": 0, "seconds": 0.1}
    ops = np.load(planes[0] / "ops.npy", allow_pickle=True).item()
    out = rw._write_discovery_outputs(
        tmp_path / "rois_f", source=planes[0], ops=ops,
        stat=stat, F=F, Fneu=np.zeros_like(F), info=info,
    )
    names = {p.name for p in out.iterdir()}
    assert {"stat.npy", "F.npy", "Fneu.npy", "spks.npy", "iscell.npy", "rois.json", "ops.npy"} <= names
    assert "roi_indices.npy" not in names
    assert json.loads((out / "rois.json").read_text()) == [{"z": 0, "npix": 2}, {"z": 0, "npix": 3}]
    assert np.load(out / "iscell.npy").shape == (2, 2)
    assert not np.load(out / "spks.npy").any()
    ops2 = np.load(out / "ops.npy", allow_pickle=True).item()
    assert ops2["roi_workflow"]["process"] == "discover"
    assert ops2["roi_workflow"]["box"] == [2, 8, 3, 11]
    assert ops2["processing_history"][-1]["step"] == "roi_discover"
    assert Path(ops2["reg_file"]) == planes[0] / "data.bin"
    res = rw.load_run_dir(out)
    assert res.kind == "discover" and res.z == 0 and res.uids is None


def test_run_discover_wiring(tmp_path, planes):
    with pytest.raises(ValueError, match="box"):
        rw.run(tmp_path, register_method="none", process="discover")
    with pytest.raises(ValueError, match="engine"):
        rw.run(tmp_path, register_method="none", process="discover",
               process_settings={"box": [0, 10, 0, 10], "engine": "bogus"})


# ---- the chain with register=none ------------------------------------------


def test_run_register_none_extract(tmp_path, planes, store):
    outs = rw.run(
        tmp_path, register_method="none", process="extract", rois=store,
        selection={"labels": ["soma"]}, tag="soma",
    )
    assert sorted(outs) == [0, 1]
    assert np.load(outs[0] / "roi_indices.npy").tolist() == [0]
    assert np.load(outs[1] / "roi_indices.npy").tolist() == [2]


def test_run_planes_filter_and_empty_selection(tmp_path, planes, store):
    outs = rw.run(tmp_path, register_method="none", rois=store, planes=[2])
    assert sorted(outs) == [1]
    with pytest.raises(ValueError, match="matched no ROIs"):
        rw.run(tmp_path, register_method="none", rois=store, selection={"planes": [5]})
    with pytest.raises(ValueError, match="save_path"):
        rw.run(tmp_path, register_method="suite2p", rois=store)


def test_run_process_none_registers_only(tmp_path, planes):
    outs = rw.run(tmp_path, register_method="none", process="none")
    assert outs == {0: planes[0], 1: planes[1]}
    assert not list(tmp_path.rglob("rois_*"))


def test_plane_store_prefers_rois_drawn_on_registered_plane(tmp_path, planes, store):
    # ROIs drawn with `mbo out/zplane02` on the registered movie: a
    # single-plane store saved inside the plane dir, z == 0 locally
    local = RoiLabelStore(nz=1, ny=LY, nx=LX, label_names=("soma",))
    local.add_roi(0, _disc(20, 20, 4))
    local.set_class(0, 0)
    LabelsZarr(planes[1] / rw.SAVE_NAME).save(local, source_path=planes[1])

    st, z = rw.plane_store(planes[1], store)
    assert z == 0 and st.nz == 1 and len(st.rois) == 1
    st0, z0 = rw.plane_store(planes[0], store)
    assert st0 is store and z0 == 0

    # no shared store at all: plane 1 (local store) runs, plane 0 is skipped
    outs = rw.run(tmp_path, register_method="none", process="extract", tag="local")
    assert sorted(outs) == [1]
    ops = np.load(outs[1] / "ops.npy", allow_pickle=True).item()
    assert ops["roi_workflow"]["roi_indices"] == [0]
    F = np.load(outs[1] / "F.npy")
    # same pixels as store ROI 2 (z1, index 2 -> amplitude 300 every 10th frame)
    assert abs((F[0, ::10].mean() - F[0, 1::10].mean()) - 300) < 5

    # with a shared store, the local one still wins for its plane; the
    # global plane filter is applied on the plane dir's own index
    outs = rw.run(tmp_path, register_method="none", rois=store, tag="mix")
    assert sorted(outs) == [0, 1]
    assert np.load(outs[0] / "roi_indices.npy").tolist() == [0, 1]
    assert np.load(outs[1] / "roi_indices.npy").tolist() == [0]
    outs = rw.run(tmp_path, register_method="none", rois=store, selection={"planes": [1]}, tag="p1")
    assert sorted(outs) == [1]


def test_run_without_any_store_errors(tmp_path, planes):
    with pytest.raises(FileNotFoundError, match="no ROIs found"):
        rw.run(tmp_path, register_method="none", process="extract")


def test_task_registered():
    from mbo_utilities.gui.tasks import TASKS

    assert "roi_workflow" in TASKS


def test_cli_roi_run(tmp_path, planes, store):
    from click.testing import CliRunner
    from mbo_utilities.cli import main

    LabelsZarr(tmp_path / "manual_labels.zarr").save(store)
    r = CliRunner().invoke(
        main, ["roi-run", str(tmp_path), "--register", "none", "--indices", "1,3", "--tag", "cli"]
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "zplane01" / "rois_cli" / "F.npy").exists()
    assert (tmp_path / "zplane02" / "rois_cli" / "F.npy").exists()
    r = CliRunner().invoke(main, ["roi-run", str(tmp_path), "--register", "none", "--labels", "axon"])
    assert r.exit_code != 0 and "unknown ROI labels" in r.output
    r = CliRunner().invoke(main, ["roi-run", str(tmp_path), "--register", "none", "--process", "none"])
    assert r.exit_code == 0, r.output
    assert "Draw ROIs on the registered movie" in r.output


# ---- optional engines --------------------------------------------------------


def test_extract_suite2p_engine_matches_mean(planes, store):
    pytest.importorskip("suite2p")
    pytest.importorskip("lbm_suite2p_python")
    a = np.load(rw.extract_rois(planes[0], store, [0, 1], engine="mean", neuropil=False, tag="m") / "F.npy")
    b = np.load(rw.extract_rois(planes[0], store, [0, 1], engine="suite2p", neuropil=False, tag="s") / "F.npy")
    assert a.shape == b.shape
    # both are (weighted) means over the same pixels; suite2p uses lam weights
    # from masks_to_stat, so allow a small relative difference
    np.testing.assert_allclose(a, b, rtol=0.05)


# cpu, and no denoiser: its patch sampler finds nothing on a 48x40 field
_MNMF = {"runtime": {"device": "cpu"}, "compression": {"denoise": False}}


def test_demix_seeded_with_masks(tmp_path, store):
    if not (_importable("masknmf") and _importable("torch")):
        pytest.skip("masknmf/torch not importable")
    # PMD's temporal filters need a few hundred frames; 60 is too short
    n = 400
    plane = _write_plane(tmp_path, "zplane01", 1, store, nframes=n)
    out = rw.demix_rois(plane, store, [0, 1], settings=_MNMF, tag="d")
    assert out == plane / "rois_d"
    assert (out / "demixing_results.hdf5").exists()
    F = np.load(out / "F.npy")
    assert F.ndim == 2 and F.shape[1] == n and 1 <= F.shape[0] <= 2
    assert np.load(out / "roi_indices.npy").tolist() == [0, 1]
    recs = json.loads((out / "rois.json").read_text())
    assert [r["uid"] for r in recs] == [store.rois[0].uid, store.rois[1].uid]
    ops = np.load(out / "ops.npy", allow_pickle=True).item()
    assert ops["roi_workflow"]["process"] == "demix"
    assert ops["roi_workflow"]["n_components"] == F.shape[0]
    from mbo_utilities.masknmf.params import PMD_FILE

    assert (plane / PMD_FILE).exists()
    # second call reuses the cached PMD (no recompression)
    out2 = rw.demix_rois(plane, store, [0], settings=_MNMF, tag="d2")
    assert np.load(out2 / "ops.npy", allow_pickle=True).item()["roi_workflow"]["compression_seconds"] == 0.0


def test_discover_masknmf(tmp_path, store):
    if not (_importable("masknmf") and _importable("torch")):
        pytest.skip("masknmf/torch not importable")
    import torch

    if not torch.cuda.is_available():
        # masknmf's superpixel init moves tensors to cuda unconditionally
        pytest.skip("masknmf unseeded init needs cuda")
    n = 400
    plane = _write_plane(tmp_path, "zplane01", 1, store, nframes=n)
    settings = dict(_MNMF, demixing={"patch_size": (20, 20)})
    out = rw.discover_rois(plane, (4, 44, 2, 38), settings=settings, tag="f")
    assert out is not None and out == plane / "rois_f"
    # the crop's PMD cache lives inside the run dir, not beside the plane
    from mbo_utilities.masknmf.params import PMD_FILE

    assert (out / PMD_FILE).exists() and not (plane / PMD_FILE).exists()
    res = rw.load_run_dir(out)
    assert res.kind == "discover" and res.z == 0
    assert res.uids is None and res.store_indices is None
    assert len(res.stat) >= 1 and res.F.shape[1] == n
    for s in res.stat:
        assert 4 <= s["ypix"].min() and s["ypix"].max() < 44
        assert 2 <= s["xpix"].min() and s["xpix"].max() < 38
    ops = np.load(out / "ops.npy", allow_pickle=True).item()
    assert ops["roi_workflow"]["box"] == [4, 44, 2, 38]


def test_discover_suite2p(tmp_path):
    if not (_importable("suite2p") and _importable("torch")):
        pytest.skip("suite2p/torch not importable")
    # random event times: a periodic signal would alias away at suite2p's
    # tau*fs binning
    rng = np.random.default_rng(0)
    n = 300
    mov = rng.integers(90, 110, size=(n, LY, LX)).astype(np.int16)
    for cy, cx in [(14, 12), (32, 26)]:
        mask = _disc(cy, cx, 4)
        for t in np.flatnonzero(rng.random(n) < 0.08):
            mov[t][mask] += 400
    d = tmp_path / "zplane01"
    d.mkdir()
    mov.tofile(d / "data.bin")
    np.save(d / "ops.npy", {"Ly": LY, "Lx": LX, "nframes": n, "plane": 1,
                            "fs": 10.0, "processing_history": []}, allow_pickle=True)
    out = rw.discover_rois(d, (4, 44, 2, 38), engine="suite2p", tag="s2p")
    assert out == d / "rois_s2p"
    res = rw.load_run_dir(out)
    assert res.kind == "discover" and res.z == 0
    assert len(res.stat) == res.F.shape[0] and res.F.shape[1] == n
    assert res.uids is None and res.store_indices is None
    # stat came back in full-frame coordinates, inside the box
    meds = [tuple(float(v) for v in s["med"]) for s in res.stat]
    for ey, ex in [(14, 12), (32, 26)]:
        assert any(abs(my - ey) <= 2 and abs(mx - ex) <= 2 for my, mx in meds)
    for s in res.stat:
        assert 4 <= s["ypix"].min() and s["ypix"].max() < 44
        assert 2 <= s["xpix"].min() and s["xpix"].max() < 38
    ops = np.load(out / "ops.npy", allow_pickle=True).item()
    assert ops["roi_workflow"]["engine"] == "suite2p"
    assert "roi_indices.npy" not in {p.name for p in out.iterdir()}


def test_plane_movie_resolves_axes_by_name():
    from mbo_utilities.roi_workflow import PlaneMovie, movie_dims

    class Named:
        def __init__(self, data, dims):
            self.data, self.dims, self.shape, self.ndim = data, dims, data.shape, data.ndim

        def __getitem__(self, key):
            return self.data[key]

    data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(3, 2, 4, 5)
    arr = Named(data, ("Z", "T", "Y", "X"))
    assert movie_dims(arr) == ("Z", "T", "Y", "X")
    movie = PlaneMovie(arr, z=2)
    assert movie.shape == (2, 4, 5) and movie.nz == 3
    np.testing.assert_array_equal(movie[1], data[2, 1])
    np.testing.assert_array_equal(movie[:, 1:3, 0], data[2, :, 1:3, 0])
    assert movie_dims(data) == ("T", "Z", "Y", "X")
