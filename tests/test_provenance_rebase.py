"""
rebase_provenance_paths: stale pipeline-recorded paths are repaired against
the opened tree or dropped — never re-embedded verbatim.

Synthetic cases run on tmp_path. The real-data case exercises the actual lab
layout (every run dir on the share embeds dead 'loson' machine paths) and
skips when X:/data/eunji is not mounted.
"""

from pathlib import Path

import numpy as np
import pytest

from mbo_utilities.metadata.base import (
    PROVENANCE_PATH_KEYS,
    rebase_provenance_paths,
)

DEAD = r"C:\Users\loson\data\movie_0001.tif"


def _make_anchor(tmp_path):
    anchor = tmp_path / "demo" / "run" / "plane01"
    anchor.mkdir(parents=True)
    return anchor


def test_dead_path_rebased_from_grandparent_raw(tmp_path):
    anchor = _make_anchor(tmp_path)
    raw_dir = anchor.parent.parent / "raw"
    raw_dir.mkdir()
    target = raw_dir / "movie_0001.tif"
    target.write_bytes(b"x")

    out = rebase_provenance_paths({"data_path": DEAD}, anchor)

    assert out["data_path"] == str(target)


def test_dead_path_no_match_dropped_rest_intact(tmp_path):
    anchor = _make_anchor(tmp_path)
    md = {"data_path": DEAD, "fs": 17.0, "nframes": 2000}

    out = rebase_provenance_paths(md, anchor)

    assert "data_path" not in out
    assert out["fs"] == 17.0
    assert out["nframes"] == 2000
    # input dict is not mutated
    assert md["data_path"] == DEAD


def test_list_with_live_and_dead_entries(tmp_path):
    anchor = _make_anchor(tmp_path)
    live = anchor / "session.tif"
    live.write_bytes(b"x")

    out = rebase_provenance_paths({"file_paths": [str(live), DEAD]}, anchor)

    assert out["file_paths"] == [str(live)]


def test_empty_string_save_path0_untouched(tmp_path):
    anchor = _make_anchor(tmp_path)

    out = rebase_provenance_paths({"save_path0": ""}, anchor)

    assert out["save_path0"] == ""


def test_non_str_values_untouched_never_raises(tmp_path):
    anchor = _make_anchor(tmp_path)
    md = {
        "data_path": 123,
        "file_list": None,
        "raw_file": ["ok", 42],
        "reg_file": {"nested": "dict"},
        "fast_disk": 3.14,
        "save_path": (1, 2),
    }

    out = rebase_provenance_paths(md, anchor)

    assert out["data_path"] == 123
    assert out["file_list"] is None
    assert out["raw_file"] == ["ok", 42]
    assert out["reg_file"] == {"nested": "dict"}
    assert out["fast_disk"] == 3.14
    assert out["save_path"] == (1, 2)


def test_string_data_path_scalar_shape_preserved(tmp_path):
    # real db.npy stores data_path as a STRING, not a list — the scalar
    # shape must survive the round trip
    anchor = _make_anchor(tmp_path)
    raw_dir = anchor.parent / "raw"
    raw_dir.mkdir()
    target = raw_dir / "movie_0001.tif"
    target.write_bytes(b"x")

    out = rebase_provenance_paths({"data_path": DEAD}, anchor)

    assert isinstance(out["data_path"], str)
    assert out["data_path"] == str(target)


EUNJI = Path("X:/data/eunji")
PLANE = EUNJI / "demo" / "s2p_sparsery" / "zplane01_tp00001-02000"
RAW_TIF = EUNJI / "raw" / "ek350_250610_00001.tif"


def _assert_no_loson(md: dict):
    for key in PROVENANCE_PATH_KEYS:
        val = md.get(key)
        entries = val if isinstance(val, (list, tuple)) else [val]
        for entry in entries:
            if isinstance(entry, str):
                assert "loson" not in entry.lower(), f"{key}: {entry}"


@pytest.mark.skipif(not EUNJI.exists(), reason="X:/data/eunji not mounted")
def test_real_ops_rebase_no_dead_machine_paths_survive():
    ops = np.load(PLANE / "ops.npy", allow_pickle=True).item()

    # anchor at the plane dir: paths whose basenames live inside the run
    # tree rebase in place. The search walks the anchor plus THREE
    # ancestors (and their raw/), so the raw tif at
    # anchor.parent.parent.parent/raw/<basename> is inside the search set
    # even from the plane dir — every source reference rebases to the
    # share's raw copy rather than dropping.
    out = rebase_provenance_paths(ops, anchor=PLANE)
    _assert_no_loson(out)
    assert out["save_path"] == str(PLANE)
    assert out["ops_path"] == str(PLANE / "ops.npy")
    assert out["reg_file"] == str(PLANE / "data.bin")
    assert out["data_path"] == str(RAW_TIF)
    assert out["raw_source"] == str(RAW_TIF)
    assert out["file_paths"] == [str(RAW_TIF)]

    # anchor at the run root (what hydration stores as _s2p_outdir): the
    # raw tif at anchor.parent.parent/raw/<basename> is inside the search
    # set, so every source reference resolves to the share's raw copy.
    out2 = rebase_provenance_paths(ops, anchor=PLANE.parent)
    _assert_no_loson(out2)
    assert out2["data_path"] == str(RAW_TIF)
    assert out2["raw_source"] == str(RAW_TIF)
    assert out2["file_paths"] == [str(RAW_TIF)]
