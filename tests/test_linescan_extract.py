"""``roi_workflow.extract_linescan_traces`` contract tests.

A linescan MESc unit already puts each ROI on its own Z-index, ragged
widths padded to a shared ``Lx`` (see ``tests/test_mesc.py``). These tests
check that the extractor crops each ROI back to its true, unpadded extent
before averaging -- the padding bug this function exists to avoid -- and
that a non-linescan unit is rejected outright.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from mbo_utilities.arrays.mesc import MescArray
from mbo_utilities.roi_workflow import extract_linescan_traces


def _protocol(pattern):
    return json.dumps(
        {
            "protocol": {"scanners": {"mainPatternIndex": 1}},
            "scanPatterns": {"patterns": [pattern]},
        }
    )


def _boxes(boxes):
    """MESc stores 1-based, lower-left/upper-right pixel corners."""
    return [
        {"lowerLeftFramePix": [c0 + 1, r0 + 1], "upperRightFramePix": [c1, r1]}
        for (r0, r1, c0, c1) in boxes
    ]


_GUIDELINE = [[[0, 1], [0, 0], [0, 0]], [[0, 1], [1, 1], [0, 0]]]


@pytest.fixture
def linescan_path(tmp_path):
    """A ``.mesc`` with one linescan unit (2 ragged ROIs) and one chessboard
    unit (real 2D tiles, used to check the modality guard)."""
    path = tmp_path / "linescan.mesc"
    with h5py.File(path, "w") as f:
        s = f.create_group("MSession_0")

        # MUnit_0 - MethodType 6 linescan: 8 frames of 4 lines packed into Y,
        # ROI 0 (width 12) padded to the shared Lx=18 of ROI 1 (width 18).
        u = s.create_group("MUnit_0")
        u.attrs.update(
            {"MethodType": 6, "VecChannelsSize": 1, "TStepInMs": 2.0,
             "MeasurementDatePosix": 1_700_000_000, "Comment": "linescan"}
        )
        u.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": _boxes([(0, 4, 0, 12), (0, 4, 12, 30)])}]}
        )
        u.attrs["MultiROIProtocolJSON"] = _protocol(
            {"guideLine": _GUIDELINE, "pixelSize": 0.5}
        )
        u.create_dataset(
            "Channel_0", data=np.arange(1 * 32 * 30, dtype=np.uint16).reshape(1, 32, 30)
        )

        # MUnit_1 - MethodType 8 chessboard: 4 ROIs tiled along X, real 2D
        # tiles - must be rejected, not silently flattened to a trace.
        u = s.create_group("MUnit_1")
        u.attrs.update(
            {"MethodType": 8, "VecChannelsSize": 1, "TStepInMs": 50.0,
             "MeasurementDatePosix": 1_700_000_100, "Comment": "chessboard"}
        )
        u.attrs["MultiROIProtocolJSON"] = _protocol(
            {
                "centerPoints": np.arange(12).reshape(3, 4).tolist(),
                "pixelSizeX": 0.8,
                "rotation": {"e": [0.0, 0.0, 0.0]},
            }
        )
        u.create_dataset(
            "Channel_0",
            data=np.arange(6 * 32 * 96, dtype=np.uint16).reshape(6, 32, 96),
        )
    return path


def test_crops_padding_before_averaging(linescan_path):
    arr = MescArray(linescan_path, unit=0)
    extents = arr.metadata["mesc_roi_extents"]
    assert (extents[0]["height"], extents[0]["width"]) == (4, 12)
    assert (extents[1]["height"], extents[1]["width"]) == (4, 18)

    out = extract_linescan_traces(arr, compute_dfof=True)
    F = np.load(out / "F.npy")
    Fneu = np.load(out / "Fneu.npy")
    dfof = np.load(out / "dfof.npy")
    stat = np.load(out / "stat.npy", allow_pickle=True)

    assert F.shape == (2, 8)
    assert np.array_equal(Fneu, np.zeros_like(F))
    assert dfof.shape == F.shape
    assert np.all(np.isfinite(dfof))
    assert [int(s["width"]) for s in stat] == [12, 18]

    # the reference: crop each ROI to its *true* extent before averaging --
    # arr[t, 0, z] without cropping would still include ROI 0's zero padding
    for z, ext in enumerate(extents):
        h, w = ext["height"], ext["width"]
        expected = np.array(
            [arr[t, 0, z][:h, :w].mean() for t in range(8)], dtype=np.float32
        )
        assert np.allclose(F[z], expected)
        # padded columns are real zeros in the unpadded array too -- confirm
        # they would have pulled the mean down if left in, so this is a
        # real assertion and not a no-op for a full-width ROI
        if w < arr.shape[-1]:
            padded_mean = np.array(
                [arr[t, 0, z][:h].mean() for t in range(8)], dtype=np.float32
            )
            assert not np.allclose(F[z], padded_mean)


def test_non_linescan_unit_rejected(linescan_path):
    arr = MescArray(linescan_path, unit=1)
    assert arr.metadata["mesc_layout"] != "packed"
    with pytest.raises(ValueError, match="linescan"):
        extract_linescan_traces(arr)


def test_dfof_window_scales_with_fs(linescan_path):
    """A window sized in seconds must actually use fs, not a fixed frame count."""
    arr = MescArray(linescan_path, unit=0)
    out_short = extract_linescan_traces(
        arr, out_dir=arr.filenames[0].parent / "rois_a", dfof_window_s=0.001
    )
    out_long = extract_linescan_traces(
        arr, out_dir=arr.filenames[0].parent / "rois_b", dfof_window_s=50.0
    )
    dfof_short = np.load(out_short / "dfof.npy")
    dfof_long = np.load(out_long / "dfof.npy")
    assert not np.allclose(dfof_short, dfof_long)
