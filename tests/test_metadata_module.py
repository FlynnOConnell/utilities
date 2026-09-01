"""
Tests for the metadata module.

Tests the centralized metadata parameter handling including:
- get_param with aliases
- VoxelSize extraction
- ScanImage detection functions
"""

import numpy as np
import pytest
from mbo_utilities.metadata import (
    MetadataParameter,
    VoxelSize,
    METADATA_PARAMS,
    ALIAS_MAP,
    get_canonical_name,
    get_param,
    get_voxel_size,
    normalize_resolution,
    normalize_metadata,
    detect_stack_type,
    is_lbm_stack,
    is_piezo_stack,
    get_saved_channel_ports,
    get_num_color_channels,
    get_num_zplanes,
    get_frames_per_slice,
    get_log_average_factor,
    get_z_step_size,
)


class TestMetadataParameter:
    """Test MetadataParameter dataclass."""

    def test_parameter_creation(self):
        """Create a basic MetadataParameter."""
        param = MetadataParameter(
            canonical="test",
            aliases=("t", "tst"),
            dtype=float,
            unit="Hz",
            default=1.0,
            description="Test param",
        )
        assert param.canonical == "test"
        assert "t" in param.aliases
        assert param.dtype == float
        assert param.default == 1.0

    def test_metadata_params_registry(self):
        """Check METADATA_PARAMS has expected entries."""
        assert "dx" in METADATA_PARAMS
        assert "fs" in METADATA_PARAMS
        assert "num_zplanes" in METADATA_PARAMS

        # check dx parameter
        dx = METADATA_PARAMS["dx"]
        assert dx.unit == "µm"
        assert "PhysicalSizeX" in dx.aliases

        # check num_zplanes has num_planes and nplanes as aliases
        num_zplanes = METADATA_PARAMS["num_zplanes"]
        assert "num_planes" in num_zplanes.aliases
        assert "nplanes" in num_zplanes.aliases

        # check num_timepoints has nframes, num_frames as aliases (suite2p/legacy compat)
        num_tp = METADATA_PARAMS["num_timepoints"]
        assert "nframes" in num_tp.aliases
        assert "num_frames" in num_tp.aliases
        assert "T" in num_tp.aliases


class TestAliasMap:
    """Test alias resolution."""

    def test_canonical_name_lookup(self):
        """get_canonical_name should resolve aliases."""
        assert get_canonical_name("dx") == "dx"
        assert get_canonical_name("PhysicalSizeX") == "dx"
        assert get_canonical_name("pixel_size_x") == "dx"
        assert get_canonical_name("frame_rate") == "fs"
        assert get_canonical_name("fps") == "fs"
        # num_timepoints aliases resolve to the canonical name
        assert get_canonical_name("num_timepoints") == "num_timepoints"
        assert get_canonical_name("nframes") == "num_timepoints"
        assert get_canonical_name("num_frames") == "num_timepoints"
        assert get_canonical_name("T") == "num_timepoints"

    def test_unknown_name_returns_none(self):
        """Unknown names return None."""
        assert get_canonical_name("unknown_param") is None

    def test_case_insensitive(self):
        """Lookups should be case-insensitive."""
        assert get_canonical_name("DX") == "dx"
        assert get_canonical_name("Fs") == "fs"


class TestGetParam:
    """Test get_param function."""

    def test_canonical_key(self):
        """Get param by canonical key."""
        meta = {"dx": 0.5}
        assert get_param(meta, "dx") == 0.5

    def test_alias_key(self):
        """Get param by alias."""
        meta = {"pixel_size_x": 0.5}
        assert get_param(meta, "dx") == 0.5

    def test_another_alias(self):
        """Get param by another alias."""
        meta = {"PhysicalSizeX": 0.5}
        assert get_param(meta, "dx") == 0.5

    def test_frame_rate_aliases(self):
        """Test frame rate with various aliases."""
        assert get_param({"fs": 30.0}, "fs") == 30.0
        assert get_param({"frame_rate": 30.0}, "fs") == 30.0
        assert get_param({"fps": 30.0}, "fs") == 30.0

    def test_default_value(self):
        """Missing param returns default."""
        meta = {}
        assert get_param(meta, "dx", default=1.0) == 1.0

    def test_override_wins(self):
        """Override value takes precedence."""
        meta = {"dx": 0.5}
        assert get_param(meta, "dx", override=0.3) == 0.3

    def test_none_metadata(self):
        """Handle None metadata gracefully."""
        result = get_param(None, "dx", default=1.0)
        assert result == 1.0

    def test_shape_fallback_lx(self):
        """Lx can be inferred from shape."""
        meta = {}
        result = get_param(meta, "Lx", shape=(10, 128, 256))
        assert result == 256

    def test_shape_fallback_ly(self):
        """Ly can be inferred from shape."""
        meta = {}
        result = get_param(meta, "Ly", shape=(10, 128, 256))
        assert result == 128

    def test_pixel_resolution_tuple_dx(self):
        """dx can be extracted from pixel_resolution tuple."""
        meta = {"pixel_resolution": (0.5, 0.6)}
        assert get_param(meta, "dx") == 0.5

    def test_pixel_resolution_tuple_dy(self):
        """dy can be extracted from pixel_resolution tuple."""
        meta = {"pixel_resolution": (0.5, 0.6)}
        assert get_param(meta, "dy") == 0.6


class TestTransformAliases:
    """Reciprocal transform aliases (finterval<->fs, XResolution<->dx)."""

    def test_fs_from_finterval(self):
        """ImageJ finterval (seconds) resolves to fs (Hz)."""
        assert get_param({"finterval": 0.1}, "fs") == 10.0

    def test_finterval_from_fs(self):
        """fs resolves back to finterval."""
        assert get_param({"fs": 10.0}, "finterval") == 0.1

    def test_direct_value_wins_over_transform(self):
        """A directly-stored fs beats deriving it from finterval."""
        assert get_param({"fs": 30.0, "finterval": 0.1}, "fs") == 30.0

    def test_rename_alias_still_wins_over_transform(self):
        """frame_rate (rename alias) resolves before the finterval transform."""
        assert get_param({"frame_rate": 7.5, "finterval": 0.1}, "fs") == 7.5

    def test_dx_from_xresolution(self):
        """TIFF XResolution (px/µm) resolves to dx (µm/px) via reciprocal."""
        assert get_param({"XResolution": 2.0}, "dx") == 0.5

    def test_no_recursion_when_neither_present(self):
        """fs<->finterval are mutually derivable but must not infinite-loop."""
        assert get_param({}, "fs", default=None) is None
        assert get_param({}, "finterval", default=None) is None

    def test_zero_finterval_does_not_crash(self):
        """A zero interval can't be inverted; falls through to default."""
        assert get_param({"finterval": 0.0}, "fs", default=None) is None

    def test_normalize_emits_finterval_from_fs(self):
        """normalize_metadata fans fs out to its finterval transform alias."""
        meta = {"fs": 10.0}
        normalize_metadata(meta)
        assert meta["finterval"] == 0.1

    def test_array_fs_resolves_finterval(self):
        """arr.fs picks up an ImageJ finterval through the registry."""
        import numpy as np
        from mbo_utilities.arrays import NumpyArray

        arr = NumpyArray(np.zeros((4, 1, 1, 8, 8), dtype=np.int16),
                         metadata={"finterval": 0.25})
        assert arr.fs == 4.0


class TestVoxelSize:
    """Test VoxelSize named tuple."""

    def test_creation(self):
        """Create VoxelSize."""
        vs = VoxelSize(0.5, 0.5, 5.0)
        assert vs.dx == 0.5
        assert vs.dy == 0.5
        assert vs.dz == 5.0

    def test_pixel_resolution_property(self):
        """pixel_resolution returns (dx, dy)."""
        vs = VoxelSize(0.5, 0.6, 5.0)
        assert vs.pixel_resolution == (0.5, 0.6)

    def test_voxel_size_property(self):
        """voxel_size returns (dx, dy, dz)."""
        vs = VoxelSize(0.5, 0.6, 5.0)
        assert vs.voxel_size == (0.5, 0.6, 5.0)

    def test_to_dict(self):
        """to_dict returns expected keys."""
        vs = VoxelSize(0.5, 0.5, 5.0)
        d = vs.to_dict()
        assert d["dx"] == 0.5
        assert d["dy"] == 0.5
        assert d["dz"] == 5.0
        assert d["pixel_resolution"] == (0.5, 0.5)

    def test_to_dict_includes_aliases(self):
        """to_dict includes standard aliases."""
        vs = VoxelSize(0.5, 0.5, 5.0)
        d = vs.to_dict(include_aliases=True)
        assert d["PhysicalSizeX"] == 0.5
        assert d["PhysicalSizeY"] == 0.5
        assert d["PhysicalSizeZ"] == 5.0
        assert d["z_step"] == 5.0


class TestGetVoxelSize:
    """Test get_voxel_size function."""

    def test_from_canonical_keys(self):
        """Extract from dx, dy, dz keys."""
        meta = {"dx": 0.5, "dy": 0.5, "dz": 5.0}
        vs = get_voxel_size(meta)
        assert vs.dx == 0.5
        assert vs.dz == 5.0

    def test_from_pixel_resolution(self):
        """Extract from pixel_resolution tuple."""
        meta = {"pixel_resolution": (0.5, 0.6)}
        vs = get_voxel_size(meta)
        assert vs.dx == 0.5
        assert vs.dy == 0.6

    def test_from_ome_keys(self):
        """Extract from OME format keys."""
        meta = {"PhysicalSizeX": 0.5, "PhysicalSizeY": 0.6, "PhysicalSizeZ": 5.0}
        vs = get_voxel_size(meta)
        assert vs.dx == 0.5
        assert vs.dy == 0.6
        assert vs.dz == 5.0

    def test_override_values(self):
        """User overrides take precedence."""
        meta = {"dx": 0.5}
        vs = get_voxel_size(meta, dx=0.3)
        assert vs.dx == 0.3

    def test_from_scanimage_nested(self):
        """Extract dz from ScanImage nested structure."""
        meta = {
            "si": {
                "hStackManager": {
                    "stackZStepSize": 5.0
                }
            }
        }
        vs = get_voxel_size(meta)
        assert vs.dz == 5.0

    def test_defaults_to_1(self):
        """Missing values default to 1.0 for non-LBM."""
        vs = get_voxel_size({})
        assert vs.dx == 1.0
        assert vs.dy == 1.0
        assert vs.dz == 1.0

    def test_lbm_no_default_dz(self):
        """LBM stacks should not get default dz - must be user-supplied."""
        # lbm_stack flag
        meta = {"lbm_stack": True}
        vs = get_voxel_size(meta)
        assert vs.dx == 1.0
        assert vs.dy == 1.0
        assert vs.dz is None

        # stack_type == "lbm"
        meta = {"stack_type": "lbm"}
        vs = get_voxel_size(meta)
        assert vs.dz is None

        # but user override still works
        vs = get_voxel_size(meta, dz=20.0)
        assert vs.dz == 20.0

    def test_lbm_ignores_scanimage_dz(self):
        """LBM stacks should not extract dz from ScanImage metadata."""
        # even if ScanImage has hStackManager.stackZStepSize, LBM should ignore it
        meta = {
            "lbm_stack": True,
            "si": {
                "hStackManager": {
                    "stackZStepSize": 5.0,
                    "actualStackZStepSize": 5.0
                }
            }
        }
        vs = get_voxel_size(meta)
        assert vs.dz is None  # should NOT be 5.0


class TestNormalizeResolution:
    """Test normalize_resolution function."""

    def test_adds_aliases(self):
        """normalize_resolution adds all standard aliases."""
        meta = {"dx": 0.5, "dy": 0.5, "dz": 5.0}
        normalize_resolution(meta)
        assert meta["PhysicalSizeX"] == 0.5
        assert meta["PhysicalSizeY"] == 0.5
        assert meta["PhysicalSizeZ"] == 5.0
        assert meta["z_step"] == 5.0


class TestScanImageDetection:
    """Test ScanImage-specific detection functions."""

    def test_detect_lbm_stack(self):
        """LBM stack detected by channelSave length > 2."""
        meta = {
            "si": {
                "hChannels": {
                    "channelSave": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                }
            }
        }
        assert detect_stack_type(meta) == "lbm"
        assert is_lbm_stack(meta) is True
        assert is_piezo_stack(meta) is False

    def test_detect_piezo_stack(self):
        """Piezo stack detected by hStackManager.enable."""
        meta = {
            "si": {
                "hStackManager": {"enable": True, "numSlices": 10},
                "hChannels": {"channelSave": 1}
            }
        }
        assert detect_stack_type(meta) == "piezo"
        assert is_piezo_stack(meta) is True
        assert is_lbm_stack(meta) is False

    def test_detect_single_plane(self):
        """Single plane when neither LBM nor piezo."""
        meta = {
            "si": {
                "hChannels": {"channelSave": 1},
                "hStackManager": {"enable": False}
            }
        }
        assert detect_stack_type(meta) == "single_plane"

    def test_empty_metadata(self):
        """Empty metadata defaults to single_plane."""
        assert detect_stack_type({}) == "single_plane"


class TestLbmColorChannels:
    """Test LBM color channel detection."""

    def test_single_color_channel(self):
        """Single AI source = 1 color channel."""
        meta = {
            "si": {
                "hChannels": {"channelSave": list(range(1, 15))},
                "hScan2D": {
                    "virtualChannelSettings__1": {"source": "AI0"},
                    "virtualChannelSettings__2": {"source": "AI0"},
                    "virtualChannelSettings__3": {"source": "AI0"},
                }
            }
        }
        sources = get_saved_channel_ports(meta)
        assert "AI0" in sources
        assert len(sources) == 1
        assert get_num_color_channels(meta) == 1

    def test_dual_color_channel(self):
        """AI0 + AI1 = 2 color channels."""
        meta = {
            "si": {
                "hChannels": {"channelSave": list(range(1, 18))},
                "hScan2D": {
                    "virtualChannelSettings__1": {"source": "AI0"},
                    "virtualChannelSettings__2": {"source": "AI0"},
                    "virtualChannelSettings__15": {"source": "AI1"},
                    "virtualChannelSettings__16": {"source": "AI1"},
                }
            }
        }
        sources = get_saved_channel_ports(meta)
        assert "AI0" in sources
        assert "AI1" in sources
        assert get_num_color_channels(meta) == 2


class TestPiezoStackParams:
    """Test piezo stack parameter extraction."""

    def test_get_num_zplanes_piezo(self):
        """numSlices from hStackManager."""
        meta = {
            "si": {
                "hStackManager": {"enable": True, "numSlices": 17},
                "hChannels": {"channelSave": 1}
            }
        }
        assert get_num_zplanes(meta) == 17

    def test_get_num_zplanes_lbm(self):
        """channelSave length for LBM."""
        meta = {
            "si": {
                "hChannels": {"channelSave": list(range(1, 15))}
            }
        }
        assert get_num_zplanes(meta) == 14

    def test_get_frames_per_slice(self):
        """framesPerSlice from hStackManager."""
        meta = {
            "si": {
                "hStackManager": {"framesPerSlice": 10}
            }
        }
        assert get_frames_per_slice(meta) == 10

    def test_get_log_average_factor(self):
        """logAverageFactor from hScan2D."""
        meta = {
            "si": {
                "hScan2D": {"logAverageFactor": 5}
            }
        }
        assert get_log_average_factor(meta) == 5

    def test_get_z_step_size(self):
        """stackZStepSize from hStackManager."""
        meta = {
            "si": {
                "hStackManager": {"stackZStepSize": 2.5}
            }
        }
        assert get_z_step_size(meta) == 2.5


class TestRoiInfo:
    """Test ROI and FOV extraction."""

    def test_get_roi_info_basic(self):
        """Extract basic ROI dimensions."""
        from mbo_utilities.metadata import get_roi_info

        meta = {
            "si": {
                "hRoiManager": {
                    "linesPerFrame": 512,
                    "pixelsPerLine": 512,
                }
            }
        }
        info = get_roi_info(meta)
        assert info["roi"] == (512, 512)
        assert info["fov"] == (512, 512)
        assert info["num_mrois"] == 1

    def test_get_roi_info_missing(self):
        """Handle missing ROI info gracefully."""
        from mbo_utilities.metadata import get_roi_info

        info = get_roi_info({})
        assert info["num_mrois"] == 1
        assert "roi" not in info

    def test_get_roi_info_respects_existing_num_rois(self):
        """get_roi_info should use existing num_rois from metadata."""
        from mbo_utilities.metadata import get_roi_info

        # simulate metadata that already has num_rois from get_metadata_single
        meta = {
            "num_rois": 7,  # set by get_metadata_single from RoiGroups
            "si": {
                "hRoiManager": {
                    "linesPerFrame": 68,
                    "pixelsPerLine": 68,
                }
            }
        }
        info = get_roi_info(meta)
        assert info["num_mrois"] == 7
        assert info["fov"] == (7 * 68, 68)

    def test_get_roi_info_uses_roi_groups(self):
        """get_roi_info should count from roi_groups if num_rois not set."""
        from mbo_utilities.metadata import get_roi_info

        meta = {
            "roi_groups": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
            "si": {
                "hRoiManager": {
                    "linesPerFrame": 100,
                    "pixelsPerLine": 100,
                }
            }
        }
        info = get_roi_info(meta)
        assert info["num_mrois"] == 4
        assert info["fov"] == (400, 100)


class TestFrameRate:
    """Test frame rate extraction."""

    def test_get_frame_rate_direct(self):
        """Get frame rate from scanFrameRate."""
        from mbo_utilities.metadata import get_frame_rate

        meta = {
            "si": {
                "hRoiManager": {
                    "scanFrameRate": 30.5
                }
            }
        }
        assert get_frame_rate(meta) == 30.5

    def test_get_frame_rate_from_period(self):
        """Compute frame rate from scanFramePeriod."""
        from mbo_utilities.metadata import get_frame_rate

        meta = {
            "si": {
                "hRoiManager": {
                    "scanFramePeriod": 0.1  # 10 Hz
                }
            }
        }
        assert get_frame_rate(meta) == 10.0

    def test_get_frame_rate_missing(self):
        """Return None if frame rate not available."""
        from mbo_utilities.metadata import get_frame_rate

        assert get_frame_rate({}) is None


class TestColorChannelsUnified:
    """Test that color channel detection works for both LBM and non-LBM."""

    def test_non_lbm_uses_virtual_channel_settings(self):
        """Non-LBM should also use virtualChannelSettings when available."""
        from mbo_utilities.metadata import get_num_color_channels

        meta = {
            "si": {
                "hChannels": {"channelSave": [1, 2]},
                "hStackManager": {"enable": True},
                "hScan2D": {
                    "virtualChannelSettings__1": {"source": "AI0"},
                    "virtualChannelSettings__2": {"source": "AI1"},
                }
            }
        }
        # both channels saved, two distinct AI ports
        assert get_num_color_channels(meta) == 2


class TestCleanScanImageMetadata:
    """Test that clean_scanimage_metadata adds derived fields."""

    def test_adds_stack_detection_fields(self):
        """clean_scanimage_metadata should add lbm_stack, piezo_stack, etc."""
        from mbo_utilities.metadata import clean_scanimage_metadata

        # simulate raw ScanImage metadata with LBM config
        raw_meta = {
            "si": {
                "SI.hChannels.channelSave": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                "SI.hRoiManager.scanFrameRate": 7.5,
                "SI.hRoiManager.linesPerFrame": 512,
                "SI.hRoiManager.pixelsPerLine": 512,
            }
        }
        result = clean_scanimage_metadata(raw_meta)

        assert result["stack_type"] == "lbm"
        assert result["lbm_stack"] is True
        assert result["piezo_stack"] is False
        assert result["num_zplanes"] == 14
        assert result["fs"] == 7.5
        assert result["roi"] == (512, 512)

    def test_adds_piezo_stack_fields(self):
        """clean_scanimage_metadata should detect piezo stack."""
        from mbo_utilities.metadata import clean_scanimage_metadata

        raw_meta = {
            "si": {
                "SI.hChannels.channelSave": 1,
                "SI.hStackManager.enable": True,
                "SI.hStackManager.numSlices": 17,
                "SI.hStackManager.stackZStepSize": 2.5,
            }
        }
        result = clean_scanimage_metadata(raw_meta)

        assert result["stack_type"] == "piezo"
        assert result["lbm_stack"] is False
        assert result["piezo_stack"] is True
        assert result["num_zplanes"] == 17
        assert result["dz"] == 2.5


class TestExtractRoiSlices:
    """Test extract_roi_slices function."""

    def test_extracts_roi_slices(self):
        """extract_roi_slices should compute correct slice boundaries."""
        from mbo_utilities.metadata import extract_roi_slices

        meta = {
            "page_height": 500,
            "page_width": 100,
            "num_fly_to_lines": 10,
            "roi_groups": [
                {"scanfields": {"pixelResolutionXY": [100, 200]}},
                {"scanfields": {"pixelResolutionXY": [100, 280]}},
            ],
        }
        rois = extract_roi_slices(meta)

        assert len(rois) == 2
        assert rois[0]["y_start"] == 0
        assert rois[0]["width"] == 100
        assert rois[1]["y_start"] == rois[0]["y_end"] + 10  # fly-to lines
        assert rois[1]["y_end"] == 500  # last roi ends at page_height

    def test_empty_roi_groups(self):
        """extract_roi_slices should return empty list for no roi_groups."""
        from mbo_utilities.metadata import extract_roi_slices

        assert extract_roi_slices({}) == []
        assert extract_roi_slices({"roi_groups": []}) == []

    def test_missing_page_dimensions(self):
        """extract_roi_slices should return empty list if page dims missing."""
        from mbo_utilities.metadata import extract_roi_slices

        meta = {
            "roi_groups": [{"scanfields": {"pixelResolutionXY": [100, 200]}}],
        }
        assert extract_roi_slices(meta) == []

    def test_single_roi(self):
        """extract_roi_slices should handle single ROI."""
        from mbo_utilities.metadata import extract_roi_slices

        meta = {
            "page_height": 512,
            "page_width": 512,
            "num_fly_to_lines": 0,
            "roi_groups": [{"scanfields": {"pixelResolutionXY": [512, 512]}}],
        }
        rois = extract_roi_slices(meta)

        assert len(rois) == 1
        assert rois[0]["y_start"] == 0
        assert rois[0]["y_end"] == 512
        assert rois[0]["height"] == 512
        assert rois[0]["slice"] == slice(0, 512)


def _rate_alias_source(fs: float = 19.66) -> dict:
    """metadata dict with every registered rate alias stamped, as the
    zarr writer's normalize_metadata pass does at ingest."""
    from mbo_utilities.metadata import METADATA_PARAMS

    src = {"fs": fs}
    for alias in METADATA_PARAMS["fs"].aliases:
        src[alias] = fs
    src["finterval"] = 1.0 / fs
    for alias in METADATA_PARAMS["finterval"].aliases:
        src[alias] = 1.0 / fs
    return src


class TestOutputMetadataRateAliases:
    """OutputMetadata.to_dict keeps every present rate alias consistent."""

    def test_strided_selection_updates_all_aliases(self):
        """stride-4 T selection divides every fs alias, multiplies every
        finterval alias."""
        from mbo_utilities.metadata import METADATA_PARAMS, OutputMetadata

        base = 19.66
        source = _rate_alias_source(base)
        out = OutputMetadata(
            source,
            (11793, 14, 512, 512),
            ("T", "Z", "Y", "X"),
            selections={"T": list(range(0, 11793, 4))},
        )
        result = out.to_dict()

        assert result["is_contiguous"] is True
        expected_fs = pytest.approx(base / 4, rel=1e-6)
        expected_fint = pytest.approx(4 / base, rel=1e-6)
        for key in ("fs", "frame_rate", *METADATA_PARAMS["fs"].aliases):
            assert result[key] == expected_fs, key
        for key in ("finterval", *METADATA_PARAMS["finterval"].aliases):
            assert result[key] == expected_fint, key

    def test_non_contiguous_nulls_all_aliases(self):
        """non-contiguous T selection nulls every present rate alias."""
        from mbo_utilities.metadata import METADATA_PARAMS, OutputMetadata

        base = 19.66
        source = _rate_alias_source(base)
        out = OutputMetadata(
            source,
            (11793, 14, 512, 512),
            ("T", "Z", "Y", "X"),
            selections={"T": [0, 1, 5, 7]},
        )
        result = out.to_dict()

        assert result["is_contiguous"] is False
        assert result["source_fs"] == pytest.approx(base, rel=1e-6)
        for key in ("fs", "frame_rate", *METADATA_PARAMS["fs"].aliases):
            assert result[key] is None, key
        for key in ("finterval", *METADATA_PARAMS["finterval"].aliases):
            assert result[key] is None, key

    def test_no_invented_alias_keys(self):
        """a source without rate aliases gains only the canonical keys."""
        from mbo_utilities.metadata import OutputMetadata

        out = OutputMetadata(
            {"fs": 10.0},
            (100, 1, 32, 32),
            ("T", "Z", "Y", "X"),
        )
        result = out.to_dict()

        assert result["fs"] == 10.0
        assert result["frame_rate"] == 10.0
        assert result["finterval"] == pytest.approx(0.1)
        for absent in ("framerate", "fps", "scanFrameRate", "dt",
                       "frame_interval", "FrameInterval", "time_interval"):
            assert absent not in result, absent


class TestTimeSelectionMetadata:
    """TimeSelection.to_metadata keeps long index lists out of store attrs."""

    def test_long_selection_emits_summary(self):
        """2949-index selection stores first/last/n, not the full list."""
        from mbo_utilities.arrays.features._slicing import (
            parse_timepoint_selection,
        )

        sel = parse_timepoint_selection("1:11793:4", 11793)
        assert sel.count == 2949
        meta = sel.to_metadata()

        assert "include_indices_0based" not in meta
        assert meta["count"] == 2949
        assert meta["include"] == "1:11793:4"
        summary = meta["include_indices_summary"]
        assert summary == {
            "first": sel.final_indices[0],
            "last": sel.final_indices[-1],
            "n": 2949,
        }

    def test_short_selection_emits_explicit_list(self):
        """selections up to max_explicit keep the explicit index list."""
        from mbo_utilities.arrays.features._slicing import (
            parse_timepoint_selection,
        )

        sel = parse_timepoint_selection("1:100", 1000)
        meta = sel.to_metadata()

        assert meta["include_indices_0based"] == list(range(100))
        assert meta["count"] == 100
        assert "include_indices_summary" not in meta

    def test_include_string_round_trips(self):
        """reparsing the stored include string reproduces the indices."""
        from mbo_utilities.arrays.features._slicing import (
            parse_timepoint_selection,
        )

        sel = parse_timepoint_selection("1:11793:4", 11793)
        meta = sel.to_metadata()
        reparsed = parse_timepoint_selection(meta["include"], 11793)

        assert reparsed.final_indices == sel.final_indices
        assert reparsed.count == meta["count"]

    def test_exclude_preserved(self):
        """exclude string and its (small) index list survive."""
        from mbo_utilities.arrays.features._slicing import (
            parse_timepoint_selection,
        )

        sel = parse_timepoint_selection("1:100,50:60", 1000)
        meta = sel.to_metadata()

        assert meta["exclude"] == "50:60"
        assert meta["exclude_indices_0based"] == sel.exclude_indices
        assert "exclude_indices_summary" not in meta

    def test_long_exclude_emits_summary(self):
        """a 40k-index exclude stores first/last/n, not the full list."""
        from mbo_utilities.arrays.features._slicing import (
            parse_timepoint_selection,
        )

        sel = parse_timepoint_selection("1:50000,1:40000", 60000)
        meta = sel.to_metadata()

        assert "exclude_indices_0based" not in meta
        assert meta["exclude"] == "1:40000"
        assert meta["exclude_indices_summary"] == {
            "first": sel.exclude_indices[0],
            "last": sel.exclude_indices[-1],
            "n": len(sel.exclude_indices),
        }

    def test_exclude_at_cap_keeps_explicit_list(self):
        """an exclude list exactly at max_explicit stays explicit."""
        from mbo_utilities.arrays.features._slicing import (
            parse_timepoint_selection,
        )

        sel = parse_timepoint_selection("1:2000,1:1024", 3000)
        assert len(sel.exclude_indices) == 1024
        meta = sel.to_metadata(max_explicit=1024)

        assert meta["exclude_indices_0based"] == sel.exclude_indices
        assert "exclude_indices_summary" not in meta


class TestNormalizeOpsArrays:
    """ops.npy must hold its images as arrays.

    Writers sanitize metadata to JSON types, so a summary image merged into
    ops came back as nested lists; suite2p and lbm_suite2p_python index
    those with .shape and every figure that touches one fails with
    "'list' object has no attribute 'shape'".
    """

    def test_lists_become_arrays_and_empties_are_dropped(self):
        from mbo_utilities.metadata import normalize_ops_arrays

        ops = normalize_ops_arrays({
            "meanImg": [[1.0, 2.0], [3.0, 4.0]],
            "max_proj": [[1.0, 2.0], [3.0, 4.0]],
            "Vcorr": [[0.5, 0.5], [0.5, 0.5]],
            "meanImgE": [],
            "xoff": [1.0, 2.0, 3.0],
            "xrange": [0, 2],
            "fs": 10.0,
        })
        for key in ("meanImg", "max_proj", "Vcorr", "xoff"):
            assert isinstance(ops[key], np.ndarray), key
            assert ops[key].dtype == np.float32, key
        assert ops["meanImg"].shape == (2, 2)
        # absent, not empty: lbm_suite2p_python recomputes meanImgE from
        # meanImg only when the key is missing
        assert "meanImgE" not in ops
        # not every list in ops is an image - suite2p's own xrange is a list
        assert ops["xrange"] == [0, 2]
        assert ops["fs"] == 10.0

    def test_arrays_pass_through_untouched(self):
        from mbo_utilities.metadata import normalize_ops_arrays

        img = np.ones((3, 3), np.float32)
        ops = normalize_ops_arrays({"meanImg": img, "Ly": 3})
        assert ops["meanImg"] is img

    def test_repair_walks_a_run_tree_and_is_idempotent(self, tmp_path):
        from mbo_utilities.metadata import repair_ops_tree

        bad = {"meanImg": [[1.0, 2.0], [3.0, 4.0]], "Ly": 2, "Lx": 2}
        for sub in ("", "zplane01_tp00001-50000", "suite2p/plane0"):
            d = tmp_path / sub if sub else tmp_path
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "ops.npy", dict(bad))

        assert repair_ops_tree(tmp_path) == 3
        for sub in ("", "zplane01_tp00001-50000", "suite2p/plane0"):
            d = tmp_path / sub if sub else tmp_path
            ops = np.load(d / "ops.npy", allow_pickle=True).item()
            assert isinstance(ops["meanImg"], np.ndarray)
        assert repair_ops_tree(tmp_path) == 0

    def test_repair_ignores_a_dir_without_ops(self, tmp_path):
        from mbo_utilities.metadata import repair_ops_tree

        assert repair_ops_tree(tmp_path) == 0
        assert repair_ops_tree(tmp_path / "nope") == 0


class TestStripForExport:
    """strip_for_export size guard and allowlist."""

    def test_drops_oversized_list(self):
        """a 100k-element list is dropped from export metadata."""
        from mbo_utilities.metadata import strip_for_export

        md = {"fs": 10.0, "huge": list(range(100_000))}
        out = strip_for_export(md)

        assert "huge" not in out
        assert out["fs"] == 10.0

    def test_allowlisted_keys_survive(self):
        """plane_shifts / scanphase / ome pass through regardless of size."""
        from mbo_utilities.metadata import strip_for_export

        md = {
            "plane_shifts": [[1, 2]] * 10_000,
            "scanphase": list(range(20_000)),
            "ome": {"multiscales": [{"datasets": [0] * 20_000}]},
        }
        out = strip_for_export(md)

        assert out["plane_shifts"] == md["plane_shifts"]
        assert out["scanphase"] == md["scanphase"]
        assert out["ome"] == md["ome"]

    def test_denylisted_suite2p_fields_dropped(self):
        """suite2p-only fields never reach export metadata."""
        from mbo_utilities.metadata import strip_for_export

        md = {"fs": 10.0, "meanImg_crop": [[0.0]], "badframes0": [0, 1],
              "ihop": [1], "plane_times": [0.1]}
        out = strip_for_export(md)

        assert out == {"fs": 10.0}

    def test_deeply_nested_value_does_not_recurse_out(self):
        """a 2000-deep nested list must not raise RecursionError."""
        from mbo_utilities.metadata import strip_for_export

        deep = [1]
        for _ in range(2000):
            deep = [deep]
        out = strip_for_export({"deep": deep, "fs": 10.0})

        assert out["fs"] == 10.0
        assert out["deep"] == deep  # one leaf element, well under the cap


class TestDecimatedZarrOmeScale:
    """decimating zarr->zarr re-saves must regenerate the OME time scale.

    regression: OutputMetadata.to_dict repaired fs aliases but carried the
    source ome/multiscales/_ome_time_scale through, and the zarr writer's
    attr loop overwrote the freshly written root.attrs['ome'] with that
    stale copy — B ended up with fs=5.0 but the source's t-scale, warning
    on every metadata access.
    """

    def test_every_2nd_frame_resave_updates_ome_time_scale(self, tmp_path):
        import json
        import logging

        import numpy as np

        from mbo_utilities import imread, imwrite
        from mbo_utilities.metadata import params

        data = np.random.randint(0, 100, size=(20, 1, 16, 16), dtype=np.int16)

        a_dir = tmp_path / "A"
        imwrite(data, a_dir, ext=".zarr", metadata={"fs": 10.0},
                dim_order="TZYX", show_progress=False, overwrite=True)
        arr_a = imread(next(a_dir.glob("*.zarr")))
        assert arr_a.metadata["fs"] == pytest.approx(10.0)
        assert arr_a.metadata["_ome_time_scale"] == pytest.approx(0.1)

        b_dir = tmp_path / "B"
        imwrite(arr_a, b_dir, ext=".zarr", timepoints=list(range(1, 21, 2)),
                show_progress=False, overwrite=True)
        b_store = next(b_dir.glob("*.zarr"))

        attrs = json.loads((b_store / "zarr.json").read_text())["attributes"]
        ms = attrs["ome"]["multiscales"][0]
        axes = [ax["name"] for ax in ms["axes"]]
        scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
        assert scale[axes.index("t")] == pytest.approx(0.2)
        assert attrs["fs"] == pytest.approx(5.0)
        assert "_ome_time_scale" not in attrs

        # metadata access on B resolves cleanly, with no stale-alias warning
        params._STALE_WARNED.clear()
        arr_b = imread(b_store)
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        log = logging.getLogger("mbo_utilities")
        log.addHandler(handler)
        try:
            from mbo_utilities.metadata.params import get_param

            assert get_param(arr_b.metadata, "fs") == pytest.approx(5.0)
        finally:
            log.removeHandler(handler)
        stale = [
            r for r in records
            if r.levelno >= logging.WARNING
            and "stale frame-rate" in r.getMessage()
        ]
        assert stale == []
