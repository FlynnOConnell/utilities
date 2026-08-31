"""
Array class unit tests.

Tests individual array class functionality:
- Instantiation
- Indexing behavior
- Properties (shape, dtype, ndim)
- Protocol compliance (LazyArrayProtocol)
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

import mbo_utilities as mbo
from mbo_utilities.arrays import (
    BinArray,
    H5Array,
    NumpyArray,
    Suite2pArray,
    TiffArray,
    ZarrArray,
)


class TestNumpyArray:
    """Test NumpyArray class."""

    def test_instantiation_from_ndarray(self, synthetic_3d_data):
        """Create NumpyArray from numpy ndarray."""
        arr = NumpyArray(synthetic_3d_data)

        # 3D input (T, Y, X) is promoted to 5D (T, C, Z, Y, X)
        assert arr.shape == (20, 1, 1, 128, 128)
        assert arr.dtype == synthetic_3d_data.dtype
        assert arr.ndim == 5

    def test_indexing_single_frame(self, synthetic_3d_data):
        """Index single frame (5D: drop T, C, Z with integer indices)."""
        arr = NumpyArray(synthetic_3d_data)

        frame = arr[0, 0, 0]
        expected = synthetic_3d_data[0]

        assert frame.shape == expected.shape
        assert np.array_equal(frame, expected)

    def test_indexing_slice(self, synthetic_3d_data):
        """Index with slice over T (C, Z indexed out)."""
        arr = NumpyArray(synthetic_3d_data)

        sliced = arr[2:5, 0, 0]
        expected = synthetic_3d_data[2:5]

        assert sliced.shape == expected.shape
        assert np.array_equal(sliced, expected)

    def test_indexing_fancy(self, synthetic_3d_data):
        """Fancy indexing with list over T (C, Z indexed out)."""
        arr = NumpyArray(synthetic_3d_data)

        indices = [0, 2, 4]
        result = arr[indices, 0, 0]
        expected = synthetic_3d_data[indices]

        assert result.shape == expected.shape
        assert np.array_equal(result, expected)

    def test_len(self, synthetic_3d_data):
        """Test __len__."""
        arr = NumpyArray(synthetic_3d_data)
        assert len(arr) == synthetic_3d_data.shape[0]

    def test_has_metadata(self, synthetic_3d_data):
        """NumpyArray should have metadata attribute."""
        arr = NumpyArray(synthetic_3d_data)
        assert hasattr(arr, "metadata")
        assert isinstance(arr.metadata, dict)


class TestBinArray:
    """Test BinArray class."""

    @pytest.fixture
    def bin_file(self, synthetic_3d_data, tmp_path):
        """Create a temporary binary file for testing."""
        bin_path = tmp_path / "test.bin"

        # Write binary file
        mmap = np.memmap(bin_path, mode="w+", dtype=np.int16, shape=synthetic_3d_data.shape)
        mmap[:] = synthetic_3d_data
        mmap.flush()
        del mmap

        # Write ops.npy
        ops = {
            "Ly": synthetic_3d_data.shape[1],
            "Lx": synthetic_3d_data.shape[2],
            "nframes": synthetic_3d_data.shape[0],
        }
        np.save(tmp_path / "ops.npy", ops)

        return bin_path, synthetic_3d_data

    def test_instantiation_with_shape(self, bin_file):
        """Create BinArray with explicit shape."""
        bin_path, expected_data = bin_file

        arr = BinArray(bin_path, shape=expected_data.shape, dtype=np.int16)

        assert arr.shape == expected_data.shape
        assert arr.dtype == np.int16

    def test_instantiation_infers_from_ops(self, bin_file):
        """BinArray should infer shape from ops.npy."""
        bin_path, expected_data = bin_file

        arr = BinArray(bin_path)

        assert arr.shape == expected_data.shape

    def test_indexing(self, bin_file):
        """Test indexing BinArray."""
        bin_path, expected_data = bin_file

        arr = BinArray(bin_path)

        frame = arr[0]
        assert np.array_equal(frame, expected_data[0])

    def test_setitem(self, tmp_path, synthetic_3d_data):
        """Test writing to BinArray."""
        bin_path = tmp_path / "writable.bin"

        arr = BinArray(bin_path, shape=synthetic_3d_data.shape, dtype=np.int16)

        # Write data
        arr[0] = synthetic_3d_data[0]
        arr.flush()

        # Read back
        assert np.array_equal(arr[0], synthetic_3d_data[0])

    def test_context_manager(self, bin_file):
        """Test BinArray as context manager."""
        bin_path, _ = bin_file

        with BinArray(bin_path) as arr:
            _ = arr[0]
        # Should not raise


class TestH5Array:
    """Test H5Array class."""

    @pytest.fixture
    def h5_file(self, synthetic_3d_data, tmp_path):
        """Create a temporary HDF5 file."""
        import h5py

        h5_path = tmp_path / "test.h5"

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("mov", data=synthetic_3d_data)

        return h5_path, synthetic_3d_data

    def test_instantiation(self, h5_file):
        """Create H5Array from file."""
        h5_path, expected_data = h5_file

        arr = H5Array(h5_path)

        # always 5D TCZYX (source is TYX)
        nt, ny, nx = expected_data.shape
        assert arr.shape == (nt, 1, 1, ny, nx)
        assert arr.ndim == 5
        assert arr.dtype == expected_data.dtype

    def test_indexing(self, h5_file):
        """Test indexing H5Array."""
        h5_path, expected_data = h5_file

        arr = H5Array(h5_path)

        # arr[0, 0, 0] -> (Y, X) frame
        frame = arr[0, 0, 0]
        assert frame.shape == expected_data[0].shape
        assert np.array_equal(frame, expected_data[0])

    def test_slice_indexing(self, h5_file):
        """Test slice indexing."""
        h5_path, expected_data = h5_file

        arr = H5Array(h5_path)

        sliced = arr[1:4, 0, 0]
        assert sliced.shape == expected_data[1:4].shape
        assert np.array_equal(sliced, expected_data[1:4])

    def test_nested_autodetect(self, tmp_path):
        """A file whose only dataset is nested ('imaging/data') opens without
        an explicit dataset kwarg and maps (T,Z,Y,X,C) onto TCZYX."""
        import h5py

        p = tmp_path / "nested.h5"
        data = np.random.randint(0, 100, (10, 1, 8, 6, 1)).astype(np.int16)
        with h5py.File(p, "w") as f:
            f.create_dataset("imaging/data", data=data)

        arr = H5Array(p)
        assert arr.dataset_name == "imaging/data"
        assert arr.shape == (10, 1, 1, 8, 6)
        assert np.array_equal(arr[0, 0, 0], data[0, 0, :, :, 0])

    def test_group_named_data_not_selected(self, tmp_path):
        """A GROUP named 'data' must not satisfy the preferred-key probe."""
        import h5py

        p = tmp_path / "groupdata.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("data/x", data=np.zeros((4, 4)))
            f.create_dataset("other", data=np.random.rand(5, 6, 7))

        arr = H5Array(p)
        assert arr.dataset_name == "other"
        assert arr.shape == (5, 1, 1, 6, 7)

    def test_group_sorting_first_regression(self, tmp_path):
        """A group that sorts before the real dataset previously crashed the
        old first-key fallback with an AttributeError."""
        import h5py

        p = tmp_path / "groupfirst.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("aaa/member", data=np.zeros((2, 2)))
            f.create_dataset("zzz", data=np.random.rand(5, 4, 4))

        arr = H5Array(p)
        assert arr.dataset_name == "zzz"
        assert arr.shape == (5, 1, 1, 4, 4)

    def test_channels_last_line_scan(self, tmp_path):
        """(T,Y,X,C) dataset with scan_mode + matching n_channel attrs maps
        the last axis to C, not Z."""
        import h5py

        p = tmp_path / "line.h5"
        raw = np.random.randint(0, 1000, (50, 1, 20, 2)).astype(np.uint16)
        with h5py.File(p, "w") as f:
            d = f.create_dataset("raw", data=raw)
            d.attrs["scan_mode"] = "multiROILineScan"
            d.attrs["n_channel"] = 2

        arr = H5Array(p)
        assert arr.dataset_name == "raw"
        assert arr.shape == (50, 2, 1, 1, 20)

        frame = arr[0, 0, 0]
        assert frame.shape == (1, 20)
        assert np.array_equal(frame, raw[0, :, :, 0])

        chan1 = arr[:, 1, 0]
        assert np.array_equal(chan1, raw[..., 1])

    def test_largest_3d_fallback_warns(self, tmp_path, caplog):
        """No preferred names -> largest >=3D dataset wins, with a warning
        that names the choice and the dataset= override."""
        import h5py
        import logging
        from mbo_utilities import log as mbo_log

        p = tmp_path / "fallback.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("small", data=np.random.rand(5, 4, 4))
            f.create_dataset("big", data=np.random.rand(10, 8, 8))

        h5_logger = mbo_log.get("arrays.h5")
        h5_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.WARNING, logger="mbo.arrays.h5"):
                arr = H5Array(p)
        finally:
            h5_logger.removeHandler(caplog.handler)

        assert arr.dataset_name == "big"
        warning = "\n".join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        )
        assert "big" in warning
        assert "small" in warning
        assert "dataset=" in warning

    def test_explicit_miss_lists_nested_paths(self, tmp_path):
        """Explicit dataset miss reports full nested dataset paths."""
        import h5py

        p = tmp_path / "miss.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("grp/inner", data=np.zeros((3, 4, 4)))

        with pytest.raises(KeyError, match="grp/inner"):
            H5Array(p, dataset="nope")

    def test_reader_kwargs_roundtrip(self, tmp_path):
        """dataset selection survives source_reader_kwargs -> imread."""
        import h5py
        from mbo_utilities.reader import imread, source_reader_kwargs

        p = tmp_path / "roundtrip.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("small", data=np.random.rand(5, 4, 4))
            f.create_dataset("big", data=np.random.rand(10, 8, 8))

        a = imread(p, dataset="small")
        assert source_reader_kwargs(a) == {"dataset": "small"}

        b = imread(p, **source_reader_kwargs(a))
        assert b.dataset_name == "small"

    def test_metadata_group_and_dataset_attrs(self, tmp_path):
        """Parent group attrs and dataset attrs feed metadata; canonical
        params resolve through registered aliases."""
        import h5py
        from mbo_utilities.metadata import get_param

        p = tmp_path / "md.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset(
                "imaging/data", data=np.zeros((5, 1, 4, 4, 1), dtype=np.int16)
            )
            f["imaging"].attrs["frame_rate_hz"] = 19.6568

        arr = H5Array(p)
        md = arr.metadata
        assert get_param(md, "fs") == pytest.approx(19.6568)
        assert md["h5_dataset"] == "imaging/data"
        assert md["h5_raw_dims"] == "TZYXC"
        assert any(d["key"] == "imaging/data" for d in md["h5_datasets"])

        p2 = tmp_path / "md2.h5"
        with h5py.File(p2, "w") as f:
            d = f.create_dataset("raw", data=np.zeros((5, 1, 4, 2), dtype=np.uint16))
            d.attrs["scan_mode"] = "multiROILineScan"
            d.attrs["n_channel"] = 2
            d.attrs["frame_period"] = 0.0006

        md2 = H5Array(p2).metadata
        assert get_param(md2, "fs") == pytest.approx(1666.667, rel=1e-4)
        assert md2["h5_raw_dims"] == "TYXC"

    def test_empty_dataset_skipped(self, tmp_path):
        """An h5py.Empty (null dataspace) dataset must not break opening the
        file or enumerating its datasets."""
        import h5py
        from mbo_utilities.arrays.h5 import list_h5_datasets

        p = tmp_path / "with_empty.h5"
        data = np.random.randint(0, 100, (5, 4, 4)).astype(np.int16)
        with h5py.File(p, "w") as f:
            f.create_dataset("empty", data=h5py.Empty("f4"))
            f.create_dataset("mov", data=data)

        arr = H5Array(p)
        assert arr.dataset_name == "mov"
        assert arr.shape == (5, 1, 1, 4, 4)
        assert np.array_equal(arr[0, 0, 0], data[0])

        keys = [d["key"] for d in list_h5_datasets(p)]
        assert "empty" not in keys
        assert "mov" in keys

    def test_permuted_too_many_indices_raises(self, tmp_path):
        """The permuted (T,Y,X,C) path rejects keys with more than 5 entries
        instead of silently ignoring the extras."""
        import h5py

        p = tmp_path / "permuted.h5"
        raw = np.random.randint(0, 1000, (5, 3, 4, 2)).astype(np.uint16)
        with h5py.File(p, "w") as f:
            d = f.create_dataset("raw", data=raw)
            d.attrs["scan_mode"] = "multiROILineScan"
            d.attrs["n_channel"] = 2

        arr = H5Array(p)
        assert arr.metadata["h5_raw_dims"] == "TYXC"
        with pytest.raises(IndexError, match="too many indices"):
            arr[0, 0, 0, 0, 0, 0]

    def test_more_than_5d_raises_at_open(self, tmp_path):
        """A >5D dataset fails fast at open with a clear error, and the file
        handle is released on the raise."""
        import h5py

        p = tmp_path / "sixd.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("mov", data=np.zeros((2, 2, 2, 2, 2, 2), dtype=np.int16))

        with pytest.raises(ValueError, match="6 dimensions.*up to 5"):
            H5Array(p)

        # handle must be closed: re-opening writable would fail if it leaked
        with h5py.File(p, "a"):
            pass


class TestZarrArray:
    """Test ZarrArray class."""

    @pytest.fixture
    def zarr_store(self, synthetic_3d_data, tmp_path):
        """Create a temporary Zarr store."""
        import zarr

        zarr_path = tmp_path / "test.zarr"

        z = zarr.open_array(
            str(zarr_path),
            mode="w",
            shape=synthetic_3d_data.shape,
            dtype=synthetic_3d_data.dtype,
            chunks=(1, *synthetic_3d_data.shape[1:]),
        )
        z[:] = synthetic_3d_data

        return zarr_path, synthetic_3d_data

    def test_instantiation(self, zarr_store):
        """Create ZarrArray from store."""
        zarr_path, expected_data = zarr_store

        arr = ZarrArray(zarr_path)

        # ZarrArray normalizes to 5D (T, C, Z, Y, X)
        if expected_data.ndim == 3:
            expected_shape = (expected_data.shape[0], 1, 1, expected_data.shape[1], expected_data.shape[2])
        else:
            expected_shape = expected_data.shape

        assert arr.shape == expected_shape
        assert arr.dtype == expected_data.dtype

    def test_indexing(self, zarr_store):
        """Test indexing ZarrArray."""
        zarr_path, expected_data = zarr_store

        arr = ZarrArray(zarr_path)

        frame = arr[0]
        # ZarrArray returns data, shape may differ due to TZYX normalization
        frame_np = np.asarray(frame).squeeze()  # Remove singleton dimensions
        expected_np = np.asarray(expected_data[0]).squeeze()
        # Just verify spatial dimensions match after squeezing
        assert frame_np.shape == expected_np.shape

    def test_has_zstats(self, zarr_store):
        """ZarrArray should have zstats property."""
        zarr_path, _ = zarr_store

        arr = ZarrArray(zarr_path)
        # zstats may be None if not computed, but property should exist
        assert hasattr(arr, "zstats")


class TestTiffArray:
    """Test TiffArray class."""

    @pytest.fixture
    def tiff_file(self, synthetic_3d_data, tmp_path):
        """Create a temporary TIFF file."""
        import tifffile

        tiff_path = tmp_path / "test.tif"
        tifffile.imwrite(tiff_path, synthetic_3d_data)

        return tiff_path, synthetic_3d_data

    def test_instantiation(self, tiff_file):
        """Create TiffArray from file."""
        tiff_path, expected_data = tiff_file

        arr = TiffArray(tiff_path)

        # TiffArray may normalize to 4D (T, Z, Y, X)
        # Check spatial dimensions and frame count
        assert arr.shape[-2:] == expected_data.shape[-2:]
        assert arr.shape[0] == expected_data.shape[0]
        assert arr.dtype == expected_data.dtype

    def test_indexing(self, tiff_file):
        """Test indexing TiffArray."""
        tiff_path, expected_data = tiff_file

        arr = TiffArray(tiff_path)

        frame = arr[0]
        # may have extra Z dimension due to 4D normalization
        frame_np = np.asarray(frame)
        # verify spatial dims are present (order may vary due to 4D normalization)
        frame_squeezed = frame_np.squeeze()
        assert frame_squeezed.shape == expected_data[0].shape, \
            f"Squeezed frame shape mismatch: {frame_squeezed.shape} vs {expected_data[0].shape}"


class TestSuite2pArray:
    """Test Suite2pArray class."""

    @pytest.fixture
    def suite2p_dir(self, synthetic_3d_data, tmp_path):
        """Create a Suite2p-style directory structure."""
        # Suite2pArray expects specific file names: data.bin or data_raw.bin
        s2p_dir = tmp_path / "suite2p" / "plane0"
        s2p_dir.mkdir(parents=True)

        # Write binary - Suite2pArray looks for data.bin or data_raw.bin
        bin_path = s2p_dir / "data.bin"
        mmap = np.memmap(bin_path, mode="w+", dtype=np.int16, shape=synthetic_3d_data.shape)
        mmap[:] = synthetic_3d_data
        mmap.flush()
        del mmap

        # Write ops.npy with required fields
        ops = {
            "Ly": synthetic_3d_data.shape[1],
            "Lx": synthetic_3d_data.shape[2],
            "nframes": synthetic_3d_data.shape[0],
            "filelist": [str(bin_path)],
            "reg_file": str(bin_path),  # Suite2pArray looks for this
        }
        np.save(s2p_dir / "ops.npy", ops)

        return s2p_dir, synthetic_3d_data

    def test_instantiation(self, suite2p_dir):
        """Create Suite2pArray from ops.npy file."""
        s2p_path, expected_data = suite2p_dir

        # Suite2pArray expects ops.npy path, not directory
        ops_path = s2p_path / "ops.npy"
        arr = Suite2pArray(ops_path)

        # Check spatial dimensions and frame count
        assert arr.shape[-2:] == expected_data.shape[-2:]
        assert arr.shape[0] == expected_data.shape[0]

    def test_indexing(self, suite2p_dir):
        """Test indexing Suite2pArray."""
        s2p_path, expected_data = suite2p_dir

        ops_path = s2p_path / "ops.npy"
        arr = Suite2pArray(ops_path)

        # single-plane: shape is (20, 1, 1, 128, 128), arr[0] -> (1, 1, 128, 128)
        frame = arr[0]
        frame_np = np.asarray(frame)
        assert frame_np.ndim == 4
        assert frame_np.shape[-2:] == expected_data[0].shape[-2:]


class TestSuite2pRegTif:
    """Test Suite2pArray reg_tif/ folder support (registered tiffs in
    place of data.bin)."""

    @staticmethod
    def _write_reg_tif_plane(plane_dir: Path, data: np.ndarray, *,
                             channel: int = 0, frames_per_file: int = 7) -> list[Path]:
        """Write `data` (T, Y, X) into plane_dir/reg_tif/ as suite2p-style
        chunks, returning the list of files in write order."""
        import tifffile

        reg_dir = plane_dir / "reg_tif"
        reg_dir.mkdir(parents=True, exist_ok=True)
        files = []
        nframes = data.shape[0]
        for start in range(0, nframes, frames_per_file):
            stop = min(start + frames_per_file, nframes)
            fname = reg_dir / f"file{start:05d}_chan{channel}.tif"
            tifffile.imwrite(fname, data[start:stop])
            files.append(fname)
        return files

    @pytest.fixture
    def suite2p_reg_tif_dir(self, synthetic_3d_data, tmp_path):
        """Plane dir with reg_tif/ files but no data.bin."""
        plane_dir = tmp_path / "suite2p_regtif" / "plane0"
        plane_dir.mkdir(parents=True)

        files = self._write_reg_tif_plane(plane_dir, synthetic_3d_data,
                                          frames_per_file=7)

        ops = {
            "Ly": synthetic_3d_data.shape[1],
            "Lx": synthetic_3d_data.shape[2],
            "nframes": synthetic_3d_data.shape[0],
        }
        np.save(plane_dir / "ops.npy", ops)
        return plane_dir, synthetic_3d_data, files

    def test_find_reg_tif_files_sorted(self, suite2p_reg_tif_dir):
        """Files are returned sorted by numeric frame_start, not lexicographic
        (so file00010 sorts after file00007)."""
        from mbo_utilities.arrays.suite2p import find_suite2p_reg_tif_files

        plane_dir, _, expected_files = suite2p_reg_tif_dir
        found = find_suite2p_reg_tif_files(plane_dir, channel=0)
        assert found == expected_files

    def test_find_reg_tif_files_filters_channel(self, suite2p_reg_tif_dir, tmp_path):
        """Only the requested channel's files come back."""
        from mbo_utilities.arrays.suite2p import find_suite2p_reg_tif_files
        import tifffile

        plane_dir, data, _ = suite2p_reg_tif_dir
        # add a chan1 file that should be excluded
        tifffile.imwrite(plane_dir / "reg_tif" / "file00000_chan1.tif", data[:2])

        chan0 = find_suite2p_reg_tif_files(plane_dir, channel=0)
        chan1 = find_suite2p_reg_tif_files(plane_dir, channel=1)
        assert all("chan0" in p.name for p in chan0)
        assert len(chan1) == 1 and "chan1" in chan1[0].name

    def test_fallback_to_reg_tif_when_no_bin(self, suite2p_reg_tif_dir):
        """Suite2pArray loads reg_tif/ when no data.bin exists."""
        plane_dir, expected, _ = suite2p_reg_tif_dir
        arr = Suite2pArray(plane_dir / "ops.npy")

        assert arr.shape == (expected.shape[0], 1, 1, expected.shape[1], expected.shape[2])
        assert arr.dtype == np.int16

    def test_reg_tif_frame_values_across_boundaries(self, suite2p_reg_tif_dir):
        """Frames read back match what was written, including across file
        boundaries (frames_per_file=7 with 20 total frames produces 3 files)."""
        plane_dir, expected, _ = suite2p_reg_tif_dir
        arr = Suite2pArray(plane_dir / "ops.npy")

        # frame 0 (first file)
        f0 = np.asarray(arr[0]).squeeze()
        assert np.array_equal(f0, expected[0])

        # frame 8 (second file - crosses boundary)
        f8 = np.asarray(arr[8]).squeeze()
        assert np.array_equal(f8, expected[8])

        # T-slice spanning file boundaries
        sl = np.asarray(arr[5:15]).squeeze()  # squeeze C=1, Z=1
        assert sl.shape == (10, expected.shape[1], expected.shape[2])
        assert np.array_equal(sl, expected[5:15])

    def test_use_reg_tif_overrides_data_bin(self, synthetic_3d_data, tmp_path):
        """When both data.bin and reg_tif exist with different content,
        use_reg_tif=True picks reg_tif; default picks data.bin."""
        plane_dir = tmp_path / "suite2p_both" / "plane0"
        plane_dir.mkdir(parents=True)

        # data.bin gets `synthetic_3d_data`
        bin_path = plane_dir / "data.bin"
        mmap = np.memmap(bin_path, mode="w+", dtype=np.int16,
                         shape=synthetic_3d_data.shape)
        mmap[:] = synthetic_3d_data
        mmap.flush()
        del mmap

        # reg_tif gets a *different* array (negated) so the two sources
        # are distinguishable
        reg_data = (-synthetic_3d_data).astype(np.int16)
        self._write_reg_tif_plane(plane_dir, reg_data, frames_per_file=7)

        np.save(plane_dir / "ops.npy", {
            "Ly": synthetic_3d_data.shape[1],
            "Lx": synthetic_3d_data.shape[2],
            "nframes": synthetic_3d_data.shape[0],
        })

        # default: data.bin wins
        arr_bin = Suite2pArray(plane_dir / "ops.npy")
        f0_bin = np.asarray(arr_bin[0]).squeeze()
        assert np.array_equal(f0_bin, synthetic_3d_data[0])

        # explicit opt-in: reg_tif wins
        arr_tif = Suite2pArray(plane_dir / "ops.npy", use_reg_tif=True)
        f0_tif = np.asarray(arr_tif[0]).squeeze()
        assert np.array_equal(f0_tif, reg_data[0])

    def test_imread_on_reg_tif_folder_routes_to_suite2p(self, suite2p_reg_tif_dir):
        """imread() pointed at a reg_tif/ folder must route through
        Suite2pArray (numeric sort) — NOT TiffArray (lexicographic glob),
        which would scramble variable-width suite2p filenames like
        file00500_chan0.tif vs file001000_chan0.tif."""
        plane_dir, expected, _ = suite2p_reg_tif_dir
        reg_dir = plane_dir / "reg_tif"

        arr = mbo.imread(reg_dir)

        # Must be a Suite2pArray (5D, int16) — TiffArray would give a
        # different shape contract.
        assert isinstance(arr, Suite2pArray), f"Got {type(arr).__name__}, expected Suite2pArray"
        assert arr.shape == (expected.shape[0], 1, 1, expected.shape[1], expected.shape[2])
        assert arr.dtype == np.int16

        # Frame order must match the original — verifies numeric sort.
        for t in (0, 8, 14, 19):
            frame = np.asarray(arr[t]).squeeze()
            assert np.array_equal(frame, expected[t]), f"Frame {t} mismatch"

    def test_use_reg_tif_raises_if_missing(self, tmp_path, synthetic_3d_data):
        """use_reg_tif=True with no reg_tif/ raises FileNotFoundError."""
        plane_dir = tmp_path / "suite2p_bin_only" / "plane0"
        plane_dir.mkdir(parents=True)
        bin_path = plane_dir / "data.bin"
        mmap = np.memmap(bin_path, mode="w+", dtype=np.int16,
                         shape=synthetic_3d_data.shape)
        mmap[:] = synthetic_3d_data
        mmap.flush()
        del mmap
        np.save(plane_dir / "ops.npy", {
            "Ly": synthetic_3d_data.shape[1],
            "Lx": synthetic_3d_data.shape[2],
            "nframes": synthetic_3d_data.shape[0],
        })

        with pytest.raises(FileNotFoundError, match="use_reg_tif"):
            Suite2pArray(plane_dir / "ops.npy", use_reg_tif=True)


class TestTiffVolumeAutoDetection:
    """Test TiffArray auto-detection of volume directories."""

    @pytest.fixture
    def tiff_volume_dir(self, synthetic_4d_data, tmp_path):
        """Create a directory with planeXX.tiff files."""
        import tifffile

        vol_dir = tmp_path / "volume"
        vol_dir.mkdir()

        # synthetic_4d_data is (T, Z, Y, X) - write each plane as separate file
        n_planes = synthetic_4d_data.shape[1]
        for z in range(n_planes):
            plane_data = synthetic_4d_data[:, z, :, :]  # (T, Y, X)
            plane_path = vol_dir / f"plane{z:02d}.tiff"
            tifffile.imwrite(plane_path, plane_data)

        return vol_dir, synthetic_4d_data

    def test_volume_directory_detection(self, tiff_volume_dir):
        """TiffArray detects volume directory; always 5D TCZYX (C=1)."""
        vol_dir, expected_data = tiff_volume_dir

        arr = TiffArray(vol_dir)

        nt, nz, ny, nx = expected_data.shape
        assert arr.ndim == 5, f"Expected 5D array, got {arr.ndim}D"
        assert arr.shape == (nt, 1, nz, ny, nx)

    def test_volume_indexing(self, tiff_volume_dir):
        """Test indexing volume TiffArray with 5D semantics."""
        vol_dir, expected_data = tiff_volume_dir

        arr = TiffArray(vol_dir)

        # arr[0, 0] indexes T and C -> (Z, Y, X)
        frame = arr[0, 0]
        frame_np = np.asarray(frame)
        assert frame_np.size == expected_data[0].size, "Element count mismatch"
        assert frame_np.ndim == 3, "T,C-indexed frame should be 3D (Z, Y, X)"

    def test_find_tiff_plane_files(self, tiff_volume_dir):
        """Test find_tiff_plane_files helper function."""
        from mbo_utilities.arrays.tiff import find_tiff_plane_files

        vol_dir, expected_data = tiff_volume_dir
        n_planes = expected_data.shape[1]

        plane_files = find_tiff_plane_files(vol_dir)

        assert len(plane_files) == n_planes
        # should be sorted by plane number
        for i, f in enumerate(plane_files):
            assert f"plane{i:02d}" in f.name


class TestSuite2pVolumeAutoDetection:
    """Test Suite2pArray auto-detection of multi-plane directories."""

    @pytest.fixture
    def suite2p_volume_dir(self, synthetic_4d_data, tmp_path):
        """Create a Suite2p multi-plane directory structure."""
        parent_dir = tmp_path / "suite2p"
        parent_dir.mkdir()

        # synthetic_4d_data is (T, Z, Y, X)
        n_planes = synthetic_4d_data.shape[1]

        for z in range(n_planes):
            plane_dir = parent_dir / f"plane{z}"
            plane_dir.mkdir()

            plane_data = synthetic_4d_data[:, z, :, :]  # (T, Y, X)

            # write binary file
            bin_path = plane_dir / "data.bin"
            mmap = np.memmap(bin_path, mode="w+", dtype=np.int16, shape=plane_data.shape)
            mmap[:] = plane_data
            mmap.flush()
            del mmap

            # write ops.npy
            ops = {
                "Ly": plane_data.shape[1],
                "Lx": plane_data.shape[2],
                "nframes": plane_data.shape[0],
                "reg_file": str(bin_path),
            }
            np.save(plane_dir / "ops.npy", ops)

        return parent_dir, synthetic_4d_data

    def test_volume_directory_detection(self, suite2p_volume_dir):
        """Suite2pArray should detect multi-plane directory and return 5D shape."""
        parent_dir, expected_data = suite2p_volume_dir

        arr = Suite2pArray(parent_dir)

        # should be 5D: (T, C, Z, Y, X) with C=1
        assert arr.ndim == 5, f"Expected 5D array, got {arr.ndim}D"
        assert arr.shape[0] == expected_data.shape[0], "Frame count mismatch"
        assert arr.shape[1] == 1, "Channel dim should be 1"
        assert arr.shape[2] == expected_data.shape[1], "Plane count mismatch"
        assert arr.shape[3:] == expected_data.shape[2:], "Spatial dims mismatch"

    def test_volume_indexing(self, suite2p_volume_dir):
        """Test indexing volume Suite2pArray."""
        parent_dir, expected_data = suite2p_volume_dir

        arr = Suite2pArray(parent_dir)

        # index single timepoint: 5D -> 4D (C, Z, Y, X) = (1, 3, 64, 64)
        frame = arr[0]
        frame_np = np.asarray(frame)
        assert frame_np.size == expected_data[0].size, "Element count mismatch"
        assert frame_np.ndim == 4, "Single frame should be 4D (C, Z, Y, X)"

    def test_find_suite2p_plane_dirs(self, suite2p_volume_dir):
        """Test find_suite2p_plane_dirs helper function."""
        from mbo_utilities.arrays.suite2p import find_suite2p_plane_dirs

        parent_dir, expected_data = suite2p_volume_dir
        n_planes = expected_data.shape[1]

        plane_dirs = find_suite2p_plane_dirs(parent_dir)

        assert len(plane_dirs) == n_planes
        # should be sorted by plane number
        for i, d in enumerate(plane_dirs):
            assert f"plane{i}" in d.name


class TestProtocolCompliance:
    """Test that array classes comply with expected protocols."""

    @pytest.fixture(params=["numpy", "bin", "h5", "zarr", "tiff"])
    def array_instance(self, request, synthetic_3d_data, tmp_path):
        """Create array instance of each type."""
        array_type = request.param

        if array_type == "numpy":
            return NumpyArray(synthetic_3d_data)

        elif array_type == "bin":
            bin_path = tmp_path / "test.bin"
            mmap = np.memmap(bin_path, mode="w+", dtype=np.int16, shape=synthetic_3d_data.shape)
            mmap[:] = synthetic_3d_data
            mmap.flush()
            del mmap
            ops = {"Ly": synthetic_3d_data.shape[1], "Lx": synthetic_3d_data.shape[2], "nframes": synthetic_3d_data.shape[0]}
            np.save(tmp_path / "ops.npy", ops)
            return BinArray(bin_path)

        elif array_type == "h5":
            import h5py
            h5_path = tmp_path / "test.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("mov", data=synthetic_3d_data)
            return H5Array(h5_path)

        elif array_type == "zarr":
            import zarr
            zarr_path = tmp_path / "test.zarr"
            z = zarr.open_array(str(zarr_path), mode="w", shape=synthetic_3d_data.shape, dtype=synthetic_3d_data.dtype)
            z[:] = synthetic_3d_data
            return ZarrArray(zarr_path)

        elif array_type == "tiff":
            import tifffile
            tiff_path = tmp_path / "test.tif"
            tifffile.imwrite(tiff_path, synthetic_3d_data)
            return TiffArray(tiff_path)

    def test_has_shape(self, array_instance):
        """All arrays should have shape attribute."""
        assert hasattr(array_instance, "shape")
        assert isinstance(array_instance.shape, tuple)

    def test_has_dtype(self, array_instance):
        """All arrays should have dtype attribute."""
        assert hasattr(array_instance, "dtype")

    def test_has_ndim(self, array_instance):
        """All arrays should have ndim attribute."""
        assert hasattr(array_instance, "ndim")
        assert isinstance(array_instance.ndim, int)

    def test_indexable(self, array_instance):
        """All arrays should support indexing."""
        frame = array_instance[0]
        assert frame is not None

    def test_has_len(self, array_instance):
        """All arrays should support len()."""
        # Some arrays may not implement __len__, use shape[0] as fallback
        try:
            length = len(array_instance)
            assert length == array_instance.shape[0]
        except TypeError:
            # If no __len__, verify shape[0] is accessible
            assert array_instance.shape[0] > 0

    def test_has_metadata(self, array_instance):
        """All arrays should have metadata attribute."""
        assert hasattr(array_instance, "metadata")


class TestImreadDispatcher:
    """Test that imread returns correct array types."""

    def test_imread_tiff(self, tmp_path, synthetic_3d_data):
        """imread should return appropriate array for TIFF."""
        import tifffile

        tiff_path = tmp_path / "test.tif"
        tifffile.imwrite(tiff_path, synthetic_3d_data)

        arr = mbo.imread(tiff_path)

        # Should return some array-like object with correct spatial dims
        assert hasattr(arr, "shape")
        assert arr.shape[-2:] == synthetic_3d_data.shape[-2:]
        assert arr.shape[0] == synthetic_3d_data.shape[0]

    def test_imread_zarr(self, tmp_path, synthetic_3d_data):
        """imread should return ZarrArray for .zarr."""
        import zarr

        zarr_path = tmp_path / "test.zarr"
        z = zarr.open_array(str(zarr_path), mode="w", shape=synthetic_3d_data.shape, dtype=synthetic_3d_data.dtype)
        z[:] = synthetic_3d_data

        arr = mbo.imread(zarr_path)

        # ZarrArray normalizes to 4D, check spatial dims and frame count
        assert hasattr(arr, "shape")
        assert arr.shape[-2:] == synthetic_3d_data.shape[-2:]
        assert arr.shape[0] == synthetic_3d_data.shape[0]

    def test_imread_h5(self, tmp_path, synthetic_3d_data):
        """imread should return H5Array for .h5."""
        import h5py

        h5_path = tmp_path / "test.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("mov", data=synthetic_3d_data)

        arr = mbo.imread(h5_path)

        assert hasattr(arr, "shape")
        nt, ny, nx = synthetic_3d_data.shape
        assert arr.shape == (nt, 1, 1, ny, nx)

    def test_imread_bin(self, tmp_path, synthetic_3d_data):
        """imread should return array for binary directory."""
        bin_dir = tmp_path / "suite2p"
        bin_dir.mkdir()

        bin_path = bin_dir / "data_raw.bin"
        mmap = np.memmap(bin_path, mode="w+", dtype=np.int16, shape=synthetic_3d_data.shape)
        mmap[:] = synthetic_3d_data
        mmap.flush()
        del mmap

        ops = {"Ly": synthetic_3d_data.shape[1], "Lx": synthetic_3d_data.shape[2], "nframes": synthetic_3d_data.shape[0]}
        np.save(bin_dir / "ops.npy", ops)

        arr = mbo.imread(bin_dir)

        # imread returns Suite2pArray which is always 5D (T, C, Z, Y, X)
        assert hasattr(arr, "shape")
        assert arr.shape == (20, 1, 1, 128, 128)


def test_plane_color_layout_channels_are_not_zplanes():
    # 2 PMT channels with sparse metadata read as colors, not depth; LBM
    # interleaves (many pages, or a stack_type saying so) stay z-planes
    from mbo_utilities.arrays.tiff import _plane_color_layout

    assert _plane_color_layout({"nchannels": 2}, 2) == (1, 2)
    assert _plane_color_layout({}, 30) == (30, 1)
    assert _plane_color_layout({"stack_type": "lbm"}, 2) == (2, 1)
    assert _plane_color_layout({"stack_type": "single_plane"}, 2) == (1, 2)
    assert _plane_color_layout({"num_planes": 1, "num_color_channels": 2}, 2) == (1, 2)
    assert _plane_color_layout({"num_planes": 14}, 28) == (14, 2)
    assert _plane_color_layout({"num_color_channels": 2}, 28) == (14, 2)
