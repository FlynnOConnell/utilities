"""
base types and data structures for metadata handling.

this module contains the core types used across the metadata system:
- MetadataParameter: standardized parameter definition
- VoxelSize: named tuple for voxel dimensions
- METADATA_PARAMS: central registry of known parameters
- alias lookup utilities
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Any


def _logger():
    # lazy: mbo_utilities.log pulls in package init; base.py is imported early
    from mbo_utilities import log
    return log.get("metadata")


@dataclass
class MetadataParameter:
    """
    Standardized metadata parameter.

    Provides a central registry for parameter names, their aliases across
    different formats (ScanImage, Suite2p, OME, TIFF tags), and type information.

    Attributes
    ----------
    canonical : str
        The standard key name (e.g., "dx", "fs", "num_zplanes").
    aliases : tuple[str, ...]
        All known aliases for this parameter.
    dtype : type
        Expected Python type (float, int, str).
    unit : str, optional
        Physical unit if applicable (e.g., "micrometer", "Hz").
    default : Any
        Default value if parameter is not found in metadata.
    description : str
        Human-readable description of the parameter.
    label : str, optional
        Display label for GUI (e.g., "Frame Rate" for "fs").
    transforms : dict, optional
        Transform aliases: keys that hold a *converted* form of this value
        (e.g. ImageJ ``finterval`` = 1/``fs``). Each value is a
        ``(to_canonical, from_canonical)`` pair of callables — `get_param`
        applies ``to_canonical`` on read, writers apply ``from_canonical``
        to emit the alias. Distinct from `aliases`, which hold the value
        verbatim.
    """

    canonical: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    dtype: type = float
    unit: str | None = None
    default: Any = None
    description: str = ""
    label: str = ""
    transforms: dict[str, tuple] = field(default_factory=dict)


class VoxelSize(NamedTuple):
    """
    Voxel size in micrometers (dx, dy, dz).

    This class represents the physical size of a voxel in 3D space.
    All values are in micrometers.

    Attributes
    ----------
    dx : float
        Pixel size in X dimension (µm / px).
    dy : float
        Pixel size in Y dimension (µm / px).
    dz : float | None, optional
        Pixel/voxel size in Z dimension (µm / px).
        For LBM configurations, this must be supplied by the user.

    Examples
    --------
    >>> vs = VoxelSize(0.5, 0.5, 5.0)
    >>> vs.dx
    0.5
    >>> vs.dz
    5.0
    >>> tuple(vs)
    (0.5, 0.5, 5.0)
    """

    dx: float
    dy: float
    dz: float | None

    @property
    def pixel_resolution(self) -> tuple[float, float]:
        """Return (dx, dy) tuple for backward compatibility."""
        return (self.dx, self.dy)

    @property
    def voxel_size(self) -> tuple[float, float, float | None]:
        """Return (dx, dy, dz) tuple."""
        return (self.dx, self.dy, self.dz)

    def to_dict(self, include_aliases: bool = True) -> dict:
        """
        Convert to dictionary with optional aliases.

        Parameters
        ----------
        include_aliases : bool
            If True, includes all standard aliases (OME, ImageJ, legacy).

        Returns
        -------
        dict
            Dictionary with resolution values and aliases.
        """
        result = {
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "pixel_resolution": self.pixel_resolution,
            "voxel_size": self.voxel_size,
        }

        if include_aliases:
            # OME format
            result["PhysicalSizeX"] = self.dx
            result["PhysicalSizeY"] = self.dy
            result["PhysicalSizeZ"] = self.dz
            result["PhysicalSizeXUnit"] = "micrometer"
            result["PhysicalSizeYUnit"] = "micrometer"
            result["PhysicalSizeZUnit"] = "micrometer"

            # additional aliases
            result["z_step"] = self.dz  # backward compat

        return result


def _reciprocal(value: Any) -> float:
    """Convert between a rate and its interval (fs <-> finterval, px/µm <-> µm/px)."""
    v = float(value)
    if v == 0:
        raise ZeroDivisionError("cannot invert a zero-valued metadata field")
    return 1.0 / v


# metadata params registry
# dimensions: TZYX (4D), TYX (3D), or YX (2D)
METADATA_PARAMS: dict[str, MetadataParameter] = {
    # spatial resolution (micrometers per pixel)
    "dx": MetadataParameter(
        canonical="dx",
        aliases=(
            "Dx",
            "PhysicalSizeX",
            "pixelResolutionX",
            "pixel_size_x",
            "pixel_resolution_um",
            # scalar µm/px stamped by the MINI2P h5 converter; applies to x and y
            "pixel_size_um",
        ),
        dtype=float,
        unit="µm",
        default=1.0,
        description="Pixel size in X dimension (µm/pixel)",
        label="Pixel Size X",
        # TIFF/ImageJ XResolution is pixels-per-µm, the inverse of dx
        transforms={"XResolution": (_reciprocal, _reciprocal)},
    ),
    "dy": MetadataParameter(
        canonical="dy",
        aliases=(
            "Dy",
            "PhysicalSizeY",
            "pixelResolutionY",
            "pixel_size_y",
            # same scalar as dx's "pixel_size_um": square pixels assumed.
            # ALIAS_MAP keeps the last registration (dy), but get_param walks
            # each parameter's own alias tuple, so dx resolves it too.
            "pixel_size_um",
        ),
        dtype=float,
        unit="µm",
        default=1.0,
        description="Pixel size in Y dimension (µm/pixel)",
        label="Pixel Size Y",
        transforms={"YResolution": (_reciprocal, _reciprocal)},
    ),
    "dz": MetadataParameter(
        canonical="dz",
        aliases=(
            "Dz",
            "PhysicalSizeZ",
            "z_step",
            "z_step_um",
            "spacing",
            "pixelResolutionZ",
            "ZResolution",
            "axial_step",
            "axial_step_um",
        ),
        dtype=float,
        unit="µm",
        default=None,
        description="Voxel size in Z dimension (µm/z-step). Must be user-supplied for LBM.",
        label="Z Step",
    ),
    # temporal
    "fs": MetadataParameter(
        canonical="fs",
        aliases=(
            "frame_rate",
            "framerate",
            "fr",
            "sampling_frequency",
            "frameRate",
            "scanFrameRate",
            "fps",
            # MINI2P h5 converter group attr (Hz)
            "frame_rate_hz",
        ),
        dtype=float,
        unit="Hz",
        default=None,
        description="Frame rate / sampling frequency (Hz)",
        label="Frame Rate",
        # ImageJ/Fiji store the per-frame interval (seconds); fs = 1/finterval
        transforms={"finterval": (_reciprocal, _reciprocal)},
    ),
    # volumes per second (for volumetric imaging like IsoView)
    "vps": MetadataParameter(
        canonical="vps",
        aliases=("volumes_per_second",),
        dtype=float,
        unit="Hz",
        default=None,
        description="Volumetric acquisition rate (volumes per second)",
        label="Volume Rate",
    ),
    # ImageJ frame interval (seconds between frames, inverse of fs)
    "finterval": MetadataParameter(
        canonical="finterval",
        aliases=(
            "frame_interval",
            "FrameInterval",
            "dt",
            "time_interval",
            # AOD mesc2h5 converter dataset attr (seconds per frame);
            # fs is derived as 1/frame_period by the fs<->finterval transform
            "frame_period",
            "TimeIncrement",
        ),
        dtype=float,
        unit="s",
        default=None,
        description="Frame interval in seconds (1/fs). Used by ImageJ/Fiji and OME.",
        label="Frame Interval",
        transforms={"fs": (_reciprocal, _reciprocal)},
    ),
    # image dimensions (pixels)
    "Lx": MetadataParameter(
        canonical="Lx",
        aliases=(
            "lx",
            "LX",
            "width",
            "nx",
            "size_x",
            "SizeX",
            "image_width",
            "fov_x",
            "num_px_x",
            "page_width",
            "ImageWidth",
        ),
        dtype=int,
        unit="px",
        default=None,
        description="Image width in pixels",
    ),
    "Ly": MetadataParameter(
        canonical="Ly",
        aliases=(
            "ly",
            "LY",
            "height",
            "ny",
            "size_y",
            "SizeY",
            "image_height",
            "fov_y",
            "num_px_y",
            "page_height",
            "ImageLength",
        ),
        dtype=int,
        unit="px",
        default=None,
        description="Image height in pixels",
    ),
    # frame/plane/channel counts
    # note: in suite2p ops.npy, "nframes" means timepoints (post-registration), not per-slice frames
    "frames_per_file": MetadataParameter(
        canonical="frames_per_file",
        aliases=(
            "frames_per_folder",   # suite2p ops.npy
            "nframes_per_file",
            "frames_per_tiff",
        ),
        dtype=list,
        default=None,
        description=(
            "Timepoints contributed by each source file, in the order of "
            "file_paths. Marks trial/acquisition boundaries along T for "
            "pipelines that treat each file as a separate trial."
        ),
        label="Frames per file",
    ),
    "file_paths": MetadataParameter(
        canonical="file_paths",
        aliases=(
            "filenames",
            "filelist",
            "tiff_list",
        ),
        dtype=list,
        default=None,
        description="Source files backing the array, ordered along T",
        label="Source files",
    ),
    "num_timepoints": MetadataParameter(
        canonical="num_timepoints",
        aliases=(
            "nframes",        # suite2p ops.npy compatibility
            "num_frames",     # legacy alias
            "n_frames",
            "frames",
            "T",
            "nt",
            "timepoints",
            "n_timepoints",
            "SizeT",
        ),
        dtype=int,
        default=None,
        description="Number of timepoints (T dimension) in the dataset",
        label="Timepoints",
    ),
    "num_zplanes": MetadataParameter(
        canonical="num_zplanes",
        aliases=(
            "num_planes",
            "nplanes",
            "n_planes",
            "planes",
            "Z",
            "nz",
            "num_z",
            "numPlanes",
            "zplanes",
            "slices",
            "SizeZ",
        ),
        dtype=int,
        default=1,
        description="Number of z-planes",
        label="Num Z-Planes",
    ),
    "nchannels": MetadataParameter(
        canonical="nchannels",
        aliases=(
            "num_channels",
            "n_channels",
            "channels",
            "C",
            "nc",
            "numChannels",
        ),
        dtype=int,
        default=1,
        description="Number of channels (typically 1 for calcium imaging)",
    ),
    # data type
    "dtype": MetadataParameter(
        canonical="dtype",
        aliases=("data_type", "pixel_type", "datatype"),
        dtype=str,
        default="int16",
        description="Data type of pixel values",
    ),
    # total number of elements
    "size": MetadataParameter(
        canonical="size",
        aliases=("num_elements", "total_elements"),
        dtype=int,
        default=None,
        description="Total number of elements in the array (product of dimensions)",
    ),
    # array shape tuple
    "shape": MetadataParameter(
        canonical="shape",
        aliases=("array_shape", "data_shape"),
        dtype=tuple,
        default=None,
        description="Array shape as tuple (T, Z, Y, X) or (T, Y, X) or (Y, X)",
    ),
    # stack detection (ScanImage-derived)
    "stack_type": MetadataParameter(
        canonical="stack_type",
        aliases=("stackType",),
        dtype=str,
        default="single_plane",
        description="Stack type: lbm, piezo, or single_plane",
    ),
    "lbm_stack": MetadataParameter(
        canonical="lbm_stack",
        aliases=("is_lbm", "lbmStack"),
        dtype=bool,
        default=False,
        description="True if Light Beads Microscopy stack",
    ),
    "piezo_stack": MetadataParameter(
        canonical="piezo_stack",
        aliases=("is_piezo", "piezoStack"),
        dtype=bool,
        default=False,
        description="True if piezo-driven z-stack",
    ),
    "num_color_channels": MetadataParameter(
        canonical="num_color_channels",
        aliases=("color_channels", "ncolors", "num_colors", "n_channel", "SizeC"),
        dtype=int,
        default=1,
        description="Number of color channels (1 or 2)",
    ),
    # ROI/FOV parameters
    "num_mrois": MetadataParameter(
        canonical="num_mrois",
        aliases=("num_rois", "scanimage_multirois", "numROIs", "nrois", "n_rois"),
        dtype=int,
        default=1,
        description="Number of mROIs (ScanImage multi-ROI scan regions)",
    ),
    "roi": MetadataParameter(
        canonical="roi",
        aliases=("roi_size", "roi_px"),
        dtype=tuple,
        unit="px",
        default=None,
        description="ROI dimensions as (width, height) in pixels",
    ),
    "fov": MetadataParameter(
        canonical="fov",
        aliases=("fov_px",),
        dtype=tuple,
        unit="px",
        default=None,
        description="Field of view as (x, y) in pixels (tiled)",
    ),
    "fov_um": MetadataParameter(
        canonical="fov_um",
        aliases=(),
        dtype=tuple,
        unit="µm",
        default=None,
        description="Field of view as (x, y) in µm (tiled)",
    ),
}


def _build_alias_map() -> dict[str, str]:
    """Build reverse lookup: alias (lowercase) -> canonical name."""
    alias_map = {}
    for param in METADATA_PARAMS.values():
        alias_map[param.canonical.lower()] = param.canonical
        for alias in param.aliases:
            alias_map[alias.lower()] = param.canonical
    return alias_map


ALIAS_MAP: dict[str, str] = _build_alias_map()


def get_canonical_name(name: str) -> str | None:
    """
    Get the canonical parameter name for an alias.

    Parameters
    ----------
    name : str
        Parameter name or alias.

    Returns
    -------
    str or None
        Canonical name, or None if not a registered parameter.
    """
    return ALIAS_MAP.get(name.lower())


# core imaging metadata keys - always shown in metadata viewers/editors
# these are the essential parameters for calcium imaging data
IMAGING_METADATA_KEYS: tuple[str, ...] = (
    "fs",
    # shown with fs so its aliases (dt, frame_interval, ...) are absorbed
    # into the imaging section instead of contradicting it under "Other"
    "finterval",
    "dx",
    "dy",
    "dz",
    "Lx",
    "Ly",
    "num_zplanes",
    "num_color_channels",
    "num_mrois",
    "num_timepoints",
    "dtype",
)


# fields stripped from tiff/h5/zarr metadata before stamping. these
# belong only in suite2p layouts (ops.npy alongside data.bin) — embedding
# them in non-suite2p outputs bloats files (regPC alone can be 500+ MB
# after JSON expansion) and clutters readers like Fiji that load the
# entire metadata blob into memory.

_SUITE2P_REGISTRATION_INTERNALS = (
    "regPC", "tPC", "regDX",
    "yblock", "xblock", "NRsm",
)

_SUITE2P_SUMMARY_IMAGES = (
    "meanImg", "meanImgE", "meanImg_chan2", "meanImg_crop",
    "Vmap", "Vcorr", "Vsplit", "Vmax",
    "max_proj",
    "refImg", "refImg1", "refAndMasks",
)

_PER_FRAME_VECTORS = (
    "xoff", "yoff", "corrXY",
    "xoff1", "yoff1", "corrXY1",
    "badframes", "badframes0",
    "ihop", "plane_times",
)

# plane_shifts / plane_shifts_params are intentionally NOT denylisted:
# register_z writes them so viewers can align planes at render time
# (see arrays/_registration.py), so they must survive export.
_MBO_ADDITIONS = (
    "processing_history", "_metadata_provenance",
    "roi_mode",
)

_SUITE2P_GEOMETRY = (
    "Ly", "Lx", "nframes", "nplanes", "nchannels",
    "num_rois", "aspect", "tau",
    "functional_chan", "align_by_chan",
)

_SUITE2P_PIPELINE_SETTINGS = (
    "do_registration", "keep_movie_raw", "two_step_registration",
    "nimg_init", "multiplane_parallel",
    "nbinned", "batch_size",
    "diameter", "cell_diameter", "spatial_scale", "spatscale_pix",
    "roidetect", "spikedetect", "neuropil_extract",
    "denoise", "anatomical_only",
    "sparse_mode", "connected",
    "threshold_scaling", "max_overlap", "max_iterations",
    "high_pass", "smooth_sigma", "smooth_sigma_time",
    "nonrigid", "block_size", "snr_thresh", "maxregshift",
    "use_builtin_classifier", "classifier_path",
    "preclassify", "chan2_thres",
    "lam_percentile", "allow_overlap",
    "inner_neuropil_radius", "min_neuropil_pixels",
    "neucoeff",
    "soma_crop", "win_baseline", "sig_baseline", "prctile_baseline",
    "data_path", "save_path", "save_path0", "save_folder",
    "fast_disk", "ops_path", "input_format",
    "save_NWB", "save_mat",
    "first_tiffs", "frames_include",
    "h5py", "h5py_key",
    "delete_bin", "combined", "report_time",
    "do_bidiphase", "bidiphase",
    "1Preg", "spatial_hp", "pre_smooth", "spatial_taper",
)

EXPORT_DENYLIST: frozenset[str] = frozenset(
    _SUITE2P_REGISTRATION_INTERNALS
    + _SUITE2P_SUMMARY_IMAGES
    + _PER_FRAME_VECTORS
    + _MBO_ADDITIONS
    + _SUITE2P_GEOMETRY
    + _SUITE2P_PIPELINE_SETTINGS
)


# backstop for ops keys the denylist doesn't know by name: anything with
# more elements than this is image/vector payload, not metadata. keys that
# legitimately carry long arrays must be allowlisted.
_MAX_EXPORT_ELEMENTS = 8192

_SIZE_GUARD_ALLOW = frozenset({
    "plane_shifts", "scanphase", "frames_per_file", "roi_groups",
    "timepoint_selection", "file_paths", "ome",
})


def _element_count(value) -> int:
    """Total leaf elements in a scalar / ndarray / (nested) list or tuple.

    Iterative so pathologically deep nesting can't hit the recursion limit.
    """
    total = 0
    stack = [value]
    while stack:
        v = stack.pop()
        size = getattr(v, "size", None)  # ndarray and numpy scalars
        if isinstance(size, int):
            total += size
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
        else:
            total += 1
    return total


def strip_for_export(md: dict) -> dict:
    """drop fields that should not be embedded in tiff/h5/zarr metadata.

    suite2p ops fields (registration internals, summary images, per-frame
    vectors, pipeline settings) and mbo-internal additions are kept only
    in suite2p layouts (ops.npy alongside data.bin). everything else —
    OME, ImageJ aliases, voxel size, frame rate, user description, etc. —
    passes through untouched, except values whose element count exceeds
    ``_MAX_EXPORT_ELEMENTS`` (embedded image planes and per-frame vectors
    under names the denylist doesn't know inflate root attrs to MBs).
    """
    out = {}
    for k, v in md.items():
        if k in EXPORT_DENYLIST:
            continue
        if k not in _SIZE_GUARD_ALLOW:
            n = _element_count(v)
            if n > _MAX_EXPORT_ELEMENTS:
                _logger().warning(
                    "dropping oversized metadata %r (%d elements) from export",
                    k, n,
                )
                continue
        out[k] = v
    return out



# ops fields that must reach suite2p as arrays: it and its plotting code
# index them with .shape / .size, never as sequences
# sdmov / refImg0 are not on the export denylist (they never reach a tiff
# or zarr attr) but suite2p indexes them the same way
_OPS_ARRAY_KEYS: frozenset[str] = frozenset(
    _SUITE2P_SUMMARY_IMAGES
    + _PER_FRAME_VECTORS
    + _SUITE2P_REGISTRATION_INTERNALS
    + ("sdmov", "refImg0")
)


def normalize_ops_arrays(ops: dict) -> dict:
    """Bring an ops dict's image and per-frame fields back to ndarrays.

    Metadata that has been through JSON — a zarr attr, a tiff description,
    an h5 attribute — comes back as (nested) lists. suite2p and
    lbm_suite2p_python index those fields with ``.shape``, so a list-valued
    ``meanImgE`` surfaces downstream as "'list' object has no attribute
    'shape'" and every figure that touches it fails.

    Empty ones are dropped rather than saved as empty arrays: the consumers
    test for the key's absence, and lbm_suite2p_python recomputes
    ``meanImgE`` from ``meanImg`` when it is missing.
    """
    import numpy as np

    out = {}
    for k, v in ops.items():
        if k not in _OPS_ARRAY_KEYS:
            out[k] = v
            continue
        if isinstance(v, (list, tuple)):
            if not len(v):
                continue
            try:
                v = np.asarray(v)
            except Exception:
                _logger().warning("ops field %r is not array-shaped; kept as is", k)
                out[k] = v
                continue
            # a list of Python floats reads back as float64; suite2p writes
            # these fields as float32 and callers index them by dtype
            if v.dtype == np.float64:
                v = v.astype(np.float32)
        if isinstance(v, np.ndarray) and v.size == 0:
            continue
        out[k] = v
    return out


def repair_ops_file(path) -> bool:
    """Normalize one ``ops.npy`` in place; True when it had to be rewritten.

    Run dirs written before the ops fields were normalized hold their
    images as JSON lists, and nothing rewrites ops.npy on a detection-only
    re-run - it is copied as is - so the plots keep failing until the file
    itself is repaired.
    """
    import numpy as np

    path = Path(path)
    if not path.exists():
        return False
    try:
        ops = np.load(path, allow_pickle=True).item()
    except Exception as error:  # noqa: BLE001 - a bad ops.npy is not fatal here
        _logger().warning("could not read %s to repair it: %s", path, error)
        return False
    if not isinstance(ops, dict):
        return False
    bad = [
        k for k, v in ops.items()
        if k in _OPS_ARRAY_KEYS and isinstance(v, (list, tuple))
    ]
    if not bad:
        return False
    try:
        np.save(path, normalize_ops_arrays(ops))
    except Exception as error:  # noqa: BLE001 - read-only / locked file
        _logger().warning("could not rewrite %s: %s", path, error)
        return False
    _logger().info("repaired %s: %s were lists, not arrays", path, ", ".join(bad))
    return True


def repair_ops_tree(path) -> int:
    """Repair every ``ops.npy`` in a run dir and its plane children.

    Returns the number of files rewritten. Accepts a dir, an ops.npy, or a
    file inside a plane dir, so callers can hand it whatever the user
    pointed the run at.
    """
    path = Path(path)
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        return 0
    # the dir itself, its plane children, and one more level for the
    # suite2p/planeNN layout
    targets = [path / "ops.npy"]
    for pattern in ("*/ops.npy", "*/*/ops.npy"):
        targets += sorted(path.glob(pattern))
    seen, fixed = set(), 0
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        fixed += bool(repair_ops_file(target))
    return fixed


# keys whose values are filesystem paths recorded by a pipeline run —
# provenance that goes stale the moment the output directory is copied to
# another machine or share (db.npy data_path, ops raw_file, ...).
PROVENANCE_PATH_KEYS: tuple[str, ...] = (
    "data_path", "save_path", "save_path0", "fast_disk", "ops_path",
    "raw_file", "reg_file", "raw_source", "file_paths", "file_list",
)


def _path_basename(p) -> str:
    """Basename of a path string regardless of which OS recorded it."""
    return str(p).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _rebase_dirs(anchor: Path) -> list[Path]:
    """Small fixed candidate set: the opened dir, three ancestors, their raw/.

    Three ancestors because run layouts nest as <share>/<user>/<group>/<run>/
    <plane> with raw/ beside <group> (e.g. eunji/raw vs eunji/demo/s2p_*/
    zplane01_*), so a plane-dir anchor needs to climb three levels.
    """
    dirs: list[Path] = []
    for d in (
        anchor,
        anchor.parent,
        anchor.parent.parent,
        anchor.parent.parent.parent,
    ):
        for sub in (d, d / "raw"):
            if sub not in dirs:
                dirs.append(sub)
    return dirs


def _rebase_one(dead, dirs: list[Path]) -> Path | None:
    name = _path_basename(dead)
    if not name:
        return None
    for d in dirs:
        try:
            cand = d / name
            if cand.exists():
                return cand
        except OSError:
            continue
    return None


def rebase_provenance_paths(md: dict, anchor: Path | str, logger=None) -> dict:
    """Repair or drop stale recorded paths against the actually-opened tree.

    For each :data:`PROVENANCE_PATH_KEYS` entry in ``md``: paths that exist
    are kept; dead paths are rebased by basename against ``anchor``, its
    three ancestors, and their ``raw/`` subdirectories (the layout lab
    shares use);
    unresolvable entries are dropped so run outputs never re-embed another
    machine's paths. Empty/None values pass through untouched. Never raises.

    Returns a new dict; ``md`` is not modified.
    """
    lg = logger or _logger()
    anchor = Path(anchor)
    dirs = _rebase_dirs(anchor)
    out = dict(md)
    for key in PROVENANCE_PATH_KEYS:
        if key not in out:
            continue
        val = out[key]
        # type-check before any equality: ndarray values make `==` elementwise
        if val is None:
            continue
        if isinstance(val, (str, Path)):
            if not str(val):
                continue
            entries, was_scalar = [val], True
        elif isinstance(val, (list, tuple)):
            if not val or not all(isinstance(v, (str, Path)) for v in val):
                continue  # empty, or unexpected element types: leave untouched
            entries, was_scalar = list(val), False
        else:
            continue  # unexpected shape (ndarray, dict, ...): leave untouched
        kept: list[str] = []
        for entry in entries:
            try:
                alive = Path(entry).exists()
            except OSError:
                alive = False
            if alive:
                kept.append(str(entry))
                continue
            found = _rebase_one(entry, dirs)
            if found is not None:
                lg.info(f"provenance rebase: {key} {str(entry)!r} -> {found}")
                kept.append(str(found))
            else:
                lg.info(
                    f"provenance: dropping stale {key}={str(entry)!r} "
                    f"(path does not exist on this machine)"
                )
        if not kept:
            del out[key]
        elif was_scalar:
            out[key] = kept[0]
        else:
            out[key] = kept
    return out
