"""
parameter access and normalization utilities.

provides functions to get/set metadata parameters using canonical names
and their aliases, with type conversion and defaults.
"""
from __future__ import annotations

import logging
from typing import Any

from .base import METADATA_PARAMS, ALIAS_MAP, VoxelSize
import contextlib


def _rate_precedence() -> tuple[dict[str, tuple[int, str]], int]:
    """Build lowercase key -> (rank, kind) for frame-rate resolution.

    kind is "interval" (seconds per frame) or "rate" (Hz). Canonical ``fs``
    ranks first — it is the key user overrides (metadata editor, suite2p
    custom_metadata merges) write — then ``frame_rate``, then ``finterval``.
    Canonical keys outrank the OME time scale, which outranks the remaining
    aliases in registry order (``fps`` last).
    """
    order: dict[str, tuple[int, str]] = {
        "fs": (0, "rate"),
        "frame_rate": (1, "rate"),
        "finterval": (2, "interval"),
    }
    ome_rank = 3
    rank = 4
    for alias in METADATA_PARAMS["finterval"].aliases:
        key = alias.lower()
        if key not in order:
            order[key] = (rank, "interval")
            rank += 1
    fs_aliases = [a for a in METADATA_PARAMS["fs"].aliases if a.lower() != "fps"]
    fs_aliases.append("fps")
    for alias in fs_aliases:
        key = alias.lower()
        if key not in order:
            order[key] = (rank, "rate")
            rank += 1
    return order, ome_rank


def _exact_rate_spellings() -> frozenset[str]:
    """Exact registered spellings of every rate key (case-sensitive)."""
    return frozenset(
        {
            "fs",
            "finterval",
            *METADATA_PARAMS["fs"].aliases,
            *METADATA_PARAMS["finterval"].aliases,
        }
    )


_RATE_PRECEDENCE, _OME_RATE_RANK = _rate_precedence()
_EXACT_RATE_SPELLINGS = _exact_rate_spellings()

# ranks of the canonical timing keys; a None stored under any of these is an
# explicit "rate invalid" sentinel (non-contiguous selections) that must not
# be bypassed by falling through to lower-precedence aliases
_CANONICAL_RATE_MAX_RANK = 2

# stale-alias divergence warnings, deduped per unique signature. hot paths
# (the masknmf panel calls get_param(md, "fs") per rendered frame) would
# otherwise emit the identical warning ~60x/s; repeats log at DEBUG.
_STALE_WARNED: set = set()
_STALE_WARNED_MAX = 256


def scale_frame_rate(metadata: dict, factor: float) -> dict:
    """A copy of ``metadata`` retimed for ``factor`` source frames per frame.

    Temporal binning (``FrameAveragedView``) divides the frame rate. Every
    *registered* rate spelling is divided and every interval spelling
    multiplied, rather than a hand-picked few: a single alias left claiming
    the original rate makes ``resolve_effective_rate`` warn about a stale
    alias, and anything reading ``fps`` or ``dt`` straight out of the dict
    would silently get the pre-binning number.

    Nested OME ``multiscales`` time scales are left alone; those describe the
    file on disk, not this view.
    """
    factor = float(factor)
    if factor == 1.0:
        return dict(metadata)
    out = dict(metadata)
    for key, value in metadata.items():
        entry = _RATE_PRECEDENCE.get(str(key).lower())
        kind = entry[1] if entry else ("interval" if key == "_ome_time_scale" else None)
        if kind is None:
            continue
        numeric = _rate_value(value)
        if numeric is None:
            continue
        out[key] = numeric * factor if kind == "interval" else numeric / factor
    return out


def _rate_value(val: Any) -> float | None:
    """Numeric, non-zero, finite value or None."""
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f == 0 or f != f:
        return None
    return f


def _ome_time_scale(metadata: dict) -> tuple[float, str] | None:
    """Level-0 time-axis scale (seconds per frame) from OME metadata.

    Checks ``metadata["ome"]["multiscales"]``, a top-level ``multiscales``,
    then the ``_ome_time_scale`` key stamped by the zarr reader. A scale of
    1.0 is the writer default and treated as unset.
    """
    sources = []
    ome = metadata.get("ome")
    if isinstance(ome, dict):
        sources.append((ome.get("multiscales"), "ome.multiscales"))
    sources.append((metadata.get("multiscales"), "multiscales"))
    for multiscales, label in sources:
        try:
            entry = multiscales[0]
            axes = entry["axes"]
            idx = next(
                i for i, ax in enumerate(axes)
                if isinstance(ax, dict) and ax.get("type") == "time"
            )
            scale = float(
                entry["datasets"][0]["coordinateTransformations"][0]["scale"][idx]
            )
        except (TypeError, KeyError, IndexError, ValueError, StopIteration):
            continue
        if scale > 0 and scale != 1.0:
            return scale, label
    raw = _rate_value(metadata.get("_ome_time_scale"))
    if raw is not None and raw > 0 and raw != 1.0:
        return raw, "_ome_time_scale"
    return None


def resolve_effective_rate(
    metadata: dict | None,
) -> tuple[float | None, float | None, str]:
    """
    Resolve the effective frame rate from a metadata dict.

    Older mbo stores stamp every registered rate alias at ingest but only
    update the canonical keys on decimating writes, so a dict can carry a
    correct ``fs``/``finterval`` alongside stale ``framerate``/``dt``/...
    values. This resolves one authoritative rate using a fixed precedence:

    1. canonical ``fs``, then ``frame_rate`` (case-insensitive) — ``fs``
       ranks first because it is the key user overrides (metadata editor,
       suite2p custom_metadata merges) write
    2. canonical ``finterval`` (case-insensitive) -> ``fs = 1/v``
    3. the OME per-level time scale (``ome.multiscales``, top-level
       ``multiscales``, or ``_ome_time_scale``; 1.0 is treated as unset)
    4. the remaining ``finterval`` then ``fs`` aliases (``fps`` last),
       case-insensitive, in registry order

    At equal precedence, the exactly-registered spelling outranks a case
    variant (``{"FS": ..., "fs": ...}`` resolves from ``fs``).

    A canonical key stored as None (stamped by non-contiguous selections)
    blocks fallback to (3) and (4) — the rate is genuinely undefined.

    Parameters
    ----------
    metadata : dict or None
        Metadata dictionary to search.

    Returns
    -------
    tuple[float | None, float | None, str]
        ``(fs_hz, finterval_s, source_key)``; ``(None, None, "")`` when no
        usable value exists. Logs one warning (per unique divergence
        signature; repeats at DEBUG) when the resolved rate disagrees with
        any other candidate by more than 0.1% relative.
    """
    if not isinstance(metadata, dict) or not metadata:
        return None, None, ""

    # (rank, key, fs, finterval, raw value)
    candidates: list[tuple[int, str, float, float, Any]] = []
    canonical_nulled = False
    for key, val in metadata.items():
        if not isinstance(key, str):
            continue
        low = key.lower()
        if low == "_ome_time_scale":
            continue
        info = _RATE_PRECEDENCE.get(low)
        if info is None:
            continue
        rank, kind = info
        value = _rate_value(val)
        if value is None:
            if val is None and rank <= _CANONICAL_RATE_MAX_RANK:
                canonical_nulled = True
            continue
        if kind == "interval":
            candidates.append((rank, key, 1.0 / value, value, val))
        else:
            candidates.append((rank, key, value, 1.0 / value, val))

    ome = _ome_time_scale(metadata)
    if ome is not None:
        scale, label = ome
        candidates.append((_OME_RATE_RANK, label, 1.0 / scale, scale, scale))

    if not candidates:
        return None, None, ""

    # at equal rank the exactly-registered spelling outranks a case variant
    # ({"FS": 10, "fs": 20} must resolve from "fs", not dict order)
    candidates.sort(
        key=lambda c: (c[0], 0 if c[1] in _EXACT_RATE_SPELLINGS else 1)
    )
    best_rank, best_key, best_fs, best_finterval, _ = candidates[0]
    if canonical_nulled and best_rank > _CANONICAL_RATE_MAX_RANK:
        return None, None, ""

    stale = [
        (key, raw)
        for _rank, key, fs, _fint, raw in candidates[1:]
        if abs(fs - best_fs) > 1e-3 * abs(best_fs)
    ]
    if stale:
        listing = ", ".join(f"{k}={v!r}" for k, v in stale)
        message = (
            f"metadata carries stale frame-rate aliases: resolved "
            f"fs={best_fs:.6g} Hz from {best_key!r}, but {listing} disagree"
        )
        signature = (
            best_key,
            round(best_fs, 6),
            frozenset((k, round(float(v), 6)) for k, v in stale),
        )
        log = logging.getLogger("mbo_utilities")
        if signature in _STALE_WARNED:
            log.debug(message)
        else:
            if len(_STALE_WARNED) >= _STALE_WARNED_MAX:
                _STALE_WARNED.clear()
            _STALE_WARNED.add(signature)
            log.warning(message)
    return best_fs, best_finterval, best_key


def get_param(
    metadata: dict | None,
    name: str,
    default: Any = None,
    *,
    override: Any = None,
    shape: tuple | None = None,
    _apply_transforms: bool = True,
) -> Any:
    """
    Get a metadata parameter, checking all known aliases.

    This provides a unified way to access metadata values without needing to
    know which alias was used to store it. The function checks the canonical
    name first, then all registered aliases.

    Parameters
    ----------
    metadata : dict or None
        Metadata dictionary to search.
    name : str
        Canonical parameter name (e.g., "dx", "fs", "num_planes").
        Case-insensitive; will be resolved to canonical form.
    default : Any, optional
        Override default value. If None, uses the parameter's registered default.
    override : Any, optional
        If provided, returns this value directly (for user-specified overrides).
    shape : tuple, optional
        Array shape for fallback dimension extraction (Lx, Ly from shape[-1], shape[-2]).

    Returns
    -------
    Any
        Parameter value converted to the correct dtype, or default if not found.

    Examples
    --------
    >>> meta = {"umPerPixX": 0.5, "frame_rate": 30.0}
    >>> get_param(meta, "dx")
    0.5
    >>> get_param(meta, "fs")
    30.0
    >>> get_param(meta, "num_planes")  # uses default
    1
    >>> get_param(meta, "dx", override=0.3)  # override wins
    0.3
    """
    # if override provided, use it directly
    if override is not None:
        return override

    # resolve canonical name (case-insensitive lookup)
    canonical = ALIAS_MAP.get(name.lower())
    if canonical is None:
        # not a registered parameter - just do simple dict lookup
        if metadata is not None and name in metadata:
            return metadata[name]
        return default

    param = METADATA_PARAMS[canonical]

    # determine final default
    final_default = default if default is not None else param.default

    if metadata is None:
        # try shape fallback for dimensions
        if shape is not None:
            if canonical == "Lx" and len(shape) >= 1:
                return int(shape[-1])
            if canonical == "Ly" and len(shape) >= 2:
                return int(shape[-2])
        return final_default

    # timing params resolve through the effective-rate precedence so stale
    # aliases stamped by older writers can never shadow the canonical value
    if canonical in ("fs", "finterval") and _apply_transforms:
        fs_hz, finterval_s, _src = resolve_effective_rate(metadata)
        resolved = fs_hz if canonical == "fs" else finterval_s
        if resolved is not None:
            with contextlib.suppress(TypeError, ValueError):
                return param.dtype(resolved)
        return final_default

    # check canonical name first, then all aliases
    keys_to_check = (param.canonical, *param.aliases)
    for key in keys_to_check:
        val = metadata.get(key)
        if val is not None:
            try:
                if param.dtype == tuple:
                    # handle tuple specially
                    if isinstance(val, (list, tuple)):
                        return tuple(val)
                    return val
                return param.dtype(val)
            except (TypeError, ValueError):
                continue

    # special handling for pixel_resolution tuple -> dx/dy
    if canonical in ("dx", "dy"):
        pixel_res = metadata.get("pixel_resolution")
        if pixel_res is not None:
            if isinstance(pixel_res, (list, tuple)) and len(pixel_res) >= 2:
                try:
                    idx = 0 if canonical == "dx" else 1
                    return float(pixel_res[idx])
                except (TypeError, ValueError, IndexError):
                    pass
            elif isinstance(pixel_res, (int, float)):
                return float(pixel_res)

    # transform aliases: derive from a converted form (e.g. fs from finterval).
    # _apply_transforms=False on the nested lookup prevents A->B->A recursion.
    if _apply_transforms and param.transforms:
        for src_key, (to_canonical, _from_canonical) in param.transforms.items():
            src_val = get_param(metadata, src_key, _apply_transforms=False)
            if src_val is not None:
                try:
                    return param.dtype(to_canonical(src_val))
                except (TypeError, ValueError, ZeroDivisionError):
                    continue

    # fallback: extract Lx/Ly from shape
    if shape is not None:
        if canonical == "Lx" and len(shape) >= 1:
            return int(shape[-1])
        if canonical == "Ly" and len(shape) >= 2:
            return int(shape[-2])

    # special: try to get dtype from shape metadata
    if canonical == "dtype":
        arr_dtype = metadata.get("dtype")
        if arr_dtype is not None:
            return str(arr_dtype)

    return final_default


def get_voxel_size(
    metadata: dict | None = None,
    dx: float | None = None,
    dy: float | None = None,
    dz: float | None = None,
) -> VoxelSize:
    """
    Extract voxel size from metadata with optional user overrides.

    Resolution values are resolved in priority order:
    1. User-provided parameter (highest priority)
    2. Canonical keys (dx, dy, dz)
    3. pixel_resolution tuple
    4. Legacy keys (umPerPixX, umPerPixY, umPerPixZ)
    5. OME keys (PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ)
    6. ScanImage SI keys
    7. Default: 1.0 micrometers

    Parameters
    ----------
    metadata : dict, optional
        Metadata dictionary to extract resolution from.
    dx : float, optional
        Override X resolution (micrometers per pixel).
    dy : float, optional
        Override Y resolution (micrometers per pixel).
    dz : float, optional
        Override Z resolution (micrometers per z-step).

    Returns
    -------
    VoxelSize
        Named tuple with (dx, dy, dz) in micrometers.

    Examples
    --------
    >>> meta = {"pixel_resolution": (0.5, 0.5)}
    >>> vs = get_voxel_size(meta, dz=5.0)
    >>> vs.dz
    5.0

    >>> vs = get_voxel_size({"dx": 0.3, "dy": 0.3, "dz": 2.0})
    >>> vs.pixel_resolution
    (0.3, 0.3)
    """
    if metadata is None:
        metadata = {}

    # helper to get first non-None value from a list of keys
    def _get_first(keys: list[str], default: float = 1.0) -> float:
        for key in keys:
            val = metadata.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return default

    # extract pixel_resolution tuple if present
    pixel_res = metadata.get("pixel_resolution")
    px_x, px_y = None, None
    if pixel_res is not None:
        if isinstance(pixel_res, (list, tuple)) and len(pixel_res) >= 2:
            try:
                px_x = float(pixel_res[0])
                px_y = float(pixel_res[1])
            except (TypeError, ValueError):
                pass
        elif isinstance(pixel_res, (int, float)):
            # single value: use for both X and Y
            px_x = px_y = float(pixel_res)

    # try to extract dz from ScanImage nested structure (NOT for LBM - user must supply)
    si_dz = None
    is_lbm = metadata.get("lbm_stack", False) or metadata.get("stack_type") == "lbm"
    if not is_lbm:
        si = metadata.get("si", {})
        if isinstance(si, dict):
            h_stack = si.get("hStackManager", {})
            if isinstance(h_stack, dict):
                si_dz = h_stack.get("actualStackZStepSize")
                if si_dz is None:
                    si_dz = h_stack.get("stackZStepSize")

    # resolve dx
    resolved_dx = dx
    if resolved_dx is None:
        resolved_dx = _get_first(["dx", "PhysicalSizeX"], default=None)
    if resolved_dx is None and px_x is not None:
        resolved_dx = px_x
    if resolved_dx is None:
        resolved_dx = 1.0

    # resolve dy
    resolved_dy = dy
    if resolved_dy is None:
        resolved_dy = _get_first(["dy", "PhysicalSizeY"], default=None)
    if resolved_dy is None and px_y is not None:
        resolved_dy = px_y
    if resolved_dy is None:
        resolved_dy = 1.0

    # resolve dz (more aliases for z-step)
    resolved_dz = dz
    if resolved_dz is None:
        resolved_dz = _get_first(
            ["dz", "z_step", "PhysicalSizeZ", "spacing"],
            default=None,
        )
    if resolved_dz is None and si_dz is not None:
        with contextlib.suppress(TypeError, ValueError):
            resolved_dz = float(si_dz)

    # for LBM stacks, dz must be user-supplied - no default
    # for non-LBM, default to 1.0 if not found
    if resolved_dz is None and not is_lbm:
        resolved_dz = 1.0

    return VoxelSize(dx=resolved_dx, dy=resolved_dy, dz=resolved_dz)


def normalize_resolution(
    metadata: dict,
    dx: float | None = None,
    dy: float | None = None,
    dz: float | None = None,
) -> dict:
    """
    Normalize resolution metadata by adding all standard aliases.

    This function ensures that resolution information is available under
    all commonly-used keys for different tools and formats:

    - Canonical: dx, dy, dz
    - Legacy: pixel_resolution (tuple), z_step, umPerPixX/Y/Z
    - OME: PhysicalSizeX/Y/Z with units
    - Convenience: voxel_size (3-tuple)

    Parameters
    ----------
    metadata : dict
        Metadata dictionary to normalize. Modified in-place AND returned.
    dx : float, optional
        Override X resolution (micrometers per pixel).
    dy : float, optional
        Override Y resolution (micrometers per pixel).
    dz : float, optional
        Override Z resolution (micrometers per z-step).

    Returns
    -------
    dict
        The same metadata dict with resolution aliases added.

    Examples
    --------
    >>> meta = {"pixel_resolution": (0.5, 0.5)}
    >>> normalize_resolution(meta, dz=5.0)
    >>> meta["dz"]
    5.0
    >>> meta["voxel_size"]
    (0.5, 0.5, 5.0)
    >>> meta["PhysicalSizeZ"]
    5.0
    """
    vs = get_voxel_size(metadata, dx=dx, dy=dy, dz=dz)
    metadata.update(vs.to_dict(include_aliases=True))
    return metadata


def normalize_metadata(
    metadata: dict,
    shape: tuple | None = None,
    **overrides,
) -> dict:
    """
    Normalize metadata by adding all standard parameter aliases.

    This ensures that metadata values are accessible under all commonly-used
    keys for different tools and formats. Modifies the dictionary in-place.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary to normalize. Modified in-place AND returned.
    shape : tuple, optional
        Array shape for inferring Lx, Ly if not present in metadata.
    **overrides
        Override values for specific parameters (e.g., dx=0.5, fs=30.0).

    Returns
    -------
    dict
        The same metadata dict with all standard aliases added.

    Examples
    --------
    >>> meta = {"umPerPixX": 0.5, "frame_rate": 30.0}
    >>> normalize_metadata(meta)
    >>> meta["dx"]
    0.5
    >>> meta["fs"]
    30.0
    >>> meta["PhysicalSizeX"]
    0.5
    """
    # handle VoxelSize (existing comprehensive resolution handling)
    vs = get_voxel_size(
        metadata,
        dx=overrides.get("dx"),
        dy=overrides.get("dy"),
        dz=overrides.get("dz"),
    )
    metadata.update(vs.to_dict(include_aliases=True))

    # normalize other parameters
    for name, param in METADATA_PARAMS.items():
        if name in ("dx", "dy", "dz"):
            continue  # already handled by VoxelSize

        value = get_param(
            metadata, name, override=overrides.get(name), shape=shape
        )
        if value is not None:
            # set canonical key
            metadata[name] = value
            # set all aliases
            for alias in param.aliases:
                metadata[alias] = value
            # emit transform aliases (e.g. finterval from fs)
            for src_key, (_to, from_canonical) in param.transforms.items():
                try:
                    metadata[src_key] = from_canonical(value)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

    return metadata
