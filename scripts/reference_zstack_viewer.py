"""Standalone reference + Z-stack viewer for .mesc files.

Picks a .mesc file (arg, or the main app's own file-open dialog), then a
"Reference" unit (filtered to real linescan units) and a "Z-stack" unit
(filtered to real zstack units), and opens ONE window with both as
genuinely independent panels: separate pan/zoom controllers, separate
sliders, movable independently.

Why not `MboNDViewer(data=[ref, zstack])` (the mbo wrapper used for a
multi-ROI recording's synced subplots): that wrapper hard-codes
`controller_ids="sync"` (all subplots share one controller - the opposite
of "move them around separately") and gives every array the SAME
positional dim names (`_labels_from_arrays`/`_dim_names` in
`gui/_ndviewer.py`), which is correct for N views of *one* recording where
axis 0 really is T for every view, but wrong here: the reference (linescan,
T/C/ROI) and the Z-stack (T squeezed away, C/Z) don't share axis meaning
position-for-position, so the shared slider bar was scrubbing the wrong
axis in one of the two panels ("ROI" moving what was actually Z, etc).

This builds directly on the lower-level `fastplotlib.widgets.nd_widget`
primitives instead (the same layer `MboNDViewer` itself is built on, per
`mbo_utilities/gui/_ndviewer.py`):
- `controller_ids=None` on the `Figure` -> every subplot gets its own
  independent controller (`fastplotlib/layouts/_figure.py`: `None` ->
  `np.arange(n_subplots)`, one id per subplot; `MboNDViewer`'s default
  `"sync"` -> all zeros, one shared id).
- Each array gets its OWN, non-overlapping slider-dim names (prefixed
  "Reference: ..." / "Z-stack: ..." from each array's own
  `slider_dim_labels`), added to the *same* `NDWidget`'s shared
  `ReferenceIndex` as independent entries
  (`fastplotlib/widgets/nd_widget/_ndw_subplot.py::_check_slider_dims` -
  each unseen dim name gets its own reference range) - so the two panels'
  sliders never collide or drive each other, even though both live in one
  window/one `NDWidget`.

`PreviewDataWidget` (the metadata-popup side panel used elsewhere in this
app) is NOT attached here: it's built against `MboNDViewer`'s API contract
("NDWidget wrapped in the mbo ImageWidget contract" - frame-averaging
hooks, `.data`, etc.), which a raw `NDWidget` doesn't provide, and
verifying that compatibility is out of scope for this pass. Each unit's
key metadata (fs, pixel size, comment) is printed to the console instead -
see the "build on it" note in `.claude/plans/flickering-roaming-squid.md`
for wiring a real metadata popup back in.

Usage:
    python scripts/reference_zstack_viewer.py [mesc_path]
"""

import sys
from pathlib import Path

import numpy as np

from _line_geometry import (
    linescan_endpoints_um,
    roi_slice_indices,
    um_to_pixels,
    viewport_geometry,
    zstack_depth_info,
)


def _console_pick_unit(units: list[dict], label: str) -> str | None:
    """Print a table of ``units`` and read a chosen index from stdin.

    Mirrors the columns of run_gui.py's Qt unit picker (``_prompt_for_mesc_unit``)
    so the information is the same, just rendered to the terminal instead of
    a Qt table -- that dialog needs a Qt binding this install may not have
    (`pyqt6` is a base dependency only on Linux, see `pyproject.toml`).
    """
    from mbo_utilities.gui.run_gui import _fmt_duration

    print(f"\n{label} -- {len(units)} unit(s):")
    print(f"{'#':>3}  {'Unit':<14} {'Type':<16} {'T':>7} {'C':>2} "
          f"{'Z/ROI':>8} {'Y':>5} {'X':>5}  {'Duration':>9}  Comment")
    for i, u in enumerate(units):
        t, c, z, y, x = u["shape"]
        if u["kind"] == "multicube" and u["nrois"] > 1:
            z_text = f"{u['nrois']}x{z // u['nrois']}"
        elif u["nrois"] > 1:
            z_text = f"{z} ROI"
        else:
            z_text = str(z)
        dur = _fmt_duration(u.get("duration_s")) or ""
        print(f"{i:>3}  {u['munit']:<14} {u['modality_name']:<16} {t:>7} {c:>2} "
              f"{z_text:>8} {y:>5} {x:>5}  {dur:>9}  {u['comment']}")

    while True:
        raw = input(f"{label} index (blank to cancel): ").strip()
        if not raw:
            return None
        try:
            idx = int(raw)
        except ValueError:
            print(f"enter a number 0-{len(units) - 1}, or blank to cancel.")
            continue
        if 0 <= idx < len(units):
            return units[idx]["key"]
        print(f"enter a number 0-{len(units) - 1}, or blank to cancel.")


def _print_metadata(label: str, arr) -> None:
    md = arr.metadata
    print(f"\n{label}: {md.get('mesc_unit')}")
    for key in ("fs", "dx", "dy", "dz", "num_timepoints", "num_zplanes",
                "nchannels", "Ly", "Lx", "comment"):
        if key in md:
            print(f"  {key}: {md[key]}")


def _panel_dims(arr, squeezed, role: str) -> tuple[tuple[str, ...], dict]:
    """This array's own (non-spatial) slider-dim names, namespaced by
    ``role`` so two different recordings never share a dim key, plus the
    1-based ``RangeContinuous`` for each -- see module docstring."""
    from fastplotlib.widgets.nd_widget._index import RangeContinuous

    labels = arr.slider_dim_labels
    dims = tuple(f"{role}: {name}" for name in labels)
    sizes = squeezed.shape[: len(dims)]
    ranges = {name: RangeContinuous(1, int(size) + 1, 1) for name, size in zip(dims, sizes)}
    return dims, ranges


def _draw_line_overlay(ndw, mesc_path, ref_key: str, zstack_key: str, zstack_arr) -> None:
    """Overlay each Reference ROI's line on the Z-stack subplot, shown only
    on the Z-slice matching that ROI's own recorded depth.

    Each linescan ROI is scanned at one depth (AOD per-ROI z-steering), so a
    ROI's line belongs on exactly one slice of the Z-stack, not on every
    slice - it hides/shows as the Z-stack's own Z slider moves, via
    `ReferenceIndex.add_event_handler` (`fastplotlib/widgets/nd_widget/_index.py`).

    Reads the raw geometry `MescArray.metadata` doesn't carry
    (`CoordinateMapJSON.driftEndPoints` / `ReferenceViewportJSON` /
    `MinZ`/`MaxZ`/`ZDim`) - see `_line_geometry.py` and
    `.claude/plans/flickering-roaming-squid.md` for how this was
    reconstructed, including the Z-origin bug that put the first version of
    this overlay on a FOV with no visible dendrite. Skips cleanly (prints
    why) rather than crashing the viewer when required metadata is missing.
    """
    from mbo_utilities.annotation.store import CLASS_COLORS
    from mbo_utilities.gui._ndviewer import _ref_to_index

    lines_um = linescan_endpoints_um(mesc_path, ref_key)
    if lines_um is None:
        print("\nno CoordinateMapJSON/driftEndPoints on the Reference unit "
              "-- skipping the line overlay.")
        return
    vp = viewport_geometry(mesc_path, zstack_key)
    if vp is None:
        print("\nno ReferenceViewportJSON on the Z-stack unit -- skipping the line overlay.")
        return
    depth = zstack_depth_info(mesc_path, zstack_key)
    if depth is None:
        print("\nno MinZ/MaxZ/ZDim on the Z-stack unit -- skipping the line overlay.")
        return

    ny, nx = int(zstack_arr.metadata["Ly"]), int(zstack_arr.metadata["Lx"])
    pixel_lines = [um_to_pixels(seg[:2].T, vp, ny, nx) for seg in lines_um]
    slice_of = roi_slice_indices(lines_um, depth)
    colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(pixel_lines))]
    lines_ndg = ndw[0, 1].subplot.add_line_collection(
        pixel_lines, colors=colors, thickness=1.5, name="line_scan_rois",
    )

    z_dim = next(
        (f"Z-stack: {label}" for label in zstack_arr.slider_dim_labels if "Z-plane" in label),
        None,
    )
    if z_dim is None:
        # no Z slider on this Z-stack (single-plane) - every ROI is on the
        # one visible slice, so just show them all rather than skip.
        lines_ndg.visibles = np.ones(len(pixel_lines), dtype=bool)
    else:
        def _update_visibility(indices: dict) -> None:
            current = _ref_to_index(indices[z_dim])
            lines_ndg.visibles = slice_of == current

        _update_visibility({z_dim: 1})  # NDWidget starts every slider at ref value 1
        ndw.indices.add_event_handler(_update_visibility)

    print(f"\nDrew {len(pixel_lines)} line-scan ROI(s) on the Z-stack "
          f"({ny}x{nx} px, FOV {vp['width']:.0f}x{vp['height']:.0f} um) -- "
          f"each visible only on its own Z-slice:")
    for i, c in enumerate(colors):
        rgb = tuple(round(v * 255) for v in c[:3])
        print(f"  ROI {i}: slice {slice_of[i]}, rgb{rgb}")
    print("If these look mirrored vertically vs. the real dendrite, "
          "pass flip_y=True to um_to_pixels() in _line_geometry.py "
          "(Y-orientation is an open question across this codebase).")


def main() -> None:
    mesc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if mesc_path is None:
        from mbo_utilities.gui.run_gui import _select_file

        selected, *_ = _select_file()
        if isinstance(selected, list):
            selected = selected[0] if selected else None
        if not selected:
            print("no file selected.")
            return
        mesc_path = Path(selected)

    from mbo_utilities.arrays.mesc import MescArray, list_mesc_units
    from mbo_utilities.gui.run_gui import _after_show, _figure_kwargs_for_here, _squeeze_for_viewer
    from mbo_utilities.gui._ndviewer import _ROW, _COL, _ref_to_index
    from fastplotlib.widgets.nd_widget import NDWidget

    units = list_mesc_units(mesc_path)
    linescan_units = [u for u in units if u["kind"] == "packed"]
    zstack_units = [u for u in units if u["modality_name"] == "zstack"]
    if not linescan_units:
        raise SystemExit(f"no linescan units found in {mesc_path}")
    if not zstack_units:
        raise SystemExit(f"no zstack units found in {mesc_path}")

    print(f"{mesc_path.name}: {len(units)} unit(s), "
          f"{len(linescan_units)} linescan, {len(zstack_units)} zstack.")
    ref_key = _console_pick_unit(linescan_units, "Reference (linescan)")
    if ref_key is None:
        print("cancelled.")
        return
    zstack_key = _console_pick_unit(zstack_units, "Z-stack")
    if zstack_key is None:
        print("cancelled.")
        return

    ref_arr = MescArray(mesc_path, unit=ref_key)
    zstack_arr = MescArray(mesc_path, unit=zstack_key)
    _print_metadata("Reference", ref_arr)
    _print_metadata("Z-stack", zstack_arr)

    ref_view = _squeeze_for_viewer(ref_arr)
    zstack_view = _squeeze_for_viewer(zstack_arr)
    ref_dims, ref_ranges = _panel_dims(ref_arr, ref_view, "Reference")
    zstack_dims, zstack_ranges = _panel_dims(zstack_arr, zstack_view, "Z-stack")

    figure_kwargs = _figure_kwargs_for_here()

    ndw = NDWidget(
        ref_ranges={**ref_ranges, **zstack_ranges},
        shape=(1, 2),
        names=[f"Reference [{ref_key}]", f"Z-stack [{zstack_key}]"],
        controller_ids=None,  # independent controller per subplot - the fix
        **figure_kwargs,
    )

    def _set_contrast(ndg, vmin: float, vmax: float) -> None:
        # NDImage doesn't take vmin/vmax at construction (see NDImage.__init__) -
        # set on its histogram widget when present, else the graphic itself,
        # same as MboNDViewer._style_graphic does.
        cb = ndg.histogram_widget
        target = cb if cb is not None else ndg.graphic
        target.vmax = float(vmax)
        target.vmin = float(vmin)

    spatial = (_ROW, _COL)
    ref_ndg = ndw[0, 0].add_nd_image(
        data=ref_view, dims=ref_dims + spatial, spatial_dims=spatial,
        compute_histogram=True, name="Reference",
        slider_dim_transforms={d: _ref_to_index for d in ref_dims},
    )
    _set_contrast(ref_ndg, -100, 4000)
    zstack_ndg = ndw[0, 1].add_nd_image(
        data=zstack_view, dims=zstack_dims + spatial, spatial_dims=spatial,
        compute_histogram=True, name="Z-stack",
        slider_dim_transforms={d: _ref_to_index for d in zstack_dims},
    )
    _set_contrast(zstack_ndg, -100, 4000)

    _draw_line_overlay(ndw, mesc_path, ref_key, zstack_key, zstack_arr)

    ndw.show()
    _after_show(ndw)

    import fastplotlib as fpl

    fpl.loop.run()


if __name__ == "__main__":
    main()
