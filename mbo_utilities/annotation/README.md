# mbo_utilities.annotation

The GUI-free half of the manual ROI + labeling tool. `gui/manual_roi.py` is
the imgui/pygfx shell; everything here imports numpy/zarr only.

## Modules

- `store.py` — `RoiLabelStore`: `(Z, Y, X)` uint16 label volume following the
  canonical TCZYX rules (any `LazyArray`'s `nz/ny/nx`; depthless data gets
  `Z == 1`; T/C are ignored), per-ROI records (plane, area, class index,
  free-text note), the user-defined class-label set, and the display palettes
  (`CLASS_COLORS` is the same tab10 set as masknmf's classification GUI, so a
  shared label set looks the same in both tools). Mutations track
  `dirty_planes` for incremental saving.
- `ngff.py` — `LabelsZarr`: OME-NGFF-style labels zarr, layout matched to
  `arrays/suite2p.py::_add_suite2p_labels` (`{"version": "0.5", "labels":
  ["0"]}` root attrs, `0` = `(Z, Y, X)` uint32 with `image-label` attrs).
  Per-ROI class/note/plane round-trip through `image-label.properties`;
  colors through `image-label.colors`; the label-name set and source-image
  path through a root `mbo` attr. One-plane chunks make autosave a
  per-stroke plane write. Foreign labels zarrs (no `properties`) load too —
  records are derived from the volume, unclassified.

## Abstraction seam with masknmf-toolbox

masknmf's `masknmf/visualization/imgui/` README lists the shared-GUI-code
candidates on its side (picking, selection wiring, panels). This package is
the counterpart on the mbo side, and the split here is drawn so a future
shared package is a file move, not a refactor:

- **shared-shape today** (duplicated by design, aligned APIs): the label-set
  model (`label_names` + `add_label_name` + tab10 colors + hotkey-per-class)
  mirrors masknmf `ClassificationVis`; autosave-on-mutation with the error
  captured into a status line mirrors `CurationVis._autosave`.
- **stays per-package**: what a "component" is (a drawn mask here, a demixed
  footprint there) and the persistence substrate (labels zarr here, results
  hdf5 there).
- **when extracting**: `store.py`'s label-set/palette block and the
  imgui label-button row in `gui/manual_roi.py` (`_draw_label_buttons`) are
  the pieces both packages would import; keep them free of mbo/masknmf
  imports.
