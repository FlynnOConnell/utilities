"""Offscreen tests for the manual ROI drawing GUI.

``ManualRoiWidget`` (mbo_utilities/gui/manual_roi.py) paints masks into a
uint16 label volume over a real ``MboNDViewer`` figure. It hangs a tool
panel off the top edge of the figure and, when PreviewDataWidget hosts it
(the ``Widgets > Manual ROI Labeling`` toggle), adds a "ROIs" tab to the
right widget. These tests pin the mask bookkeeping (fill, overlap
rejection, delete + renumber), the pointer-event wiring through the real
pygfx renderer, persistence, z-planes, the background / overlay controls,
trace jobs, the on/off toggle, and that the panel, tab and menu draw
without raising.

``RENDERCANVAS_FORCE_OFFSCREEN=1`` must be set before *any* fastplotlib
import in the process — tests/conftest.py does this for the whole suite.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")

import numpy as np
import pytest
import tifffile


def _offscreen_selected() -> bool:
    try:
        from rendercanvas.auto import RenderCanvas
    except Exception:
        return False
    return "offscreen" in RenderCanvas.__module__


pytestmark = pytest.mark.skipif(
    not _offscreen_selected(),
    reason="offscreen rendercanvas backend not selected (another backend "
    "was imported before RENDERCANVAS_FORCE_OFFSCREEN took effect)",
)

# big enough that the reserved edge windows still leave a usable viewport
FIGURE_SIZE = (1000, 800)


@pytest.fixture
def widget():
    from mbo_utilities.gui._ndviewer import MboNDViewer
    from mbo_utilities.gui.manual_roi import ManualRoiWidget

    data = np.random.default_rng(0).random((6, 64, 64)).astype(np.float32)
    iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
    iw.show()
    yield ManualRoiWidget(iw, fpath=None)
    iw.close()


def square(x0, y0, size):
    """Stroke tracing a square with its top-left corner at (x0, y0)."""
    return [
        (float(x0), float(y0)),
        (float(x0 + size), float(y0)),
        (float(x0 + size), float(y0 + size)),
        (float(x0), float(y0 + size)),
    ]


def label(widget, index, class_index):
    """Give ROI ``index`` a class the way the buttons / hotkeys do."""
    widget.select_roi(index)
    widget.assign_class(class_index)


class TestMasks:
    def test_setup(self, widget):
        assert (widget.ny, widget.nx) == (64, 64)
        assert widget.labels.shape == (64, 64)
        assert widget.overlay.data.value.shape == (64, 64, 4)
        assert widget.counts == []

    def test_add_roi_fills_polygon(self, widget):
        widget.add_roi(square(10, 10, 9))
        assert widget.counts == [100]
        assert widget.labels[15, 15] == 1
        assert widget.labels[0, 0] == 0
        assert widget.selected == 0

    def test_short_stroke_rejected(self, widget):
        widget.add_roi([(1.0, 1.0), (2.0, 2.0)])
        assert widget.counts == []
        assert "too short" in widget.status

    def test_tiny_roi_rejected(self, widget):
        widget.add_roi(square(10, 10, 1))
        assert widget.counts == []
        assert "not added" in widget.status

    def test_overlap_keeps_only_free_pixels(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.add_roi(square(15, 10, 9))
        # second ROI loses the 5 columns already owned by the first
        assert widget.counts == [100, 50]
        assert widget.labels[15, 17] == 1
        assert widget.labels[15, 22] == 2

    def test_stroke_clipped_to_image(self, widget):
        widget.add_roi(square(-20, -20, 40))
        assert widget.counts[0] > 0
        assert widget.labels.max() == 1

    def test_delete_renumbers_labels(self, widget):
        for i in range(3):
            widget.add_roi(square(2 + 12 * i, 2, 9))
        widget.delete_roi(0)
        assert widget.counts == [100, 100]
        assert set(np.unique(widget.labels)) == {0, 1, 2}

    def test_delete_out_of_range_is_a_noop(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.delete_roi(-1)
        widget.delete_roi(5)
        assert widget.counts == [100]

    def test_clear(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.clear()
        assert widget.counts == []
        assert widget.labels.max() == 0
        assert widget.selected == -1

    def test_the_outline_costs_the_mask_no_pixels(self, widget):
        """`outer` placement is what keeps a 1-3 px structure readable."""
        widget.add_roi(square(10, 10, 9))
        widget.selected = -1
        widget.refresh_overlay()
        alpha = widget.overlay.data.value[..., 3]
        fill = int(255 * widget.opacity)
        # the mask spans rows/cols 10..19; every pixel of it keeps the fill,
        # including the outermost, and the outline lands on the row above
        assert alpha[15, 15] == fill
        assert alpha[10, 15] == fill
        assert alpha[9, 15] == 255
        assert not ((widget.labels > 0) & (alpha == 255)).any()

    @pytest.mark.parametrize("placement", ["inner", "center"])
    def test_the_other_placements_eat_into_the_mask(self, widget, placement):
        from masknmf.visualization.imgui import OUTLINE_PLACEMENTS

        widget.add_roi(square(10, 10, 9))
        widget.selected = -1
        widget.placement_idx = OUTLINE_PLACEMENTS.index(placement)
        widget.refresh_overlay()
        alpha = widget.overlay.data.value[..., 3]
        assert ((widget.labels > 0) & (alpha == 255)).any()

    def test_a_tiny_roi_keeps_its_pixels(self, widget):
        """A 3x3 structure: `center`/`inner` leave one pixel, `outer` all nine."""
        from masknmf.visualization.imgui import OUTLINE_PLACEMENTS

        widget.add_roi([(4.0, 6.0), (6.0, 6.0), (6.0, 8.0), (4.0, 8.0)])
        assert widget.counts == [9]
        kept = {}
        for placement in OUTLINE_PLACEMENTS:
            outline = widget.edges(1, placement)
            kept[placement] = 9 - int(((outline > 0) & (widget.labels > 0)).sum())
        assert kept == {"outer": 9, "center": 1, "inner": 1}

    def test_turning_outlines_off_leaves_no_lines(self, widget):
        """Including the selected ROI's rim, which is an outline too."""
        widget.add_roi(square(4, 4, 12))
        widget.add_roi(square(30, 4, 12))
        widget.select_roi(0)
        widget.show_outlines = False
        widget.refresh_overlay()
        rgba = widget.overlay.data.value
        assert not (rgba[..., 3] == 255).any()
        white = (rgba[..., :3] == 255).all(axis=-1) & (rgba[..., 3] > 0)
        assert not white.any()
        # the fill is still there, and the selection still reads through it
        alpha = rgba[..., 3]
        assert alpha[widget.labels == 1].max() > alpha[widget.labels == 2].max()

    def test_only_the_selected_roi_gets_a_white_rim(self, widget):
        from masknmf.visualization.imgui import selected_rim

        widget.add_roi(square(2, 2, 9))
        widget.add_roi(square(20, 20, 9))
        widget.selected = 0
        widget.refresh_overlay()
        rgb = widget.overlay.data.value[..., :3]
        rims = [
            selected_rim(
                widget.labels == i, widget.outline_width, widget.outline_placement
            )
            for i in (1, 2)
        ]
        assert (rgb[rims[0]] == 255).all(axis=1).any()
        assert not (rgb[rims[1]] == 255).all(axis=1).any()

    def test_hiding_both_layers_hides_the_overlay(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.show_masks = False
        widget.show_outlines = False
        widget.refresh_overlay()
        assert not widget.overlay.visible
        widget.show_outlines = True
        widget.refresh_overlay()
        assert widget.overlay.visible
        assert widget.overlay.data.value[..., 3].any()

    def test_touching_rois_keep_their_seam(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.add_roi(square(20, 10, 9))
        widget.selected = -1
        widget.refresh_overlay()
        alpha = widget.overlay.data.value[..., 3]
        assert (widget.labels[15, 19], widget.labels[15, 20]) == (1, 2)
        # neighbours share no background for an `outer` line to sit on, so
        # the seam is drawn one-sided rather than merging the two masks
        assert alpha[15, 20] == 255
        assert alpha[15, 19] == int(255 * widget.opacity)

    def test_save_writes_labels_zarr(self, widget, tmp_path):
        from mbo_utilities.annotation import LabelsZarr

        widget.fpath = tmp_path / "movie.tif"
        widget.add_roi(square(10, 10, 9))
        widget.save()
        restored = LabelsZarr.load(tmp_path / "manual_labels.zarr")
        assert np.array_equal(restored.labels, widget.store.labels)
        assert restored.counts == [100]


class TestSelection:
    def test_select_roi_out_of_range_clears(self, widget):
        widget.add_roi(square(10, 10, 9))
        assert widget.selected == 0
        widget.select_roi(7)
        assert widget.selected == -1
        widget.select_roi(-1)
        assert widget.selected == -1
        widget.select_roi(None)
        assert widget.selected == -1

    def test_selected_fill_is_more_opaque(self, widget):
        from mbo_utilities.gui.manual_roi import SELECTED_OPACITY

        widget.add_roi(square(4, 4, 20))
        widget.add_roi(square(34, 34, 20))
        widget.select_roi(0)
        alpha = widget.overlay.data.value[..., 3]
        # interiors, well clear of both rims
        assert alpha[14, 14] == int(255 * SELECTED_OPACITY)
        assert alpha[44, 44] == int(255 * widget.opacity)

    def test_clicking_an_roi_selects_it(self, widget):
        widget.add_roi(square(10, 10, 20))
        widget.select_roi(-1)
        click(widget, 20, 20)
        assert widget.selected == 0
        assert widget.scroll_to_selection

    def test_clicking_the_background_clears_the_selection(self, widget):
        widget.add_roi(square(10, 10, 20))
        click(widget, 55, 55)
        assert widget.selected == -1

    def test_dragging_does_not_select(self, widget):
        widget.add_roi(square(10, 10, 20))
        widget.select_roi(-1)
        x, y = screen_pos(widget, 20, 20)
        send(widget, "pointer_down", x, y)
        send(widget, "pointer_up", x + 40, y + 40)
        assert widget.selected == -1

    def test_outlines_toggle_changes_pixels(self, widget):
        widget.add_roi(square(10, 10, 12))
        widget.select_roi(-1)
        with_outlines = widget.overlay.data.value.copy()
        widget.show_outlines = False
        widget.refresh_overlay()
        assert (with_outlines != widget.overlay.data.value).any()

    def test_overlays_are_not_pickable(self, widget):
        """the tooltip must keep reporting the image intensity, not our rgba"""
        tiles = widget.overlay.world_object.children
        assert tiles and not any(t.material.pick_write for t in tiles)
        assert not widget.stroke_line.world_object.material.pick_write
        widget.add_roi(square(10, 10, 9))
        assert not any(
            t.material.pick_write for t in widget.overlay.world_object.children
        )

    def test_stepping_walks_the_view(self, widget):
        for i in range(3):
            widget.add_roi(square(2 + 14 * i, 2, 9))
        widget.select_roi(0)
        widget.step(1)
        assert widget.selected == 1
        widget.step(10)
        assert widget.selected == 2
        widget.step(-1)
        assert widget.selected == 1

    def test_next_unlabeled_skips_labelled_rois(self, widget):
        for i in range(3):
            widget.add_roi(square(2 + 14 * i, 2, 9))
        widget.store.add_label_name("soma")
        label(widget, 1, 0)
        widget.select_roi(0)
        widget.next_unlabeled()
        assert widget.selected == 2


class TestDrawMode:
    def test_arming_lifts_the_pan_binding(self, widget):
        controls = widget.subplot.controller.controls
        assert "mouse1" in controls
        widget.set_drawing(True)
        assert "mouse1" not in controls
        assert "wheel" in controls
        widget.set_drawing(False)
        assert controls["mouse1"] == ("pan", "drag", (1.0, 1.0))

    def test_pointer_events_ignored_while_disarmed(self, widget):
        drag(widget)
        assert widget.counts == []

    def test_drag_adds_an_roi(self, widget):
        widget.set_drawing(True)
        drag(widget)
        assert len(widget.counts) == 1
        assert widget.counts[0] > 0
        assert not widget.stroke
        assert not widget.stroke_line.visible

    def test_stroke_line_tracks_the_drag(self, widget):
        widget.set_drawing(True)
        x, y, w, h = widget.subplot.viewport.rect
        cx, cy = x + w / 2, y + h / 2
        send(widget, "pointer_down", cx, cy)
        for dx in (20, 40, 60):
            send(widget, "pointer_move", cx + dx, cy + dx)
        assert len(widget.stroke) == 4
        assert widget.stroke_line.visible
        assert widget.stroke_line.data.value.shape == (4, 3)
        send(widget, "pointer_up", cx, cy)
        assert not widget.stroke_line.visible


class TestClassLabels:
    def test_assign_class_recolors_and_counts(self, widget):
        from mbo_utilities.annotation import class_color

        widget.add_roi(square(10, 10, 9))
        widget.store.add_label_name("soma")
        widget.assign_class(0)
        assert widget.store.rois[0].class_index == 0
        assert widget.store.class_counts() == [1]
        expected = tuple(int(round(c * 255)) for c in class_color(0))
        assert tuple(widget.overlay.data.value[15, 15, :3]) == expected

    def test_assign_class_without_selection_is_a_noop(self, widget):
        widget.store.add_label_name("soma")
        widget.assign_class(0)
        assert widget.store.class_counts() == [0]

    def test_unlabel(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.store.add_label_name("soma")
        widget.assign_class(0)
        widget.assign_class(-1)
        assert widget.store.rois[0].class_index == -1

    def test_unlabel_all_clears_every_roi(self, widget):
        for i in range(3):
            widget.add_roi(square(2 + 14 * i, 2, 9))
        widget.store.add_label_name("a")
        widget.store.add_label_name("b")
        label(widget, 0, 0)
        label(widget, 1, 1)
        assert list(widget.classes.labels) == [0, 1, -1]
        widget.unlabel_all()
        assert list(widget.classes.labels) == [-1, -1, -1]
        assert [r.class_index for r in widget.store.rois] == [-1, -1, -1]
        assert "cleared 3 labels" in widget.status

    def test_unlabel_all_has_its_own_return_value(self, widget):
        from masknmf.visualization.imgui import UNLABEL_ALL, UNLABELED

        # the shared draw_label_buttons signals it out of band so callers
        # never assign -2 as if it were a class index
        assert UNLABEL_ALL != UNLABELED
        widget.add_roi(square(10, 10, 9))
        widget.select_roi(0)
        widget.label_selected(UNLABELED)
        assert list(widget.classes.labels) == [UNLABELED]

    def test_seed_label_names(self):
        from mbo_utilities.gui._ndviewer import MboNDViewer
        from mbo_utilities.gui.manual_roi import ManualRoiWidget

        data = np.zeros((4, 32, 32), np.float32)
        iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
        iw.show()
        try:
            w = ManualRoiWidget(iw, label_names=("soma", "dendrite"))
            assert w.store.label_names == ("soma", "dendrite")
        finally:
            iw.close()


class TestMaskAppearance:
    def test_overlay_is_not_contrast_stretched(self, widget):
        """RGBA bytes must reach the screen as written.

        Auto-ranging off the initial all-zero array gives vmin == vmax == 0,
        which saturates every non-zero channel: tab10 class colours all come
        out white and only hues with a zero channel survive.
        """
        assert (widget.overlay.vmin, widget.overlay.vmax) == (0, 255)

    def test_outlines_are_one_pixel_by_default(self, widget):
        from masknmf.visualization.imgui import OUTLINE_WIDTH

        assert widget.outline_width == OUTLINE_WIDTH == 1
        widget.add_roi(square(10, 10, 20))
        widget.select_roi(-1)
        alpha = widget.overlay.data.value[..., 3]
        # one row across the ROI hits the left and right boundary once each
        assert int((alpha[20] == 255).sum()) == 2

    @pytest.mark.parametrize("width, band", [(1, 2), (3, 6), (5, 10)])
    def test_the_width_slider_thickens_the_outline(self, widget, width, band):
        widget.add_roi(square(10, 10, 20))
        widget.select_roi(-1)
        widget.outline_width = width
        widget.refresh_overlay()
        alpha = widget.overlay.data.value[..., 3]
        assert int((alpha[20] == 255).sum()) == band

    def test_line_opacity_is_independent_of_the_fill(self, widget):
        widget.add_roi(square(10, 10, 20))
        widget.select_roi(-1)
        widget.opacity = 0.3
        widget.outline_alpha = 0.6
        widget.refresh_overlay()
        alpha = widget.overlay.data.value[..., 3]
        edges = widget.edges() > 0
        interior = (widget.labels == 1) & ~edges
        assert int(alpha[edges].max()) == int(255 * 0.6)
        assert int(alpha[interior].max()) == int(255 * 0.3)

    def test_color_modes(self, widget):
        from mbo_utilities.annotation import class_color
        from mbo_utilities.gui.manual_roi import COLOR_MODES, UNLABELED_COLOR

        for i in range(3):
            widget.add_roi(square(2 + 14 * i, 2, 9))
        widget.store.add_label_name("soma")
        label(widget, 0, 0)
        label(widget, 1, 0)  # same group, ROI 2 left unlabeled

        widget.color_mode_idx = COLOR_MODES.index("label")
        colors = widget._colors()
        assert np.allclose(colors[0], colors[1])  # one colour per group
        assert np.allclose(colors[0], class_color(0), atol=1 / 255)  # byte-exact
        assert np.allclose(colors[2], UNLABELED_COLOR)

        widget.color_mode_idx = COLOR_MODES.index("label + roi")
        colors = widget._colors()
        assert np.allclose(colors[0], colors[1])
        assert not np.allclose(colors[2], UNLABELED_COLOR)  # its own hue

        widget.color_mode_idx = COLOR_MODES.index("roi")
        colors = widget._colors()
        assert not np.allclose(colors[0], colors[1])  # every ROI distinct

    def test_grouped_colors_survive_to_the_overlay(self, widget):
        from mbo_utilities.annotation import class_color
        from mbo_utilities.gui.manual_roi import COLOR_MODES

        widget.add_roi(square(4, 4, 12))
        widget.add_roi(square(30, 4, 12))
        widget.add_roi(square(4, 30, 12))
        widget.store.add_label_name("a")
        widget.store.add_label_name("b")
        label(widget, 0, 0)
        label(widget, 2, 0)
        label(widget, 1, 1)
        widget.color_mode_idx = COLOR_MODES.index("label")
        widget.select_roi(-1)
        rgb = widget.overlay.data.value[..., :3]
        first = rgb[widget.labels == 1][0]
        second = rgb[widget.labels == 2][0]
        third = rgb[widget.labels == 3][0]
        assert np.array_equal(first, third)  # same class, same colour
        assert not np.array_equal(first, second)
        expected = np.array([int(round(c * 255)) for c in class_color(0)], np.uint8)
        assert np.array_equal(first, expected)


class TestBackground:
    """The bg half of ClassificationVis's overlay row, over the live viewer."""

    def test_sources_are_the_movie_plus_projections(self, widget):
        assert widget.bg_sources == ["movie", "mean", "max", "std"]

    def test_show_and_opacity_drive_the_viewer_graphic(self, widget):
        widget.show_bg = False
        widget.apply_background()
        assert not widget.iw.graphics[0].visible
        widget.show_bg = True
        widget.bg_alpha = 0.4
        widget.apply_background()
        assert widget.iw.graphics[0].visible
        assert widget.iw.graphics[0].alpha == pytest.approx(0.4)

    def test_b_is_bound_to_the_background(self, widget):
        from mbo_utilities.gui.manual_roi import KEYBINDS

        assert ("b", "toggle background") in KEYBINDS

    def test_movie_is_the_plane_on_screen(self, widget):
        movie = widget.movie()
        assert movie.shape == (6, 64, 64)
        assert movie[0].shape == (64, 64)
        assert movie[1, 0:8, 0:4].shape == (8, 4)
        block = movie[1:4]
        assert block.shape == (3, 64, 64)
        assert np.allclose(block[0], movie[1])

    def test_projections_reduce_over_time(self, widget):
        from mbo_utilities.gui.manual_roi import compute_projections

        movie = widget.movie()
        projections = compute_projections(movie)
        assert set(projections) == {"mean", "max", "std"}
        stack = np.stack([np.asarray(movie[t]) for t in range(movie.shape[0])])
        assert np.allclose(projections["mean"], stack.mean(axis=0), atol=1e-5)
        assert np.allclose(projections["max"], stack.max(axis=0), atol=1e-5)

    def test_picking_a_projection_freezes_the_graphic(self, widget):
        widget._projections = drain_projections(widget)
        widget.set_bg_source(1)  # mean
        assert np.allclose(
            widget.iw.graphics[0].data.value, widget._projections["mean"]
        )
        widget.set_bg_source(0)  # back to the live movie
        assert not np.allclose(
            widget.iw.graphics[0].data.value, widget._projections["mean"]
        )

    def test_scrubbing_hands_the_graphic_back_to_the_movie(self, widget):
        widget._projections = drain_projections(widget)
        widget.set_bg_source(2)
        assert widget.bg_source_idx == 2
        # the NDWidget repaints the image on any slider move, so the frozen
        # projection is gone and the combo has to follow it
        widget.iw.indices["t"] = 3
        widget._follow_viewer()
        assert widget.bg_source_idx == 0
        assert widget._frozen_index is None

    def test_a_new_plane_drops_the_cached_projections(self, widget):
        widget._projections = drain_projections(widget)
        assert widget._projection_key == (0, 0)
        widget._projection_key = (7, 0)  # as if reduced from another plane
        widget.drop_stale_projections()
        assert widget._projections == {}
        assert widget._projection_key is None

    def test_close_hands_the_graphic_back(self, widget):
        widget._projections = drain_projections(widget)
        widget.set_bg_source(1)
        widget.show_bg = False
        widget.apply_background()
        widget.close()
        assert widget.iw.graphics[0].visible
        assert not np.allclose(
            widget.iw.graphics[0].data.value, widget._projections["mean"]
        )


class TestSummaryViewer:
    """The shared SummaryImageViewer, wired up as "Open full FOV"."""

    def test_open_populates_images_movie_and_highlight(self, widget):
        widget.add_roi(square(10, 10, 20))
        widget._projections = drain_projections(widget)
        widget.select_roi(0)
        widget.open_full_fov()
        assert widget.summary.is_open
        assert set(widget.summary.images) == {"current frame", "mean", "max", "std"}
        assert widget.summary._movies["movie"].shape == (6, 64, 64)
        assert widget.summary._highlight == widget.selected_bbox()

    def test_selected_bbox_bounds_the_roi(self, widget):
        widget.add_roi(square(10, 12, 9))
        widget.select_roi(0)
        assert widget.selected_bbox() == (12, 10, 10, 10)
        widget.select_roi(-1)
        assert widget.selected_bbox() is None

    def test_roi_contours_outline_every_roi(self, widget):
        for i in range(3):
            widget.add_roi(square(2 + 14 * i, 2, 9))
        contours = widget.roi_contours()
        assert len(contours) == 3
        assert all(c.ndim == 2 and c.shape[1] == 2 for c in contours)
        # cached, and dropped when the masks change
        assert widget.roi_contours() is contours
        widget.delete_roi(0)
        assert len(widget.roi_contours()) == 2

    def test_export_writes_the_shown_image(self, widget, tmp_path):
        widget.fpath = tmp_path / "movie.tif"
        image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        widget.export_image("mean", image)
        written = tifffile.imread(tmp_path / "summary_mean.tif")
        assert np.array_equal(written, image)
        assert "summary_mean.tif" in widget.status


class TestImguiWindows:
    def test_registers_the_top_panel_only(self, widget):
        from mbo_utilities.gui.manual_roi import PANEL_LOCATION

        # the tools live in a top edge panel like ClassificationVis; the
        # table is a tab of the host's right widget, not an edge window
        assert PANEL_LOCATION == "top"
        windows = widget.iw.figure.imgui_windows
        assert windows["top"] is widget.tools_window
        assert windows.get("left") is None
        assert windows.get("right") is None

    def test_panel_draws_without_raising(self, widget):
        for i in range(4):
            widget.add_roi(square(2 + 12 * i, 2, 9))
        widget.store.add_label_name("soma")
        label(widget, 0, 0)
        label(widget, 2, 0)
        widget.select_roi(1)
        errors = draw_frames(widget, 4)
        assert not errors, errors[0]
        assert widget.overlay.data.value[..., 3].any()

    def test_close_takes_everything_off_the_figure(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.set_drawing(True)
        names = lambda: {g.name for g in widget.subplot.graphics}  # noqa: E731
        assert {"manual_roi_overlay", "stroke"} <= names()
        widget.close()
        assert widget.iw.figure.imgui_windows.get("top") is None
        assert not ({"manual_roi_overlay", "stroke"} & names())
        # pan is handed back and a stroke no longer lands anywhere
        assert "mouse1" in widget.subplot.controller.controls
        before = widget.counts[:]
        drag(widget)
        assert widget.counts == before
        widget.close()  # idempotent


class TestPersistence:
    def test_autosave_and_restore(self, tmp_path):
        from mbo_utilities.gui._ndviewer import MboNDViewer
        from mbo_utilities.gui.manual_roi import ManualRoiWidget

        fpath = tmp_path / "movie.tif"
        data = np.random.default_rng(1).random((4, 64, 64)).astype(np.float32)

        iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
        iw.show()
        try:
            w = ManualRoiWidget(iw, fpath=fpath)
            w.add_roi(square(10, 10, 9))  # autosaves: fpath is set
            w.store.add_label_name("soma")
            w.assign_class(0)
            w.store.set_note(0, "check me")
            w._autosave()
        finally:
            iw.close()
        assert (tmp_path / "manual_labels.zarr").exists()

        iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
        iw.show()
        try:
            w2 = ManualRoiWidget(iw, fpath=fpath)
            assert w2.counts == [100]
            assert w2.store.label_names == ("soma",)
            assert w2.store.rois[0].class_index == 0
            assert w2.store.rois[0].note == "check me"
            assert "restored" in w2.status
        finally:
            iw.close()

    def test_shape_mismatch_starts_fresh(self, tmp_path):
        from mbo_utilities.annotation import LabelsZarr, RoiLabelStore
        from mbo_utilities.gui._ndviewer import MboNDViewer
        from mbo_utilities.gui.manual_roi import ManualRoiWidget

        fpath = tmp_path / "movie.tif"
        other = RoiLabelStore(2, 16, 16)
        other.add_roi(0, np.ones((16, 16), bool))
        LabelsZarr(tmp_path / "manual_labels.zarr").save(other)

        data = np.zeros((4, 64, 64), np.float32)
        iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
        iw.show()
        try:
            w = ManualRoiWidget(iw, fpath=fpath)
            assert w.counts == []
            assert "starting fresh" in w.status
        finally:
            iw.close()


@pytest.fixture
def zwidget():
    """widget over 4D (T, Z, Y, X) data -> sliders ('t', 'z'), nz == 3"""
    from mbo_utilities.gui._ndviewer import MboNDViewer
    from mbo_utilities.gui.manual_roi import ManualRoiWidget

    data = np.random.default_rng(0).random((5, 3, 64, 64)).astype(np.float32)
    iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
    iw.show()
    yield ManualRoiWidget(iw, fpath=None)
    iw.close()


class TestZPlanes:
    def test_z_axis_detected(self, zwidget):
        assert zwidget.zdim == "z"
        assert zwidget.store.nz == 3
        assert zwidget.z == 0
        assert zwidget.labels.shape == (64, 64)

    def test_stroke_lands_on_the_current_plane(self, zwidget):
        zwidget.iw.indices["z"] = 1
        assert zwidget.z == 1
        zwidget.add_roi(square(10, 10, 9))
        assert zwidget.store.rois[0].z == 1
        assert zwidget.store.labels[1, 15, 15] == 1
        assert zwidget.store.labels[0].max() == 0
        assert zwidget.store.labels[2].max() == 0

    def test_z_change_refreshes_the_overlay(self, zwidget):
        zwidget.add_roi(square(10, 10, 9))
        assert zwidget.overlay.data.value[..., 3].any()
        zwidget.iw.indices["z"] = 2
        assert not zwidget.overlay.data.value[..., 3].any()
        zwidget.iw.indices["z"] = 0
        assert zwidget.overlay.data.value[..., 3].any()

    def test_selecting_a_roi_on_another_plane_jumps_z(self, zwidget):
        zwidget.add_roi(square(10, 10, 9))  # plane 0
        zwidget.iw.indices["z"] = 2
        zwidget.add_roi(square(30, 30, 9))  # plane 2
        zwidget.select_roi(0)
        assert zwidget.z == 0
        assert zwidget.iw.indices["z"] == 0

    def test_same_pixels_usable_on_each_plane(self, zwidget):
        zwidget.add_roi(square(10, 10, 9))
        zwidget.iw.indices["z"] = 1
        zwidget.add_roi(square(10, 10, 9))
        assert zwidget.counts == [100, 100]

    def test_picking_only_sees_the_current_plane(self, zwidget):
        zwidget.add_roi(square(10, 10, 20))
        zwidget.iw.indices["z"] = 1
        click(zwidget, 20, 20)
        assert zwidget.selected == -1
        zwidget.iw.indices["z"] = 0
        click(zwidget, 20, 20)
        assert zwidget.selected == 0

    def test_z_jump_drops_an_in_progress_stroke(self, zwidget):
        zwidget.set_drawing(True)
        x, y, w, h = zwidget.subplot.viewport.rect
        send(zwidget, "pointer_down", x + w / 2, y + h / 2)
        send(zwidget, "pointer_move", x + w / 2 + 20, y + h / 2 + 20)
        assert zwidget.stroke
        zwidget.iw.indices["z"] = 1
        assert not zwidget.stroke
        assert not zwidget.stroke_line.visible

    def test_contours_and_projections_follow_the_plane(self, zwidget):
        zwidget.add_roi(square(10, 10, 9))
        assert len(zwidget.roi_contours()) == 1
        zwidget._projections = drain_projections(zwidget)
        assert zwidget._projection_key == (0, 0)
        zwidget.iw.indices["z"] = 1
        assert zwidget.roi_contours() == []
        zwidget.drop_stale_projections()
        assert zwidget._projections == {}


def pump(widget, seconds: float = 30.0):
    """Poll the widget's background jobs until they finish."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        widget._poll_jobs()
        if not (widget.trace_busy or widget._run_job.busy):
            widget._poll_jobs()
            return
        time.sleep(0.02)
    raise TimeoutError("background job did not finish")


class TestTraces:
    """Quick traces (ROI button, pixel click), pipeline extraction and per-ROI
    pipeline runs go through the movie contract ``arr[t, c, z, y, x]`` on the
    viewer's own array, off the draw thread."""

    def test_row_actions_are_icon_only(self, widget):
        from mbo_utilities.gui.manual_roi import (
            EXTRACT_TRACE_ICON,
            QUICK_TRACE_ICON,
            RUN_ICON,
        )

        actions = widget.row_actions
        assert [a.icon for a in actions] == [RUN_ICON, QUICK_TRACE_ICON, EXTRACT_TRACE_ICON]
        # icons only, with the name carried by the hover text
        assert all(len(a.icon) <= 2 for a in actions)
        assert actions[0].tooltip.startswith("Run")
        assert actions[1].tooltip.startswith("Quick trace")
        assert actions[2].tooltip.startswith("Extract trace")

    def test_quick_trace_plots_the_roi_mean(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.quick_trace(0)
        pump(widget)
        assert widget.traces is not None and widget.traces.visible
        # a floating window, not an edge window: the viewer's edges stay as they were
        assert widget.iw.figure.imgui_windows["top"] is widget.tools_window
        assert widget.iw.figure.imgui_windows.get("right") is None
        y = widget.traces.traces["ROI 0"]
        data = np.asarray(widget.iw.data[0])
        expected = data[:, widget.labels == 1].mean(axis=1)
        np.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_quick_trace_replaces_by_name(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.quick_trace(0)
        pump(widget)
        widget.quick_trace(0)
        pump(widget)
        assert list(widget.traces.traces) == ["ROI 0"]

    def test_two_rois_trace_at_once(self, widget):
        widget.add_roi(square(4, 4, 9))
        widget.add_roi(square(30, 4, 9))
        widget.quick_trace(0)
        widget.quick_trace(1)
        assert len(widget._trace_threads) == 2, "one thread per click, not one at a time"
        pump(widget)
        assert set(widget.traces.traces) == {"ROI 0", "ROI 1"}

    def test_clicking_background_traces_the_pixel(self, widget):
        widget.add_roi(square(10, 10, 9))
        click(widget, 40, 40)
        pump(widget)
        assert widget.selected == -1
        y = widget.traces.traces["px (40, 40) z1"]
        data = np.asarray(widget.iw.data[0])
        np.testing.assert_allclose(y, data[:, 40, 40], rtol=1e-5)

    def test_clicking_an_roi_does_not_trace_a_pixel(self, widget):
        widget.add_roi(square(10, 10, 9))
        click(widget, 15, 15)
        pump(widget)
        assert widget.selected == 0
        assert widget.traces is None

    def test_pixel_trace_can_be_switched_off(self, widget):
        widget.pixel_traces = False
        click(widget, 40, 40)
        pump(widget)
        assert widget.traces is None

    def test_actions_are_disabled_without_a_movie(self, widget, monkeypatch):
        monkeypatch.setattr(widget, "movie", lambda z=None: None)
        widget.add_roi(square(10, 10, 9))
        assert "movie" in widget.trace_disabled(0)
        assert "movie" in widget.extract_disabled(0)
        widget.quick_trace(0)
        assert not widget.trace_busy

    def test_extract_is_disabled_without_a_pipeline(self, widget, monkeypatch):
        monkeypatch.setattr(widget, "trace_extractor", lambda: None)
        widget.add_roi(square(10, 10, 9))
        assert widget.trace_disabled(0) is None  # quick trace still works
        assert "no installed pipeline" in widget.extract_disabled(0)

    def test_extract_trace_lands_every_returned_trace(self, widget, monkeypatch):
        class Fake:
            name = "fake"

            @staticmethod
            def extract_traces(movie, labels):
                n = int(movie.shape[0])
                return {"F": np.ones((1, n)), "Fneu": np.zeros((1, n))}

        monkeypatch.setattr(widget, "trace_extractor", lambda: Fake)
        widget.add_roi(square(10, 10, 9))
        widget.extract_trace(0)
        pump(widget)
        assert set(widget.traces.traces) == {"ROI 0 F (fake)", "ROI 0 Fneu (fake)"}
        assert widget.traces.traces["ROI 0 F (fake)"].shape == (6,)

    def test_deleting_an_roi_drops_cached_traces(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.add_roi(square(30, 10, 9))
        widget.quick_trace(1)
        pump(widget)
        assert widget.traces.traces
        # delete renumbers, so a name-keyed panel is no longer valid
        widget.delete_roi(0)
        assert widget.traces.traces == {}

    def test_trace_panel_draws_and_hides(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.quick_trace(0)
        pump(widget)
        errors = draw_frames(widget, 3)
        assert not errors, errors[0]
        assert widget.traces.draw_count >= 1
        widget.traces.hide()
        n = widget.traces.draw_count
        widget.iw.figure.canvas.draw()
        assert widget.traces.draw_count == n  # hidden panels draw nothing
        widget.close()
        assert widget.traces is None

    def test_run_roi_writes_outputs_beside_the_data(self, tmp_path):
        from mbo_utilities.gui._ndviewer import MboNDViewer
        from mbo_utilities.gui.manual_roi import ManualRoiWidget

        data = np.random.default_rng(1).random((6, 64, 64)).astype(np.float32)
        iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
        iw.show()
        try:
            w = ManualRoiWidget(iw, fpath=tmp_path / "movie.tif")
            w.add_roi(square(10, 10, 9))
            w.add_roi(square(30, 30, 9))
            w.run_roi(1)
            pump(w)
            assert w.run_status.startswith("done"), w.run_status
            out = tmp_path / "rois_roi0001"
            F = np.load(out / "F.npy")
            assert F.shape == (1, 6)
            np.testing.assert_allclose(F[0], data[:, w.labels == 2].mean(axis=1), rtol=1e-5)
            assert np.load(out / "roi_indices.npy").tolist() == [1]
            # everything listed, under the shared tag
            w.run_in_view()
            pump(w)
            assert np.load(tmp_path / "rois_manual" / "F.npy").shape == (2, 6)
        finally:
            iw.close()

    def test_run_without_a_path_is_refused(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.run_roi(0)
        assert "no data path" in widget.run_status
        assert not widget._run_job.busy


class TestTraceJobs:
    """Every click has to show up in the process manager, pass or fail."""

    def _manager(self):
        from mbo_utilities.gui.widgets.process_manager import get_process_manager

        return get_process_manager()

    def test_a_click_creates_a_job(self, widget):
        pm = self._manager()
        before = {j.job_id for j in pm.get_jobs()}
        widget.add_roi(square(10, 10, 9))
        widget.quick_trace(0)
        new = [j for j in pm.get_jobs() if j.job_id not in before]
        assert len(new) == 1
        assert new[0].task_type == "roi_trace"
        assert "ROI 0" in new[0].description
        pump(widget)
        assert new[0].status == "completed"
        assert new[0].progress == 1.0
        assert "frames" in new[0].status_message

    def test_two_rois_extract_at_once(self, widget):
        pm = self._manager()
        before = {j.job_id for j in pm.get_jobs()}
        widget.add_roi(square(4, 4, 9))
        widget.add_roi(square(30, 4, 9))
        widget.quick_trace(0)
        widget.quick_trace(1)
        new = [j for j in pm.get_jobs() if j.job_id not in before]
        assert len(new) == 2, "one job per click, not one at a time"
        pump(widget)
        assert {"ROI 0", "ROI 1"} <= set(widget.traces.traces)

    def test_a_failure_is_reported_not_swallowed(self, widget):
        pm = self._manager()
        before = {j.job_id for j in pm.get_jobs()}
        widget.add_roi(square(10, 10, 9))
        widget._start_trace("boom", lambda: 1 / 0)
        job = next(j for j in pm.get_jobs() if j.job_id not in before)
        pump(widget)
        assert job.status == "error"
        assert "ZeroDivisionError" in job.status_message
        assert "failed" in widget.status
        assert widget.traces.traces == {}


class TestTraceCursor:
    def test_the_cursor_follows_the_viewer(self, widget):
        widget.iw.indices["t"] = 3
        assert widget.current_frame() == 3
        widget._trace_panel()
        draw_frames(widget, 1)
        assert widget.traces.frame_marker == 3

    def test_dragging_the_cursor_scrubs_the_movie(self, widget):
        panel = widget._trace_panel()
        panel.on_scrub(4)
        assert widget.iw.indices["t"] == 4
        assert widget.current_frame() == 4

    def test_the_frame_is_clamped_to_the_movie(self, widget):
        widget.set_frame(9999)
        assert widget.current_frame() == 5  # the fixture movie is 6 frames
        widget.set_frame(-5)
        assert widget.current_frame() == 0


class TestPipelineTraceExtraction:
    def test_the_base_pipeline_declines_by_default(self):
        from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget

        assert PipelineWidget.extracts_traces is False
        assert PipelineWidget.extract_traces(None, None) is None

    def test_registry_lists_only_extractors(self):
        from mbo_utilities.gui.widgets.pipelines import (
            get_available_pipelines,
            get_trace_extractors,
        )

        extractors = get_trace_extractors()
        assert set(extractors) <= set(get_available_pipelines())
        assert all(p.extracts_traces for p in extractors)

    def test_suite2p_extracts_cell_and_neuropil(self):
        pytest.importorskip("suite2p")
        from mbo_utilities.gui.widgets.pipelines.suite2p import (
            Suite2pPipelineWidget,
        )

        rng = np.random.default_rng(0)
        movie = rng.random((20, 40, 40)).astype(np.float32)
        labels = np.zeros((40, 40), np.uint16)
        labels[10:14, 10:14] = 1
        result = Suite2pPipelineWidget.extract_traces(movie, labels)
        assert set(result) == {"F", "Fneu"}
        assert result["F"].shape == (1, 20)
        # uniform lam, so suite2p's weighted mean is the plain mask mean
        assert np.allclose(
            result["F"][0], movie[:, labels == 1].mean(axis=1), atol=1e-4
        )

    def test_suite2p_declines_an_empty_label_image(self):
        pytest.importorskip("suite2p")
        from mbo_utilities.gui.widgets.pipelines.suite2p import (
            Suite2pPipelineWidget,
        )

        empty = np.zeros((16, 16), np.uint16)
        assert Suite2pPipelineWidget.extract_traces(np.zeros((4, 16, 16)), empty) is None


class TestSorting:
    """`RoiOrder` maps an imgui column index onto the right sort key."""

    def _order(self, widget):
        for spec in ((10, 10, 9), (30, 10, 20), (10, 30, 14)):
            widget.add_roi(square(*spec))
        widget.store.add_label_name("a")
        widget.store.add_label_name("b")
        label(widget, 0, 1)
        label(widget, 2, 0)
        widget.order.rebuild()
        return widget.order

    def test_sorting_by_label_groups_the_rois(self, widget):
        order = self._order(widget)
        order.sort_column = 1  # the "label" column
        order.rebuild()
        labels = widget.classes.labels[order.order]
        assert list(labels) == sorted(labels)

    def test_sorting_by_area_orders_by_area(self, widget):
        order = self._order(widget)
        order.sort_column = 2  # the "area" column
        order.rebuild()
        areas = np.asarray(widget.counts)[order.order]
        assert list(areas) == sorted(areas)

    def test_descending_reverses(self, widget):
        order = self._order(widget)
        order.sort_column = 2
        order.ascending = False
        order.rebuild()
        areas = np.asarray(widget.counts)[order.order]
        assert list(areas) == sorted(areas, reverse=True)


def drain_projections(widget, timeout=10.0):
    """Run the real async reduce to completion and return its result."""
    widget.request_projections()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        widget._poll_projections()
        if widget._projections:
            return widget._projections
        time.sleep(0.01)
    raise AssertionError(f"projections never landed: {widget._loader.error}")


def draw_frames(widget, n=4):
    """Render n frames with the ROI panel guarded; returns its tracebacks.

    A raise inside an imgui update call is swallowed by rendercanvas, so
    collect it off the guard rather than letting the draw look clean.
    """
    import traceback

    errors = []

    def guarded(*_args):
        try:
            widget.draw_panel()
        except Exception:
            errors.append(traceback.format_exc())

    widget.tools_window._update_calls[:] = [guarded]
    for _ in range(n):
        widget.iw.figure.canvas.draw()
    return errors


def send(widget, kind, x, y, button=1):
    import pygfx

    event = pygfx.PointerEvent(type=kind, x=x, y=y, button=button)
    widget.subplot.renderer.handle_event(event)


def screen_pos(widget, col, row):
    """Screen position of image pixel (col, row), via the world->screen scale."""
    x, y, w, h = widget.subplot.viewport.rect
    near = widget.subplot.map_screen_to_world((x + 1, y + 1))
    far = widget.subplot.map_screen_to_world((x + w - 1, y + h - 1))
    fx = (col + 0.5 - near[0]) / (far[0] - near[0])
    fy = (row + 0.5 - near[1]) / (far[1] - near[1])
    return x + 1 + fx * (w - 2), y + 1 + fy * (h - 2)


def click(widget, col, row):
    x, y = screen_pos(widget, col, row)
    send(widget, "pointer_down", x, y)
    send(widget, "pointer_up", x, y)


def drag(widget, size=60):
    """Drag a square stroke around the middle of the viewport."""
    x, y, w, h = widget.subplot.viewport.rect
    cx, cy = x + w / 2, y + h / 2
    send(widget, "pointer_down", cx - size, cy - size)
    for dx, dy in ((size, -size), (size, size), (-size, size), (-size, -size)):
        send(widget, "pointer_move", cx + dx, cy + dy)
    send(widget, "pointer_up", cx - size, cy - size)


class TestWidgetAttach:
    """The dispatch in _create_image_widget, which the CLI test mocks past,
    and the Widgets-menu toggle it feeds."""

    @staticmethod
    def _open(widget, shape=(4, 1, 1, 32, 32)):
        from mbo_utilities.arrays.numpy import NumpyArray
        from mbo_utilities.gui.run_gui import _create_image_widget

        data = np.random.default_rng(0).random(shape).astype(np.float32)
        return _create_image_widget(
            NumpyArray(data, dims="TCZYX"),
            widget=widget,
            figure_kwargs_override={"size": FIGURE_SIZE},
        )

    @staticmethod
    def _preview(iw):
        from mbo_utilities.gui.widgets.preview_data import PreviewDataWidget

        found = [
            w for w in iw.figure.imgui_windows.values()
            if isinstance(w, PreviewDataWidget)
        ]
        return found[0] if found else None

    @pytest.fixture(autouse=True)
    def _session_toggle(self):
        """The CLI flips the toggle for the session only; start each test
        with it off and leave it off, without touching preferences."""
        from mbo_utilities.gui.widgets.widget_toggles import set_widget_enabled

        set_widget_enabled("manual_roi", False, persist=False)
        yield
        set_widget_enabled("manual_roi", False, persist=False)

    @pytest.mark.parametrize("widget", ["preview", "manualroi", "none"])
    def test_widget_attaches_without_raising(self, widget):
        iw = self._open(widget)
        try:
            assert iw is not None
        finally:
            iw.close()

    def test_preview_starts_with_the_roi_widget_off(self):
        iw = self._open("preview")
        try:
            gui = self._preview(iw)
            assert gui is not None
            assert gui.manual_roi is None
            assert iw.figure.imgui_windows.get("top") is None
        finally:
            iw.close()

    def test_manualroi_keeps_the_preview_widget_and_turns_rois_on(self):
        iw = self._open("manualroi")
        try:
            gui = self._preview(iw)
            assert gui is not None, "manualroi must keep PreviewDataWidget"
            assert gui.manual_roi is not None
            assert iw.figure.imgui_windows["top"] is gui.manual_roi.tools_window
            assert iw.figure.imgui_windows.get("left") is None
        finally:
            iw.close()

    def test_toggle_off_and_on_keeps_the_rois(self):
        iw = self._open("preview", shape=(4, 1, 1, 64, 64))
        try:
            gui = self._preview(iw)
            gui.sync_manual_roi(True)
            w = gui.manual_roi
            assert w is not None
            w.add_roi(square(10, 10, 9))
            gui.sync_manual_roi(True)
            assert gui.manual_roi is w, "attach is idempotent"

            gui.sync_manual_roi(False)
            assert gui.manual_roi is None
            assert iw.figure.imgui_windows.get("top") is None
            gui.sync_manual_roi(False)  # idempotent

            gui.sync_manual_roi(True)
            w2 = gui.manual_roi
            assert w2 is not w and w2.counts == [100]
        finally:
            iw.close()

    def test_menu_toggle_reaches_the_widget(self):
        from mbo_utilities.gui.widgets.widget_toggles import (
            WIDGET_REGISTRY,
            set_widget_enabled,
        )

        entry = next(e for e in WIDGET_REGISTRY if e.key == "manual_roi")
        iw = self._open("preview", shape=(4, 1, 1, 64, 64))
        try:
            gui = self._preview(iw)
            set_widget_enabled("manual_roi", True, persist=False)
            entry.on_toggle(gui, True)
            assert gui.manual_roi is not None
            set_widget_enabled("manual_roi", False, persist=False)
            entry.on_toggle(gui, False)
            assert gui.manual_roi is None
        finally:
            iw.close()

    def test_cleanup_tears_the_widget_down(self):
        iw = self._open("manualroi", shape=(4, 1, 1, 64, 64))
        gui = self._preview(iw)
        w = gui.manual_roi
        try:
            gui.cleanup()
            assert gui.manual_roi is None
            assert w._closed
        finally:
            iw.close()

    def test_saved_annotations_turn_the_widget_on(self, tmp_path):
        from mbo_utilities.annotation import LabelsZarr, RoiLabelStore
        from mbo_utilities.arrays.numpy import NumpyArray
        from mbo_utilities.gui.run_gui import _create_image_widget

        store = RoiLabelStore(1, 32, 32)
        store.add_roi(0, np.pad(np.ones((8, 8), bool), 12))
        LabelsZarr(tmp_path / "manual_labels.zarr").save(store)

        data = np.random.default_rng(0).random((4, 1, 1, 32, 32)).astype(np.float32)
        arr = NumpyArray(data, dims="TCZYX")
        arr.path = tmp_path / "movie.tif"  # what source_path derives from
        iw = _create_image_widget(
            arr, widget="preview", figure_kwargs_override={"size": FIGURE_SIZE}
        )
        try:
            gui = self._preview(iw)
            assert gui.manual_roi is not None
            assert gui.manual_roi.counts == [64]
        finally:
            iw.close()

    def test_roi_tab_is_in_the_tab_bar_and_renders(self):
        from imgui_bundle import imgui

        import mbo_utilities.gui.viewers.time_series as ts

        iw = self._open("manualroi", shape=(8, 1, 1, 64, 64))
        gui = self._preview(iw)
        roi = gui.manual_roi
        roi.add_roi(square(10, 10, 20))
        roi.focus_tab = True

        # focus_tab selects the ROI tab on the first frame, so its body
        # (draw_tab) actually runs; a draw error inside the imgui update is
        # swallowed by rendercanvas, so capture it here
        seen, errors = [], []
        real = imgui.begin_tab_item

        def spy(label, *args, **kwargs):
            seen.append(label)
            return real(label, *args, **kwargs)

        original_draw_tab = roi.draw_tab
        ran = []

        def guarded():
            ran.append(True)
            try:
                original_draw_tab()
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(exc)
                raise

        ts.imgui.begin_tab_item = spy
        roi.draw_tab = guarded
        try:
            panel_errors = draw_frames(roi, 4)
        finally:
            ts.imgui.begin_tab_item = real
            iw.close()

        assert not panel_errors, f"ROI panel raised: {panel_errors[0]}"
        assert "ROIs" in seen, f"ROIs tab missing from tab bar: {seen}"
        assert "Preview" in seen and "Run" in seen, "preview tabs must survive"
        assert ran, "ROI tab body never drew"
        assert not errors, f"ROI tab raised: {errors}"
        assert roi.focus_tab is False, "focus must be one-shot"


class TestWidgetSelection:
    def test_cli_routes_widget_name(self):
        from unittest import mock

        from click.testing import CliRunner

        from mbo_utilities.cli import main

        cases = {
            (): "preview",
            ("--widget", "manualroi"): "manualroi",
            ("--widget",): "preview",
            ("--no-widget",): "none",
        }
        for flags, expected in cases.items():
            with mock.patch("mbo_utilities.gui.run_gui.run_gui") as run:
                result = CliRunner().invoke(main, ["/data/x.tif", *flags])
                assert result.exit_code == 0, result.output
                assert run.call_args.kwargs["widget"] == expected
