"""Offscreen tests for the manual ROI drawing GUI.

``ManualRoiWidget`` (mbo_utilities/gui/manual_roi.py) hangs two imgui edge
windows off a real ``MboNDViewer`` figure — controls on top, the ROI table on
the left, the way masknmf's ``ClassificationVis`` lays them out — and paints
masks into a uint16 label image. These tests pin the mask bookkeeping (fill,
overlap rejection, delete + renumber), the pointer-event wiring through the
real pygfx renderer, and that both imgui windows draw every frame without
raising.

``RENDERCANVAS_FORCE_OFFSCREEN=1`` must be set before *any* fastplotlib
import in the process — tests/conftest.py does this for the whole suite.
"""

from __future__ import annotations

import os

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

    def test_interior_is_tinted_and_boundary_is_opaque(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.selected = -1
        widget.refresh_overlay()
        alpha = widget.overlay.data.value[..., 3]
        assert alpha[15, 15] == int(255 * widget.opacity)  # interior fill
        assert alpha[10, 15] == 255  # boundary outline

    def test_only_the_selected_roi_gets_a_white_outline(self, widget):
        widget.add_roi(square(2, 2, 9))
        widget.add_roi(square(20, 20, 9))
        widget.selected = 0
        widget.refresh_overlay()
        rgb = widget.overlay.data.value[..., :3]
        assert (rgb[widget.labels == 1] == 255).all(axis=1).any()
        assert not (rgb[widget.labels == 2] == 255).all(axis=1).any()

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
        # both sides of the shared edge between cols 19 and 20 are outlined
        assert alpha[15, 19] == 255
        assert alpha[15, 20] == 255

    def test_save_writes_label_image(self, widget, tmp_path):
        widget.fpath = tmp_path / "movie.tif"
        widget.add_roi(square(10, 10, 9))
        widget.save()
        saved = np.load(tmp_path / "manual_masks.npy")
        assert np.array_equal(saved, widget.labels)


class TestSelection:
    def test_select_roi_out_of_range_clears(self, widget):
        widget.add_roi(square(10, 10, 9))
        assert widget.selected == 0
        widget.select_roi(7)
        assert widget.selected == -1
        widget.select_roi(-1)
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


class TestLabelling:
    def test_unlabel_all_clears_every_roi(self, widget):
        for i in range(3):
            widget.add_roi(square(2 + 14 * i, 2, 9))
        widget.select_roi(0)
        widget.label_selected(0)
        widget.select_roi(1)
        widget.label_selected(1)
        assert list(widget.classes.labels) == [0, 1, -1]
        widget.unlabel_all()
        assert list(widget.classes.labels) == [-1, -1, -1]
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

    def test_plane_movie_pins_the_other_slider_dims(self, widget):
        movie = widget.plane_movie()
        assert movie.shape == (6, 64, 64)
        assert movie[0].shape == (64, 64)
        assert movie[1, slice(0, 8), slice(0, 4)].shape == (8, 4)

    def test_projections_reduce_over_time(self, widget):
        from mbo_utilities.gui.manual_roi import compute_projections

        movie = widget.plane_movie()
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
        widget.iw.current_index = {"t": 3}
        widget._follow_viewer()
        assert widget.bg_source_idx == 0
        assert widget._frozen_index is None

    def test_a_new_plane_drops_the_cached_projections(self, widget):
        widget._projections = drain_projections(widget)
        assert widget._projection_key == {}
        widget._projection_key = {"Z": 7}  # as if reduced from another plane
        widget.drop_stale_projections()
        assert widget._projections == {}
        assert widget._projection_key is None


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


class TestOutlineWidth:
    def test_outlines_are_one_pixel_by_default(self, widget):
        from masknmf.visualization.imgui import OUTLINE_WIDTH

        assert widget.outline_width == OUTLINE_WIDTH == 1
        widget.add_roi(square(10, 10, 20))
        widget.select_roi(-1)
        alpha = widget.overlay.data.value[..., 3]
        # one row across the ROI hits the left and right boundary once each
        assert int((alpha[20] == 255).sum()) == 2

    def test_width_is_tunable_per_widget(self, widget):
        widget.add_roi(square(10, 10, 20))
        widget.select_roi(-1)
        widget.outline_width = 3
        widget.refresh_overlay()
        alpha = widget.overlay.data.value[..., 3]
        assert int((alpha[20] == 255).sum()) == 6


class TestImguiWindows:
    def test_controls_on_top_table_on_the_left(self, widget):
        # ClassificationVis's split: every ROI control in the top edge
        # window, the ROI table in a side one. The right edge belongs to
        # PreviewDataWidget and the bottom to the NDWidget sliders, so the
        # table takes the left.
        from mbo_utilities.gui.manual_roi import PANEL_LOCATION, TABLE_LOCATION

        assert (PANEL_LOCATION, TABLE_LOCATION) == ("top", "left")
        windows = widget.iw.figure.imgui_windows
        assert windows.get(PANEL_LOCATION) is not None
        assert windows.get(TABLE_LOCATION) is not None
        assert windows.get("right") is None  # PreviewDataWidget's edge

    def test_close_gives_both_edges_back(self, widget):
        from mbo_utilities.gui.manual_roi import PANEL_LOCATION, TABLE_LOCATION

        widget.close()
        windows = widget.iw.figure.imgui_windows
        assert windows.get(PANEL_LOCATION) is None
        assert windows.get(TABLE_LOCATION) is None

    def test_table_window_follows_its_toggle(self, widget):
        from mbo_utilities.gui.manual_roi import TABLE_LOCATION

        # an empty reserved panel would be worse than no panel, so the
        # window itself comes and goes with the subwidget toggle
        widget.sync_table_window(False)
        assert widget.iw.figure.imgui_windows.get(TABLE_LOCATION) is None
        widget.sync_table_window(True)
        assert widget.iw.figure.imgui_windows.get(TABLE_LOCATION) is not None

    def test_windows_draw_without_raising(self, widget):
        for i in range(4):
            widget.add_roi(square(2 + 12 * i, 2, 9))
        widget.classes.assign([0, 2], 0)
        widget.select_roi(1)
        # rendercanvas swallows a raise inside an imgui update call, so
        # collect it off the guard rather than letting the draw look clean
        errors = draw_frames(widget, 4)
        assert not errors, errors[0]
        assert widget.overlay.data.value[..., 3].any()


def drain_projections(widget, timeout=10.0):
    """Run the real async reduce to completion and return its result."""
    import time

    widget.request_projections()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        widget._poll_projections()
        if widget._projections:
            return widget._projections
        time.sleep(0.01)
    raise AssertionError(f"projections never landed: {widget._loader.error}")


def draw_frames(widget, n=4):
    """Render n frames with both ROI panels guarded; returns their tracebacks."""
    import traceback

    from mbo_utilities.gui.manual_roi import PANEL_LOCATION, TABLE_LOCATION

    errors = []

    def guard(draw):
        def call(*_args):
            try:
                draw()
            except Exception:
                errors.append(traceback.format_exc())

        return call

    figure = widget.iw.figure
    for location, draw in (
        (PANEL_LOCATION, widget.draw_panel),
        (TABLE_LOCATION, widget.draw_table),
    ):
        figure.imgui_windows[location]._update_calls = [guard(draw)]
    for _ in range(n):
        figure.canvas.draw()
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
    """The dispatch in _create_image_widget, which the CLI test mocks past."""

    @pytest.mark.parametrize("widget", ["preview", "manualroi", "none"])
    def test_widget_attaches_without_raising(self, widget):
        from mbo_utilities.arrays.numpy import NumpyArray
        from mbo_utilities.gui.run_gui import _create_image_widget

        data = np.random.default_rng(0).random((4, 1, 1, 32, 32)).astype(np.float32)
        arr = NumpyArray(data, dims="TCZYX")

        iw = _create_image_widget(
            arr, widget=widget, figure_kwargs_override={"size": FIGURE_SIZE}
        )
        try:
            assert iw is not None
        finally:
            iw.close()

    def test_manualroi_keeps_the_preview_widget(self):
        from mbo_utilities.arrays.numpy import NumpyArray
        from mbo_utilities.gui.run_gui import _create_image_widget
        from mbo_utilities.gui.widgets.preview_data import PreviewDataWidget

        data = np.random.default_rng(0).random((4, 1, 1, 32, 32)).astype(np.float32)
        arr = NumpyArray(data, dims="TCZYX")

        iw = _create_image_widget(
            arr, widget="manualroi", figure_kwargs_override={"size": FIGURE_SIZE}
        )
        try:
            windows = iw.figure.imgui_windows.values()
            preview = [w for w in windows if isinstance(w, PreviewDataWidget)]
            assert preview, "manualroi must keep PreviewDataWidget for the windowing tabs"
            assert getattr(preview[0], "manual_roi", None) is not None
        finally:
            iw.close()


    def test_roi_panels_render_beside_the_preview_tabs(self):
        """Both edge windows draw, and the preview tab bar is untouched."""
        from imgui_bundle import imgui

        import mbo_utilities.gui.viewers.time_series as ts
        from mbo_utilities.arrays.numpy import NumpyArray
        from mbo_utilities.gui.manual_roi import PANEL_LOCATION, TABLE_LOCATION
        from mbo_utilities.gui.run_gui import _create_image_widget
        from mbo_utilities.gui.widgets.preview_data import PreviewDataWidget

        data = np.random.default_rng(0).random((8, 1, 1, 64, 64)).astype(np.float32)
        iw = _create_image_widget(
            NumpyArray(data, dims="TCZYX"),
            widget="manualroi",
            figure_kwargs_override={"size": FIGURE_SIZE},
        )
        gui = next(
            w for w in iw.figure.imgui_windows.values()
            if isinstance(w, PreviewDataWidget)
        )
        roi = gui.manual_roi
        roi.add_roi(square(10, 10, 20))
        assert iw.figure.imgui_windows[PANEL_LOCATION] is not None
        assert iw.figure.imgui_windows[TABLE_LOCATION] is not None

        seen = []
        real = imgui.begin_tab_item

        def spy(label, *args, **kwargs):
            seen.append(label)
            return real(label, *args, **kwargs)

        ts.imgui.begin_tab_item = spy
        try:
            errors = draw_frames(roi, 4)
        finally:
            ts.imgui.begin_tab_item = real
            iw.close()

        assert not errors, f"ROI panel raised: {errors[0]}"
        assert "Preview" in seen and "Run" in seen, "preview tabs must survive"
        assert "ROI" not in seen, "ROI moved to the edge windows; no tab for it"


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
