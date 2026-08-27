"""Offscreen tests for the manual ROI drawing GUI.

``ManualRoiWidget`` (mbo_utilities/gui/manual_roi.py) hangs two imgui edge
windows off a real ``MboNDViewer`` figure and paints masks into a uint16
label image. These tests pin the mask bookkeeping (fill, overlap rejection,
delete + renumber), the pointer-event wiring through the real pygfx
renderer, and that both imgui windows draw every frame without raising.

``RENDERCANVAS_FORCE_OFFSCREEN=1`` must be set before *any* fastplotlib
import in the process — tests/conftest.py does this for the whole suite.
"""

from __future__ import annotations

import os

os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")

import numpy as np
import pytest


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


class TestImguiWindows:
    def test_both_edge_windows_registered(self, widget):
        windows = widget.iw.figure.imgui_windows
        assert windows["top"] is not None
        assert windows["right"] is not None

    def test_windows_draw_without_raising(self, widget):
        for i in range(4):
            widget.add_roi(square(2 + 12 * i, 2, 9))
        widget.selected = 1
        widget.refresh_overlay()
        # a draw error inside an imgui update call is swallowed by
        # rendercanvas, so assert on the state the draw depends on instead
        widget.iw.figure.canvas.draw()
        assert widget.overlay.data.value[..., 3].any()


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
