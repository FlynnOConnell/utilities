"""Offscreen tests for the manual ROI drawing GUI.

``ManualRoiWidget`` (mbo_utilities/gui/manual_roi.py) paints masks into a
uint16 label volume over a real ``MboNDViewer`` figure. It hangs a tool
panel off the top edge of the figure and, when PreviewDataWidget hosts it
(the ``Widgets > ROIs`` toggle), adds a "ROIs" tab to the right widget.
These tests pin the mask bookkeeping (fill, overlap rejection, delete +
renumber), the pointer-event wiring through the real pygfx renderer,
persistence, the on/off toggle, and that the panel, tab and menu draw
without raising.

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
    def test_registers_the_top_panel_only(self, widget):
        # the tools live in a top edge panel like ClassificationVis; the
        # table is a tab of the host's right widget, not an edge window
        windows = widget.iw.figure.imgui_windows
        assert windows["top"] is widget.tools_window
        assert windows.get("right") is None

    def test_panel_draws_without_raising(self, widget):
        for i in range(4):
            widget.add_roi(square(2 + 12 * i, 2, 9))
        widget.selected = 1
        widget.refresh_overlay()
        # a draw error inside an imgui update call is swallowed by
        # rendercanvas, so capture it here
        errors = []
        real = widget.draw_panel

        def guarded():
            try:
                real()
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(exc)
                raise

        widget.tools_window._update_calls[:] = [guarded]
        for _ in range(3):
            widget.iw.figure.canvas.draw()
        assert not errors, errors
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


def pump(widget, seconds: float = 10.0):
    """Poll the widget's background jobs until they finish."""
    import time

    deadline = time.time() + seconds
    while time.time() < deadline:
        widget._poll_jobs()
        if not (widget._trace_job.busy or widget._run_job.busy):
            return
        time.sleep(0.02)
    raise TimeoutError("background job did not finish")


class TestTracesAndRuns:
    """Quick traces (ROI button, pixel click) and per-ROI pipeline runs go
    through the movie contract ``arr[t, c, z, y, x]`` on the viewer's own
    array, off the draw thread."""

    def test_row_actions_are_icon_only(self, widget):
        from imgui_bundle import icons_fontawesome_6 as fa

        icons = [a[0] for a in widget.row_actions]
        assert icons == [fa.ICON_FA_PLAY, fa.ICON_FA_CHART_LINE]
        assert all(len(i) == 1 for i in icons)  # a glyph, no text

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

    def test_trace_panel_draws_and_hides(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.quick_trace(0)
        pump(widget)
        # the panel is drawn from the tools panel's update call; a draw error
        # there is swallowed by rendercanvas, so capture it here
        errors = []
        real = widget.draw_panel

        def guarded():
            try:
                real()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                raise

        widget.tools_window._update_calls[:] = [guarded]
        for _ in range(3):
            widget.iw.figure.canvas.draw()
        assert not errors, errors
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
            assert gui.manual_roi.focus_tab is True
            assert iw.figure.imgui_windows["top"] is gui.manual_roi.tools_window
        finally:
            iw.close()

    def test_toggle_off_and_on_keeps_the_rois(self):
        from mbo_utilities.gui.manual_roi import attach_roi_widget, detach_roi_widget

        iw = self._open("preview", shape=(4, 1, 1, 64, 64))
        try:
            gui = self._preview(iw)
            w = attach_roi_widget(gui, focus=True)
            assert gui.manual_roi is w and w.focus_tab
            w.add_roi(square(10, 10, 9))
            assert attach_roi_widget(gui) is w, "attach is idempotent"

            detach_roi_widget(gui)
            assert gui.manual_roi is None
            assert iw.figure.imgui_windows.get("top") is None
            detach_roi_widget(gui)  # idempotent

            w2 = attach_roi_widget(gui)
            assert w2 is not w and w2.counts == [100]
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
            assert gui.manual_roi.focus_tab is False
        finally:
            iw.close()

    def test_widgets_menu_sits_between_file_and_docs(self):
        import mbo_utilities.gui.widgets.menu_bar as mb

        iw = self._open("preview")
        seen = []
        real = mb.imgui.begin_menu

        def spy(label, *args, **kwargs):
            seen.append(label)
            return real(label, *args, **kwargs)

        mb.imgui.begin_menu = spy
        try:
            iw.figure.canvas.draw()
        finally:
            mb.imgui.begin_menu = real
            iw.close()
        assert seen[:3] == ["File", "Widgets", "Docs"], seen

    def test_biohpc_is_a_popup_not_a_tab(self):
        import mbo_utilities.gui._biohpc as bh
        import mbo_utilities.gui.viewers.time_series as ts

        iw = self._open("preview")
        gui = self._preview(iw)
        tabs, windows = [], []
        real_tab, real_begin = ts.imgui.begin_tab_item, bh.imgui.begin

        def tab_spy(label, *args, **kwargs):
            tabs.append(label)
            return real_tab(label, *args, **kwargs)

        def begin_spy(label, *args, **kwargs):
            windows.append(label)
            return real_begin(label, *args, **kwargs)

        ts.imgui.begin_tab_item = tab_spy
        bh.imgui.begin = begin_spy
        try:
            iw.figure.canvas.draw()
            assert bh.POPUP_ID not in windows, "popup must stay closed until asked"
            gui._show_biohpc_popup = True
            iw.figure.canvas.draw()
        finally:
            ts.imgui.begin_tab_item = real_tab
            bh.imgui.begin = real_begin
            iw.close()
        assert "BioHPC" not in tabs, tabs
        assert bh.POPUP_ID in windows
        assert gui._show_biohpc_popup is True, "stays open until its close button"

    def test_roi_tab_is_in_the_tab_bar_and_renders(self):
        from imgui_bundle import imgui

        import mbo_utilities.gui.viewers.time_series as ts

        iw = self._open("manualroi", shape=(8, 1, 1, 64, 64))
        gui = self._preview(iw)
        gui.manual_roi.add_roi(square(10, 10, 20))

        # focus_tab selects the ROI tab on the first frame, so its body
        # (draw_tab) actually runs; a draw error inside the imgui update is
        # swallowed by rendercanvas, so capture it here
        seen, errors = [], []
        real = imgui.begin_tab_item

        def spy(label, *args, **kwargs):
            seen.append(label)
            return real(label, *args, **kwargs)

        original_draw_tab = gui.manual_roi.draw_tab
        ran = []

        def guarded():
            ran.append(True)
            try:
                original_draw_tab()
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(exc)
                raise

        ts.imgui.begin_tab_item = spy
        gui.manual_roi.draw_tab = guarded
        try:
            for _ in range(4):
                iw.figure.canvas.draw()
        finally:
            ts.imgui.begin_tab_item = real
            iw.close()

        assert "ROIs" in seen, f"ROIs tab missing from tab bar: {seen}"
        assert "Preview" in seen and "Run" in seen, "preview tabs must survive"
        assert ran, "ROI tab body never drew"
        assert not errors, f"ROI tab raised: {errors}"
        assert gui.manual_roi.focus_tab is False, "focus must be one-shot"


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
