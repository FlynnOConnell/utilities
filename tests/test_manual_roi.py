"""Offscreen tests for the manual ROI drawing GUI.

``ManualRoiWidget`` (mbo_utilities/gui/manual_roi.py) paints masks into a
uint16 label volume over a real ``MboNDViewer`` figure. It hangs a card
strip off the top edge of the figure and, when PreviewDataWidget hosts it
(the ``Widgets > Manual ROI Labeling`` toggle), adds ROIs / Traces / Runs
tabs to the right widget. These tests pin the mask bookkeeping (fill,
overlap rejection, delete + renumber), the pointer-event wiring through the
real pygfx renderer, persistence, z-planes, the overlay controls, region
mode, uid-keyed traces, derived sets (rows, picking, promote / discard),
run submission and restore, the on/off toggle, and that the panel, tabs and
menu draw without raising.

``RENDERCANVAS_FORCE_OFFSCREEN=1`` must be set before *any* fastplotlib
import in the process — tests/conftest.py does this for the whole suite.
The whole module skips when masknmf's shared imgui widgets (or its theme
helpers) cannot import — a broken or half-merged masknmf install must never
break collection.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")

import numpy as np
import pytest


def _offscreen_selected() -> bool:
    try:
        from rendercanvas.auto import RenderCanvas
    except Exception:
        return False
    return "offscreen" in RenderCanvas.__module__


def _roi_widgets_available() -> bool:
    # importing manual_roi itself pulls the masknmf widgets and theme; the
    # current masknmf can raise SyntaxError, which importorskip cannot catch
    try:
        from mbo_utilities.gui.manual_roi import roi_widgets_available
    except Exception:
        return False
    try:
        return roi_widgets_available()
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _offscreen_selected(),
        reason="offscreen rendercanvas backend not selected (another backend "
        "was imported before RENDERCANVAS_FORCE_OFFSCREEN took effect)",
    ),
    pytest.mark.skipif(
        not _roi_widgets_available(),
        reason="masknmf's shared imgui widgets / theme helpers do not import",
    ),
]

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


def disc(y, x, r=3):
    """Square footprint of side 2r centred near (y, x), as (ypix, xpix)."""
    yy, xx = np.mgrid[y - r : y + r, x - r : x + r]
    return yy.ravel().astype(np.int32), xx.ravel().astype(np.int32)


def make_result(widget, footprints, z=0, name="find01", kind="discover", with_traces=False):
    """A synthetic ``RunResult`` shaped like a loaded discovery dir."""
    from mbo_utilities.roi_workflow import RunResult

    rows = []
    for ypix, xpix in footprints:
        rows.append(
            {
                "ypix": np.asarray(ypix, np.int32),
                "xpix": np.asarray(xpix, np.int32),
                "lam": np.ones(len(ypix), np.float32),
                "med": (float(np.mean(ypix)), float(np.mean(xpix))),
                "npix": int(len(ypix)),
            }
        )
    F = None
    if with_traces:
        F = np.arange(len(rows) * 6, dtype=np.float32).reshape(len(rows), 6)
    return RunResult(
        path=Path(name), kind=kind, z=z, shape=(widget.ny, widget.nx),
        stat=np.array(rows, dtype=object), F=F,
        Fneu=np.zeros_like(F) if F is not None else None, norm=None,
        iscell=None, uids=None, store_indices=None,
    )


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

    def test_masks_render_feathered(self, widget):
        """The lbm_suite2p_python look: soft edges, nothing past the mask."""
        widget.add_roi(square(10, 10, 9))
        widget.selected = -1
        widget.refresh_overlay()
        alpha = widget.overlay.data.value[..., 3]
        assert alpha[15, 15] == round(255 * widget.opacity)
        assert 0 < alpha[10, 15] < alpha[15, 15]
        assert alpha[9, 15] == 0

    def test_only_the_selected_roi_gets_a_white_rim(self, widget):
        from mbo_utilities.gui.roi_runs import _rim

        widget.add_roi(square(2, 2, 9))
        widget.add_roi(square(20, 20, 9))
        widget.selected = 0
        widget.refresh_overlay()
        rgb = widget.overlay.data.value[..., :3]
        rims = [_rim(widget.labels == i) for i in (1, 2)]
        assert (rgb[rims[0]] == 255).all(axis=1).any()
        assert not (rgb[rims[1]] == 255).all(axis=1).any()

    def test_hiding_masks_hides_the_overlay(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.show_masks = False
        widget.refresh_overlay()
        assert not widget.overlay.visible
        widget.show_masks = True
        widget.refresh_overlay()
        assert widget.overlay.visible
        assert widget.overlay.data.value[..., 3].any()

    def test_touching_rois_keep_distinct_colors(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.add_roi(square(20, 10, 9))
        widget.selected = -1
        widget.refresh_overlay()
        rgb = widget.overlay.data.value[..., :3]
        assert (widget.labels[15, 19], widget.labels[15, 20]) == (1, 2)
        assert tuple(rgb[15, 19]) == widget.store.roi_rgb(0)
        assert tuple(rgb[15, 20]) == widget.store.roi_rgb(1)

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
        assert alpha[14, 14] == round(255 * SELECTED_OPACITY)
        assert alpha[44, 44] == round(255 * widget.opacity)

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

    def test_opacity_changes_pixels(self, widget):
        widget.add_roi(square(10, 10, 12))
        widget.select_roi(-1)
        before = widget.overlay.data.value.copy()
        widget.opacity = 0.8
        widget.refresh_overlay()
        assert (before != widget.overlay.data.value).any()

    def test_clicking_a_filtered_out_roi_clears_the_label_filter(self, widget):
        """A click on the image always lands on the table cursor, so the row
        highlights alongside the mask."""
        from mbo_utilities.gui.imgui import FILTER_ALL

        widget.add_roi(square(10, 10, 20))
        widget.add_roi(square(40, 40, 15))
        widget.store.add_label_name("a")
        label(widget, 1, 0)
        widget.order.filter_label = 0
        widget.order.rebuild()
        assert 0 not in widget.order.order
        click(widget, 20, 20)
        assert widget.selected == 0
        assert widget.order.current == 0
        assert widget.order.filter_label == FILTER_ALL

    def test_clicking_a_drawn_roi_clears_the_source_filter(self, widget):
        widget.add_roi(square(10, 10, 20))
        widget._add_derived(make_result(widget, [disc(45, 45)]))
        widget.order.source = 1  # the derived set only
        widget.order.rebuild()
        assert 0 not in widget.order.order
        click(widget, 20, 20)
        assert widget.selected == 0
        assert widget.order.source is None
        assert widget.order.current == 0

    def test_clicking_a_derived_component_reveals_its_row(self, widget):
        widget.add_roi(square(10, 10, 20))
        widget._add_derived(make_result(widget, [disc(45, 45)]))
        widget.order.source = 0  # drawn only
        widget.order.rebuild()
        click(widget, 45, 45)
        assert widget.selected_derived == (0, 0)
        assert widget.order.source is None
        assert widget.order.current == widget._row_index[(0, 0)]

    def test_selecting_a_visible_row_keeps_the_filters(self, widget):
        widget.add_roi(square(10, 10, 20))
        widget.add_roi(square(40, 40, 15))
        widget.store.add_label_name("a")
        label(widget, 1, 0)
        widget.order.filter_label = 0
        widget.order.rebuild()
        widget.select_roi(1)
        assert widget.order.filter_label == 0
        assert widget.order.current == 1

    def test_overlays_are_not_pickable(self, widget):
        """the tooltip must keep reporting the image intensity, not our rgba"""
        for overlay in (widget.overlay, widget.derived_overlay):
            tiles = overlay.world_object.children
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


class TestRegionMode:
    def test_region_drag_sets_the_box_not_an_roi(self, widget):
        widget.set_region_mode(True)
        assert widget.drawer.armed and widget.region_mode
        assert not widget.drawing
        drag(widget)
        assert widget.counts == []
        assert widget.region is not None
        y0, y1, x0, x1 = widget.region
        assert 0 <= y0 < y1 <= widget.ny and 0 <= x0 < x1 <= widget.nx
        assert widget.region_line.visible
        assert np.allclose(tuple(widget.region_line.offset), (0, 0, 1.75))
        assert f"region {y1 - y0}x{x1 - x0}" in widget.status

    def test_clear_region(self, widget):
        widget.set_region_mode(True)
        drag(widget)
        widget.clear_region()
        assert widget.region is None
        assert not widget.region_line.visible

    def test_modes_are_exclusive(self, widget):
        widget.set_region_mode(True)
        widget.set_drawing(True)
        assert widget.drawing and not widget.region_mode
        widget.set_region_mode(True)
        assert widget.region_mode and not widget.drawing
        widget.set_region_mode(False)
        assert not widget.drawer.armed

    def test_tiny_region_ignored(self, widget):
        widget.set_region_mode(True)
        widget._on_stroke([(10.0, 10.0), (11.0, 11.0)])
        assert widget.region is None
        assert "ignored" in widget.status

    def test_discover_without_a_region_is_refused(self, widget):
        widget.discover_region("masknmf")
        assert "region" in widget.status
        assert not widget.manager.busy


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

    def test_follow_mode_centers_the_selection(self, widget):
        widget.add_roi(square(4, 4, 9))     # fills 4..13, centroid 8.5
        widget.add_roi(square(40, 40, 9))   # fills 40..49, centroid 44.5
        widget.select_roi(0)
        cam = widget.subplot.camera
        widget.toggle_follow()
        assert widget.follow
        assert abs(cam.local.position[0] - 8.5) < 1.5
        assert abs(cam.local.position[1] - 8.5) < 1.5
        widget.select_roi(1)
        assert abs(cam.local.position[0] - 44.5) < 1.5
        assert abs(cam.local.position[1] - 44.5) < 1.5

    def test_labeling_advances_only_in_follow_mode(self, widget):
        widget.add_roi(square(4, 4, 9))
        widget.add_roi(square(40, 40, 9))
        widget.store.add_label_name("soma")
        widget.select_roi(0)
        widget.assign_class(0)
        assert widget.selected == 0  # follow off: stay put
        widget.store.set_class(0, -1)
        widget.follow = True
        widget.select_roi(0)
        widget.assign_class(0)
        assert widget.selected == 1  # follow on: step to the next unlabeled
        # clearing a label never advances
        widget.assign_class(-1)
        assert widget.selected == 1

    def test_follow_mode_advances_through_derived_rows(self, widget):
        widget._add_derived(make_result(widget, [disc(40, 40), disc(20, 20)]))
        widget.store.add_label_name("soma")
        widget.follow = True
        widget.select_derived(0, 0)
        widget.assign_class(0)
        assert widget.selected_derived == (0, 1)

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
        from mbo_utilities.gui.imgui import UNLABEL_ALL, UNLABELED

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
    def test_overlays_are_not_contrast_stretched(self, widget):
        """RGBA bytes must reach the screen as written.

        Auto-ranging off the initial all-zero array gives vmin == vmax == 0,
        which saturates every non-zero channel: tab10 class colours all come
        out white and only hues with a zero channel survive.
        """
        assert (widget.overlay.vmin, widget.overlay.vmax) == (0, 255)
        assert (widget.derived_overlay.vmin, widget.derived_overlay.vmax) == (0, 255)

    def test_feather_ramps_toward_the_edge(self, widget):
        widget.add_roi(square(10, 10, 20))
        widget.select_roi(-1)
        alpha = widget.overlay.data.value[..., 3].astype(int)
        row = alpha[20, 10:30]
        assert row[0] < row[1] < row[2]  # the 3 px ramp
        assert row[9] == round(255 * widget.opacity)


class TestImguiWindows:
    def test_registers_the_top_panel_only(self, widget):
        from mbo_utilities.gui.manual_roi import PANEL_LOCATION

        # the controls are two panels on the shared top strip; the tables are
        # tabs of the host's right widget, not edge windows of their own
        assert PANEL_LOCATION == "top"
        windows = widget.iw.figure.imgui_windows
        assert windows["top"] is widget.tools_window
        assert [p.key for p in widget.tools_window.panels] == ["roi", "traces"]
        assert windows.get("left") is None
        assert windows.get("right") is None

    def test_closing_gives_the_panels_back(self, widget):
        strip = widget.tools_window
        widget.close()
        assert [p.key for p in strip.panels] == []
        assert strip.hooks == []

    def test_panel_draws_the_cards_and_popups(self, widget):
        from imgui_bundle import imgui

        from mbo_utilities.gui.widgets.widget_toggles import set_widget_enabled

        for i in range(4):
            widget.add_roi(square(2 + 12 * i, 2, 9))
        widget.store.add_label_name("soma")
        label(widget, 0, 0)
        widget.select_roi(1)
        widget._add_derived(make_result(widget, [disc(40, 40)], with_traces=True))
        widget.set_region_mode(True)
        widget._on_stroke([(30.0, 30.0), (50.0, 50.0)])
        widget.help_open = True
        widget.keybinds_open = True

        seen = []
        real = imgui.begin_child

        def spy(name, *args, **kwargs):
            if isinstance(name, str):
                seen.append(name)
            return real(name, *args, **kwargs)

        set_widget_enabled("manual_roi", True, persist=False)
        imgui.begin_child = spy
        try:
            errors = draw_frames(widget, 4)
        finally:
            imgui.begin_child = real
            set_widget_enabled("manual_roi", False, persist=False)
        assert not errors, errors[0]
        assert {"##nav", "##draw", "##view", "##labels", "##process"} <= set(seen)
        assert "##roi_counts" in seen  # the status row's right-aligned counts

    def test_up_down_arrows_are_claimed_for_the_widget(self, widget):
        from mbo_utilities.gui import _keyboard

        errors = draw_frames(widget, 3)
        assert not errors, errors[0]
        claims = _keyboard._arrow_claims
        assert claims["up_arrow"] == claims["down_arrow"] > 0
        # left/right stay with the viewer's T scrub
        assert claims["left_arrow"] < claims["up_arrow"]

    def test_close_takes_everything_off_the_figure(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.set_drawing(True)
        names = lambda: {g.name for g in widget.subplot.graphics}  # noqa: E731
        assert {"manual_roi_overlay", "manual_roi_derived", "stroke"} <= names()
        widget.close()
        assert widget.iw.figure.imgui_windows.get("top") is None
        assert not ({"manual_roi_overlay", "manual_roi_derived", "stroke"} & names())
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

    def test_toggle_survives_a_failed_autosave(self, tmp_path):
        # an adopted parked store is the in-session truth: the zarr on disk
        # can be behind it when an autosave failed, and restoring it over
        # the parked store silently dropped ROIs
        from mbo_utilities.gui._ndviewer import MboNDViewer
        from mbo_utilities.gui.manual_roi import ManualRoiWidget

        fpath = tmp_path / "movie.tif"
        data = np.zeros((4, 64, 64), np.float32)
        iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
        iw.show()
        try:
            w = ManualRoiWidget(iw, fpath=fpath)
            w.add_roi(square(10, 10, 9))

            class Boom:
                path = w._writer.path

                def save_dirty(self, *a, **k):
                    raise OSError("read only")

            w._writer = Boom()
            w.add_roi(square(35, 35, 9))
            assert w.n_rois == 2 and "autosave failed" in w._save_error
            parked = w.store
            w.close()

            w2 = ManualRoiWidget(iw, fpath=fpath, store=parked)
            assert w2.n_rois == 2
        finally:
            iw.close()

    def test_a_raising_stroke_is_surfaced_not_swallowed(self, widget):
        def boom(*a, **k):
            raise RuntimeError("no")

        widget.store.add_roi = boom
        widget._on_stroke(square(10, 10, 9))  # must not raise
        assert "stroke failed: RuntimeError: no" in widget.status

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

    def test_derived_overlay_follows_z(self, zwidget):
        zwidget._add_derived(make_result(zwidget, [disc(40, 40)], z=2))
        assert not zwidget.derived_overlay.visible  # the set lives on z 2
        zwidget.iw.indices["z"] = 2
        assert zwidget.derived_overlay.visible
        assert zwidget.derived_overlay.data.value[40, 40, 3] > 0

    def test_selecting_a_derived_row_jumps_z(self, zwidget):
        zwidget._add_derived(make_result(zwidget, [disc(40, 40)], z=2))
        zwidget.select_derived(0, 0)
        assert zwidget.z == 2
        assert zwidget.selected_derived == (0, 0)


def pump(widget, seconds: float = 60.0):
    """Poll the widget's background work until it finishes."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        widget._poll_jobs()
        if not widget.busy:
            widget._poll_jobs()
            return
        time.sleep(0.02)
    raise TimeoutError("background work did not finish")


class TestTraces:
    """Quick traces and run outputs land in uid-keyed trace sets, for the
    Traces tab; quick traces run off the draw thread as process-manager jobs."""

    def test_row_actions_are_icon_only(self, widget):
        from mbo_utilities.gui.manual_roi import (
            REMOVE_ICON,
            RUN_ICON,
            TRACE_ICON,
        )

        actions = widget.row_actions
        assert [a.icon for a in actions] == [RUN_ICON, TRACE_ICON, REMOVE_ICON]
        assert all(len(a.icon) <= 2 for a in actions)
        assert actions[0].tooltip.startswith("Run")
        assert actions[1].tooltip.startswith("Quick trace")

    def test_quick_trace_is_the_roi_mean(self, widget):
        widget.add_roi(square(10, 10, 9))
        uid = widget.store.rois[0].uid
        widget.quick_trace(0)
        pump(widget)
        from mbo_utilities.roi_workflow import feather_mask

        entry = widget.trace_sets["quick"].data[uid]
        data = np.asarray(widget.iw.data[0])
        mask = widget.labels == 1
        w = feather_mask(mask)[mask]
        expected = data[:, mask] @ (w / w.sum())
        np.testing.assert_allclose(entry["F"], expected, rtol=1e-5)
        assert widget.trace_uid == uid
        assert widget.focus_traces
        assert widget.has_traces()

    def test_two_rois_trace_at_once(self, widget):
        widget.add_roi(square(4, 4, 9))
        widget.add_roi(square(30, 4, 9))
        uids = [r.uid for r in widget.store.rois]
        widget.quick_trace(0)
        widget.quick_trace(1)
        assert len(widget._trace_threads) == 2, "one thread per click, not one at a time"
        pump(widget)
        assert set(widget.trace_sets["quick"].data) == set(uids)

    def test_trace_uses_the_rois_plane(self, zwidget):
        zwidget.iw.indices["z"] = 2
        zwidget.add_roi(square(10, 10, 9))
        zwidget.iw.indices["z"] = 0
        zwidget.quick_trace(0)
        pump(zwidget)
        from mbo_utilities.roi_workflow import feather_mask

        data = np.asarray(zwidget.iw.data[0])
        mask = zwidget.store.labels[2] == 1
        w = feather_mask(mask)[mask]
        expected = data[:, 2][:, mask] @ (w / w.sum())
        uid = zwidget.store.rois[0].uid
        np.testing.assert_allclose(
            zwidget.trace_sets["quick"].data[uid]["F"], expected, rtol=1e-5
        )

    def test_disabled_without_a_movie(self, widget, monkeypatch):
        monkeypatch.setattr(widget, "movie", lambda z=None: None)
        widget.add_roi(square(10, 10, 9))
        assert "movie" in widget.trace_disabled(0)
        widget.quick_trace(0)
        assert not widget.trace_busy and widget.trace_sets == {}

    def test_a_click_is_a_process_manager_job(self, widget):
        from mbo_utilities.gui.widgets.process_manager import get_process_manager

        pm = get_process_manager()
        before = {j.job_id for j in pm.get_jobs()}
        widget.add_roi(square(10, 10, 9))
        widget.quick_trace(0)
        new = [j for j in pm.get_jobs() if j.job_id not in before]
        assert len(new) == 1
        assert new[0].task_type == "roi_trace"
        assert "ROI 0" in new[0].description
        pump(widget)
        assert new[0].status == "completed"
        assert "frames" in new[0].status_message

    def test_a_failure_is_reported_not_swallowed(self, widget, monkeypatch):
        import mbo_utilities.gui.manual_roi as mr
        from mbo_utilities.gui.widgets.process_manager import get_process_manager

        monkeypatch.setattr(mr, "roi_trace", lambda *a, **k: 1 / 0)
        pm = get_process_manager()
        before = {j.job_id for j in pm.get_jobs()}
        widget.add_roi(square(10, 10, 9))
        widget.quick_trace(0)
        job = next(j for j in pm.get_jobs() if j.job_id not in before)
        pump(widget)
        assert job.status == "error"
        assert "ZeroDivisionError" in job.status_message
        assert "failed" in widget.status
        assert widget.trace_sets == {}

    def test_delete_preserves_the_other_rois_traces(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.add_roi(square(30, 10, 9))
        uids = [r.uid for r in widget.store.rois]
        widget.quick_trace(0)
        widget.quick_trace(1)
        pump(widget)
        widget.delete_roi(0)
        # uid keying means the survivor's trace neither moves nor vanishes
        assert list(widget.trace_sets["quick"].data) == [uids[1]]
        widget.delete_roi(0)
        assert widget.trace_sets["quick"].data == {}

    def test_selecting_a_traced_roi_shows_it(self, widget):
        widget.add_roi(square(4, 4, 9))
        widget.add_roi(square(30, 4, 9))
        uids = [r.uid for r in widget.store.rois]
        widget.quick_trace(0)
        widget.quick_trace(1)
        pump(widget)
        widget.select_roi(0)
        assert widget.trace_uid == uids[0]
        widget.select_roi(1)
        assert widget.trace_uid == uids[1]


class TestRuns:
    def test_run_outputs_land_beside_the_data_and_in_traces(self, tmp_path):
        from mbo_utilities.gui._ndviewer import MboNDViewer
        from mbo_utilities.gui.manual_roi import ManualRoiWidget
        from mbo_utilities.gui.roi_runs import load_run_registry, registry_path

        data = np.random.default_rng(1).random((6, 64, 64)).astype(np.float32)
        iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
        iw.show()
        try:
            w = ManualRoiWidget(iw, fpath=tmp_path / "movie.tif")
            w.add_roi(square(10, 10, 9))
            w.add_roi(square(30, 30, 9))
            uid1 = w.store.rois[1].uid
            w.run_roi(1)
            pump(w)
            assert w._run_error is None
            assert w.status.startswith("done"), w.status
            out = tmp_path / "rois_roi0001"
            F = np.load(out / "F.npy")
            assert F.shape == (1, 6)
            np.testing.assert_allclose(F[0], data[:, w.labels == 2].mean(axis=1), rtol=1e-5)
            assert np.load(out / "roi_indices.npy").tolist() == [1]
            ts = w.trace_sets["rois_roi0001"]
            assert list(ts.data) == [uid1]
            np.testing.assert_allclose(ts.data[uid1]["F"], F[0])
            assert "Fneu" in ts.data[uid1]
            assert w.trace_uid == uid1 and w.focus_traces

            w.run_in_view()
            pump(w)
            assert np.load(tmp_path / "rois_manual" / "F.npy").shape == (2, 6)
            assert set(w.trace_sets["rois_manual"].data) == {w.store.rois[0].uid, uid1}
            # both runs are remembered in the sidecar for the next session
            paths = {e["path"] for e in load_run_registry(registry_path(w.fpath))}
            assert {str(out), str(tmp_path / "rois_manual")} <= paths
        finally:
            iw.close()

    def test_registry_restores_run_traces(self, tmp_path):
        from mbo_utilities.gui._ndviewer import MboNDViewer
        from mbo_utilities.gui.manual_roi import ManualRoiWidget

        data = np.random.default_rng(2).random((6, 64, 64)).astype(np.float32)
        iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
        iw.show()
        try:
            w = ManualRoiWidget(iw, fpath=tmp_path / "movie.tif")
            w.add_roi(square(10, 10, 9))
            uid = w.store.rois[0].uid
            w.run_roi(0)
            pump(w)
        finally:
            iw.close()

        iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
        iw.show()
        try:
            w2 = ManualRoiWidget(iw, fpath=tmp_path / "movie.tif")
            assert w2.counts == [100]
            # the run's traces come back keyed by the same persistent uid
            assert list(w2.trace_sets["rois_roi0000"].data) == [uid]
            assert w2.has_traces()
        finally:
            iw.close()

    def test_run_without_a_path_is_refused(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.run_roi(0)
        assert "no data path" in widget.status
        assert not widget.manager.busy

    def test_a_tag_still_being_written_is_refused(self, widget, tmp_path):
        from mbo_utilities.gui.roi_runs import RoiRun

        widget.fpath = tmp_path / "movie.tif"
        widget.add_roi(square(10, 10, 9))
        gate = threading.Event()
        run = RoiRun(kind="extract", tag="manual", description="extract rois_manual")
        widget.manager.submit(run, lambda job: gate.wait(5) and [])
        widget.run_in_view()
        assert "still being written" in widget.status
        gate.set()
        pump(widget)

    def test_run_errors_reach_the_status_row(self, widget, tmp_path):
        from mbo_utilities.gui.roi_runs import RoiRun

        widget.fpath = tmp_path / "movie.tif"

        def boom(job):
            raise ValueError("boom")

        widget.manager.submit(RoiRun(kind="extract", tag="x", description="extract rois_x"), boom)
        pump(widget)
        assert "failed" in widget._run_error
        color, text = widget._status_message()
        assert text == widget._run_error


class TestDerived:
    """Loaded run outputs: combined table rows, picking, promote / discard."""

    def _set(self, widget, with_traces=False):
        s = widget._add_derived(
            make_result(widget, [disc(40, 40), disc(52, 52)], with_traces=with_traces)
        )
        assert s is not None
        return s

    def test_combined_rows_list_drawn_first(self, widget):
        widget.add_roi(square(10, 10, 9))
        self._set(widget)
        assert widget.rows == [(-1, 0), (0, 0), (0, 1)]
        assert widget.order.sources.tolist() == [0, 1, 1]
        fmt = widget._formatters()
        assert fmt["source"](0) == "drawn"
        assert fmt["source"](1) == "find01"
        assert fmt["ok"](0) == "" and fmt["ok"](1) == "yes"
        assert len(widget.classes.labels) == 3

    def test_source_filter(self, widget):
        widget.add_roi(square(10, 10, 9))
        self._set(widget)
        widget.order.source = 0
        widget.order.rebuild()
        assert [widget.rows[int(r)][0] for r in widget.order.order] == [-1]
        widget.order.source = 1
        widget.order.rebuild()
        assert [widget.rows[int(r)][0] for r in widget.order.order] == [0, 0]
        widget.order.source = None
        widget.order.rebuild()
        assert len(widget.order.order) == 3

    def test_source_column_sorts(self, widget):
        self._set(widget)
        widget.add_roi(square(10, 10, 9))
        widget.order.sort_column = 2  # the "source" column
        widget.order.ascending = False
        widget.order.rebuild()
        codes = widget.order.sources[widget.order.order]
        assert codes.tolist() == sorted(codes, reverse=True)  # derived first

    def test_invisible_or_discarded_rows_are_not_listed(self, widget):
        s = self._set(widget)
        assert len(widget.rows) == 2
        widget.discard_derived(0, 0)
        assert widget.rows == [(0, 1)]
        widget.undiscard_derived(0, 0)
        assert len(widget.rows) == 2
        s.visible = False
        widget._resync()
        assert widget.rows == []

    def test_pick_prefers_the_derived_overlay(self, widget):
        widget.add_roi(square(10, 10, 9))
        self._set(widget)
        widget._pick(40, 40)
        assert widget.selected_derived == (0, 0)
        assert widget.selected == -1
        widget._pick(15, 15)
        assert widget.selected == 0
        assert widget.selected_derived is None
        widget._pick(2, 2)
        assert widget.selected == -1 and widget.selected_derived is None

    def test_pick_ignores_hidden_derived(self, widget):
        self._set(widget)
        widget.show_derived = False
        widget._pick(40, 40)
        assert widget.selected_derived is None
        widget.show_derived = True
        widget.discard_derived(0, 0)
        widget._pick(40, 40)
        assert widget.selected_derived is None

    def test_derived_overlay_draws_the_footprints(self, widget):
        self._set(widget)
        assert widget.derived_overlay.visible
        assert np.allclose(tuple(widget.derived_overlay.offset), (0, 0, 1.5))
        alpha = widget.derived_overlay.data.value[..., 3]
        assert alpha[40, 40] > 0 and alpha[52, 52] > 0 and alpha[0, 0] == 0
        widget.discard_derived(0, 0)
        assert widget.derived_overlay.data.value[40, 40, 3] == 0
        widget.toggle_derived_overlay()
        assert not widget.derived_overlay.visible

    def test_promote_copies_the_footprint(self, widget):
        self._set(widget, with_traces=True)
        widget.promote_derived(0, 0)
        assert widget.counts == [36]
        record = widget.store.rois[0]
        assert record.source == "find01:0"
        ypix, xpix = disc(40, 40)
        assert (widget.labels[ypix, xpix] == 1).all()
        assert widget.promoted_index(0, 0) == 0
        # the run's trace came along, keyed by the new uid
        np.testing.assert_allclose(
            widget.trace_sets["find01"].data[record.uid]["F"], np.arange(6)
        )
        # promote advances to the next promotable derived row in view
        assert widget.selected_derived == (0, 1)

    def test_promote_twice_is_refused(self, widget):
        self._set(widget)
        assert widget.promote_derived(0, 1) == 0
        assert widget.promote_derived(0, 1) is None
        assert "already promoted" in widget.status
        assert widget.counts == [36]

    def test_deleting_a_promoted_roi_reverts_the_row(self, widget):
        self._set(widget)
        widget.promote_derived(0, 0)
        assert widget.promoted_index(0, 0) == 0
        widget.delete_roi(0)
        assert widget.promoted_index(0, 0) is None
        assert widget._formatters()["source"](widget._row_index[(0, 0)]) == "find01"

    def test_promote_with_no_free_pixels_is_refused(self, widget):
        widget.add_roi(square(34, 34, 14))  # covers disc(40, 40) completely
        self._set(widget)
        assert widget.promote_derived(0, 0) is None
        assert "overlaps" in widget.status
        assert widget.counts == [225]

    def test_promote_set_reports_the_split(self, widget):
        widget.add_roi(square(34, 34, 14))  # blocks row 0
        self._set(widget)
        widget.promote_set(0)
        assert "promoted 1 / skipped 1" in widget.status
        assert widget.counts == [225, 36]

    def test_discard_advances_the_selection(self, widget):
        self._set(widget)
        widget.select_derived(0, 0)
        widget.discard_derived(0, 0, advance=True)
        assert widget.selected_derived == (0, 1)
        assert (0, 0) not in widget._row_index

    def test_delete_key_routes_by_selection_kind(self, widget):
        widget.add_roi(square(10, 10, 9))
        s = self._set(widget)
        widget.select_derived(0, 0)
        widget.delete_selected()
        assert widget.counts == [100]  # the store was not touched
        assert 0 in s.discarded
        widget.select_roi(0)
        widget.delete_selected()
        assert widget.counts == []

    def test_unload_drops_rows_and_traces(self, widget):
        self._set(widget, with_traces=True)
        widget.promote_derived(0, 0)
        assert "find01" in widget.trace_sets
        widget.unload_set(0)
        assert widget.derived == []
        assert widget.rows == [(-1, 0)]
        assert "find01" not in widget.trace_sets

    def test_select_row_routes_both_kinds(self, widget):
        widget.add_roi(square(10, 10, 9))
        self._set(widget)
        widget.select_row(1)
        assert widget.selected_derived == (0, 0) and widget.selected == -1
        widget.select_row(0)
        assert widget.selected == 0 and widget.selected_derived is None
        widget.select_row(None)
        assert widget.selected == -1


    def test_opened_result_dir_auto_loads_its_rois(self, tmp_path):
        # a suite2p / masknmf plane dir opened directly shows its own ROIs,
        # mapped onto the single-plane store whatever z the run recorded
        from mbo_utilities.gui._ndviewer import MboNDViewer
        from mbo_utilities.gui.manual_roi import ManualRoiWidget

        fpath = tmp_path / "data.bin"
        fpath.write_bytes(b"")
        stat = np.array([{
            "ypix": np.array([40, 41], np.int32),
            "xpix": np.array([40, 41], np.int32),
            "lam": np.ones(2, np.float32),
            "med": (40.0, 40.0), "npix": 2,
        }], dtype=object)
        np.save(tmp_path / "stat.npy", stat)
        np.save(tmp_path / "ops.npy",
                {"Ly": 64, "Lx": 64, "plane": 5, "pipeline": "masknmf"},
                allow_pickle=True)
        data = np.zeros((4, 64, 64), np.float32)
        iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
        iw.show()
        try:
            w = ManualRoiWidget(iw, fpath=fpath)
            assert len(w.derived) == 1
            assert w.derived[0].result.kind == "masknmf"
            assert w.derived[0].result.z == 0
            assert (0, 0) in w._row_index
        finally:
            iw.close()

    def test_accept_reject_round_trips_iscell(self, widget, tmp_path):
        res = make_result(widget, [disc(40, 40), disc(20, 20)])
        d = tmp_path / "run"
        d.mkdir()
        res.path = d
        s = widget._add_derived(res)
        assert s.accepted.all()
        widget.set_accepted(0, 1)
        assert not s.accepted[1]
        iscell = np.load(d / "iscell.npy")
        assert iscell[1, 0] == 0.0 and iscell[0, 0] == 1.0
        assert widget._formatters()["ok"](widget._row_index[(0, 1)]) == "no"
        widget.set_accepted(0, 1)
        assert np.load(d / "iscell.npy")[1, 0] == 1.0

    def test_rejected_rows_load_and_stay_curatable(self, widget, tmp_path):
        # iscell is read unfiltered so rejected cells can be re-accepted
        d = tmp_path / "run"
        d.mkdir()
        stat = np.array([{
            "ypix": np.array([40, 41], np.int32),
            "xpix": np.array([40, 41], np.int32),
            "lam": np.ones(2, np.float32), "med": (40.0, 40.0), "npix": 2,
        }] * 2, dtype=object)
        np.save(d / "stat.npy", stat)
        np.save(d / "iscell.npy", np.array([[1, 0.5], [0, 0.5]], np.float32))
        np.save(d / "ops.npy", {"Ly": 64, "Lx": 64, "plane": 1}, allow_pickle=True)
        assert widget.load_run(d)
        s = widget.derived[0]
        assert len(s.result.stat) == 2
        assert s.accepted.tolist() == [True, False]

    def test_derived_labels_persist_through_the_registry(self, widget):
        res = make_result(widget, [disc(40, 40), disc(20, 20)])
        widget._add_derived(res)
        widget.select_derived(0, 1)
        widget.assign_class(0)
        assert widget.derived[0].classes == {1: 0}
        row = widget._row_index[(0, 1)]
        assert widget.classes.labels[row] == 0
        # a reload of the same dir keeps the label
        widget._add_derived(make_result(widget, [disc(40, 40), disc(20, 20)]))
        assert widget.derived[0].classes == {1: 0}
        # promoting carries it into the drawn store
        index = widget.promote_derived(0, 1)
        assert index is not None
        assert widget.store.rois[index].class_index == 0

    def test_out_of_range_plane_is_refused(self, widget):
        assert widget._add_derived(make_result(widget, [disc(40, 40)], z=4)) is None
        assert widget.derived == []
        assert "plane 5" in widget._run_error

    def test_reload_keeps_promoted_traces(self, widget):
        self._set(widget, with_traces=True)
        assert widget.promote_derived(0, 0) is not None
        uid = widget.store.rois[0].uid
        assert uid in widget.trace_sets["find01"].data
        self._set(widget, with_traces=True)  # same path: replaces the set
        assert widget.promoted_index(0, 0) == 0
        assert uid in widget.trace_sets["find01"].data

    def test_row_actions_defer_until_after_the_table_draw(self, widget):
        import traceback

        self._set(widget)
        row = widget.rows.index((0, 0))
        widget._act_remove(row)
        # recorded only: the table may still be iterating the old rows
        assert widget._pending_row_action == ("remove", 0, 0)
        assert 0 not in widget.derived[0].discarded
        errors = []

        def body(*_args):
            try:
                widget.draw_tab()
            except Exception:
                errors.append(traceback.format_exc())

        widget.tools_window._update_calls[:] = [body]
        widget.iw.figure.canvas.draw()
        assert not errors, errors[0]
        assert 0 in widget.derived[0].discarded
        assert widget._pending_row_action is None


class TestTracesTab:
    def test_the_cursor_follows_the_viewer(self, widget):
        widget.iw.indices["t"] = 3
        assert widget.current_frame() == 3

    def test_scrubbing_moves_the_viewer(self, widget):
        widget.set_frame(4)
        assert widget.iw.indices["t"] == 4
        widget.set_frame(9999)
        assert widget.current_frame() == 5
        widget.set_frame(-5)
        assert widget.current_frame() == 0

    def test_traces_live_in_the_top_panel_and_render(self, widget):
        # a finished quick trace focuses the top panel's Traces tab, and its
        # plot body must actually run (pins the implot API this build has)
        from imgui_bundle import implot

        widget.add_roi(square(10, 10, 9))
        widget.quick_trace(0)
        pump(widget)
        assert widget.focus_traces
        plotted = []
        real = implot.plot_line

        def spy(name, *a, **k):
            plotted.append(name)
            return real(name, *a, **k)

        implot.plot_line = spy
        try:
            errors = draw_frames(widget, 3)
        finally:
            implot.plot_line = real
        assert not errors, errors[0]
        assert plotted, "the Traces tab never plotted a line"
        assert not widget.focus_traces

    def test_a_derived_selection_renders_its_trace(self, widget):
        widget._add_derived(make_result(widget, [disc(40, 40)], with_traces=True))
        widget.select_derived(0, 0)
        widget.focus_traces = True
        errors = draw_frames(widget, 3)
        assert not errors, errors[0]

    def test_trace_table_rows_and_stats(self, widget):
        widget.add_roi(square(10, 10, 9))
        widget.add_roi(square(35, 35, 9))
        widget.quick_trace(0)
        widget.quick_trace(1)
        pump(widget)
        rows = widget._trace_rows()
        assert len(rows) == 2 and all(r[:2] == ("uid", "quick") for r in rows)
        n, mean, peak, snr = widget._trace_stat(rows[0])
        assert n == 6 and peak >= mean
        # deleting an ROI drops its row and its cached stats
        widget.delete_roi(1)
        keep = ("uid", "quick", widget.store.rois[0].uid)
        assert widget._trace_rows() == [keep]
        assert list(widget._trace_stats) in ([], [keep])

    def test_derived_traces_fill_the_table(self, widget):
        widget._add_derived(make_result(widget, [disc(40, 40), disc(20, 20)],
                                        with_traces=True))
        rows = widget._trace_rows()
        assert ("row", "find01", 0) in rows and ("row", "find01", 1) in rows
        n, _mean, _peak, _snr = widget._trace_stat(("row", "find01", 1))
        assert n == 6
        widget.discard_derived(0, 1)
        assert ("row", "find01", 1) not in widget._trace_rows()

    def test_multi_select_plots_every_selected_trace(self, widget):
        from imgui_bundle import implot

        widget.add_roi(square(10, 10, 9))
        widget.add_roi(square(35, 35, 9))
        widget.quick_trace(0)
        widget.quick_trace(1)
        pump(widget)
        widget.trace_sel = set(widget._trace_rows())
        widget.focus_traces = True
        plotted = []
        real = implot.plot_line

        def spy(name, *a, **k):
            plotted.append(name)
            return real(name, *a, **k)

        implot.plot_line = spy
        try:
            errors = draw_frames(widget, 2)
        finally:
            implot.plot_line = real
        assert not errors, errors[0]
        assert len(set(plotted)) == 2, plotted

    def test_trace_table_renders(self, widget):
        import traceback

        widget.add_roi(square(10, 10, 9))
        widget.quick_trace(0)
        pump(widget)
        errors = []

        def body(*_args):
            try:
                widget.draw_trace_table()
            except Exception:
                errors.append(traceback.format_exc())

        widget.tools_window._update_calls[:] = [body]
        widget.iw.figure.canvas.draw()
        assert not errors, errors[0]

    def test_arrows_step_traces_on_the_traces_panel(self, widget):
        """Up / down walk the trace table when it is what the top panel is
        showing, and the ROI order otherwise."""
        for i in range(3):
            widget.add_roi(square(4 + 12 * i, 4, 9))
            widget.quick_trace(i)
        pump(widget)
        rows = widget._sorted_trace_rows()
        assert len(rows) == 3

        widget.tools_window.active = "traces"
        widget.select_trace(rows[0])
        widget.step(1)
        assert widget.trace_sel == {rows[1]}
        assert widget.selected == 1, "the image follows the trace"
        widget.step(-1)
        assert widget.trace_sel == {rows[0]} and widget.selected == 0
        widget.step(-1)  # clamps at the top
        assert widget.trace_sel == {rows[0]}

        widget.tools_window.active = "roi"
        widget.select_roi(0)
        widget.step(1)
        assert widget.selected == 1

    def test_selecting_an_roi_shows_its_trace(self, widget):
        for i in range(2):
            widget.add_roi(square(4 + 20 * i, 4, 9))
            widget.quick_trace(i)
        pump(widget)
        widget.select_roi(0)
        header, lines = widget._plot_lines()
        assert header == "ROI 0"
        widget.select_roi(1)
        header, lines = widget._plot_lines()
        assert header == "ROI 1", "the plot must follow the image"
        assert widget.trace_sel == {key for _label, key in lines}

    def test_a_traceless_roi_plots_nothing_rather_than_a_stale_trace(self, widget):
        widget.add_roi(square(4, 4, 9))
        widget.quick_trace(0)
        pump(widget)
        widget.add_roi(square(40, 40, 9))  # no trace of its own
        widget.select_roi(1)
        assert widget._plot_lines() is None

    def test_ctrl_click_multi_selection_survives_reselecting_a_member(self, widget):
        for i in range(2):
            widget.add_roi(square(4 + 20 * i, 4, 9))
            widget.quick_trace(i)
        pump(widget)
        rows = widget._sorted_trace_rows()
        widget.select_trace(rows[0])
        widget.toggle_trace(rows[1])
        assert widget.trace_sel == set(rows)
        widget.select_roi(0)
        assert widget.trace_sel == set(rows), "selection already covers ROI 0"

    def test_trace_columns_fit_the_narrow_tab(self, widget):
        """The tab is a ~250px column, so the table stretches to it and the
        two least useful columns start hidden instead of running off the
        right edge."""
        from mbo_utilities.gui.manual_roi import TRACE_COLUMNS

        assert [c[0] for c in TRACE_COLUMNS] == [
            "roi", "source", "frames", "mean", "peak", "snr"
        ]
        assert [c[0] for c in TRACE_COLUMNS if c[2]] == ["frames", "peak"]

    def test_trace_sort_keys_line_up_with_the_columns(self, widget):
        """draw_trace_table indexes one tuple by the clicked column index."""
        from mbo_utilities.gui.manual_roi import TRACE_COLUMNS

        widget.add_roi(square(10, 10, 9))
        widget.quick_trace(0)
        pump(widget)
        key = widget._trace_rows()[0]
        values = (widget._trace_shown(key)[0], key[1], *widget._trace_stat(key))
        assert len(values) == len(TRACE_COLUMNS)

    def test_top_choice_survives_stale_right_reports(self, widget):
        # the right bar redraws its old tab for a frame or two after the top
        # switches; those reports must not yank the top back (the ping-pong
        # that made switching to Traces flicker)
        widget._right_tab_now = "rois"
        draw_frames(widget, 2)
        widget.focus_top = "traces"
        draw_frames(widget, 2)  # set_selected lands on the second frame
        assert widget.top_tab == "traces"
        widget._right_tab_now = "runs"
        draw_frames(widget, 1)
        widget._right_tab_now = "rois"  # stale edge inside the hold window
        draw_frames(widget, 1)
        assert widget.top_tab == "traces"
        draw_frames(widget, 4)  # and it stays put once the hold expires
        assert widget.top_tab == "traces"

    def test_top_and_right_tabs_sync(self, widget):
        # a right-bar tab reporting itself pulls the top panel over
        widget._right_tab_now = "traces"
        errors = draw_frames(widget, 2)
        assert not errors, errors[0]
        assert widget.top_tab == "traces"
        # and a top change asks the right bar to follow
        widget.focus_top = "roi"
        errors = draw_frames(widget, 2)
        assert not errors, errors[0]
        assert widget.top_tab == "roi"
        assert widget._focus_right == "rois"


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
    """`RoiOrder` maps an imgui column index onto the right sort key: the
    columns dict sits in display order after "label"."""

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

    def test_sorting_by_source_groups_the_sets(self, widget):
        widget._add_derived(make_result(widget, [disc(40, 40)]))
        order = self._order(widget)
        order.sort_column = 2  # the "source" column
        order.rebuild()
        codes = widget.order.sources[order.order]
        assert list(codes) == sorted(codes)

    def test_descending_reverses(self, widget):
        widget._add_derived(make_result(widget, [disc(40, 40)]))
        order = self._order(widget)
        order.sort_column = 2
        order.ascending = False
        order.rebuild()
        codes = widget.order.sources[order.order]
        assert list(codes) == sorted(codes, reverse=True)


def draw_frames(widget, n=4):
    """Render n frames with the top strip guarded; returns its tracebacks.

    A raise inside an imgui update call is swallowed by rendercanvas, so
    collect it off the guard rather than letting the draw look clean.
    """
    import traceback

    errors = []

    strip = widget.tools_window

    def guarded(*_args):
        try:
            strip.update()
        except Exception:
            errors.append(traceback.format_exc())

    strip._update_calls[:] = [guarded]
    for _ in range(n):
        widget.iw.figure.canvas.draw()
    return errors


def draw_tab_frames(widget, n=2):
    """Same as draw_frames for the ROIs tab body, which the tools window is
    happy to host: it only needs to sit inside some imgui window."""
    import traceback

    errors = []

    def guarded(*_args):
        try:
            widget.draw_tab()
        except Exception:
            errors.append(traceback.format_exc())

    widget.tools_window._update_calls[:] = [guarded]
    for _ in range(n):
        widget.iw.figure.canvas.draw()
    return errors


def send(widget, kind, x, y, button=1, modifiers=()):
    import pygfx

    event = pygfx.PointerEvent(
        type=kind, x=x, y=y, button=button, modifiers=list(modifiers)
    )
    widget.subplot.renderer.handle_event(event)


def screen_pos(widget, col, row):
    """Screen position of image pixel (col, row), via the world->screen scale."""
    x, y, w, h = widget.subplot.viewport.rect
    near = widget.subplot.map_screen_to_world((x + 1, y + 1))
    far = widget.subplot.map_screen_to_world((x + w - 1, y + h - 1))
    fx = (col + 0.5 - near[0]) / (far[0] - near[0])
    fy = (row + 0.5 - near[1]) / (far[1] - near[1])
    return x + 1 + fx * (w - 2), y + 1 + fy * (h - 2)


def click(widget, col, row, modifiers=()):
    x, y = screen_pos(widget, col, row)
    send(widget, "pointer_down", x, y, modifiers=modifiers)
    send(widget, "pointer_up", x, y, modifiers=modifiers)


def drag(widget, size=60):
    """Drag a square stroke around the middle of the viewport."""
    x, y, w, h = widget.subplot.viewport.rect
    cx, cy = x + w / 2, y + h / 2
    send(widget, "pointer_down", cx - size, cy - size)
    for dx, dy in ((size, -size), (size, size), (-size, size), (-size, -size)):
        send(widget, "pointer_move", cx + dx, cy + dy)
    send(widget, "pointer_up", cx - size, cy - size)


class TestOverlaySurvivesGraphicRebuild:
    """A float-casting func (a gaussian sigma, a window projection) over
    integer data makes the viewer recreate the image graphic. fastplotlib
    stacks graphics in z by add order, so the rebuilt image took the front
    slot and buried the ROI masks — and the camera re-frame parked the view
    on top of them, so they never came back.
    """

    @staticmethod
    def _roi_pixel(widget):
        x, y = screen_pos(widget, 20, 20)
        frame = np.asarray(widget.iw.figure.canvas.draw())[..., :3]
        return frame[int(y), int(x)].tolist()

    @pytest.fixture
    def drawn(self):
        """An ROI over INTEGER data: that is what makes a float-casting func
        recreate the graphic in the first place."""
        from mbo_utilities.gui._ndviewer import MboNDViewer
        from mbo_utilities.gui.manual_roi import ManualRoiWidget

        data = (np.random.default_rng(0).random((6, 64, 64)) * 100).astype(np.int16)
        iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
        iw.show()
        roi = ManualRoiWidget(iw, fpath=None)
        roi.add_roi(square(10, 10, 20))
        roi.opacity = 1.0
        roi.refresh_overlay()
        for _ in range(2):
            iw.figure.canvas.draw()
        assert np.asarray(iw._ndgraphics[0].graphic.data.value).dtype.kind in "bui"
        yield roi
        iw.close()

    def test_the_mask_is_still_drawn_after_a_float_upgrade(self, drawn):
        before = self._roi_pixel(drawn)
        drawn.iw.spatial_func = lambda frame: frame  # what a gaussian does
        for _ in range(2):
            drawn.iw.figure.canvas.draw()
        after = self._roi_pixel(drawn)
        # the float texture can shift a channel by a quantisation step; being
        # covered up swaps the overlay colour for a colormap one
        assert max(abs(a - b) for a, b in zip(after, before)) <= 8, (
            f"the ROI mask was covered up: {before} -> {after}"
        )

    def test_the_stacking_and_camera_survive(self, drawn):
        iw = drawn.iw
        subplot = drawn.subplot
        z_before = {g.name: round(float(g.offset[2]), 3) for g in subplot.graphics}
        camera_before = subplot.camera.get_state()

        iw.spatial_func = lambda frame: frame
        for _ in range(2):
            iw.figure.canvas.draw()

        z_after = {g.name: round(float(g.offset[2]), 3) for g in subplot.graphics}
        for name, z in z_before.items():
            assert z_after.get(name) == z, f"{name} moved in z"
        assert round(float(subplot.camera.local.position[2]), 3) == round(
            float(camera_before["position"][2]), 3
        ), "the camera was re-framed"


class TestRoiTab:
    """The ROIs tab body: filters on one row, then the table."""

    def test_tab_draws_with_drawn_and_derived_rows(self, widget):
        widget.add_roi(square(10, 10, 20))
        widget._add_derived(make_result(widget, [disc(45, 45)]))
        widget.order.set_range_column("ok")
        assert draw_tab_frames(widget) == []

    def test_tab_draws_with_no_rois(self, widget):
        assert draw_tab_frames(widget) == []


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
            # the strip stays for the menu row, with no ROI panels on it
            assert iw.figure.imgui_windows["top"] is gui.top_strip
            assert not gui.top_strip.has("roi")
        finally:
            iw.close()

    def test_manualroi_keeps_the_preview_widget_and_turns_rois_on(self):
        iw = self._open("manualroi")
        try:
            gui = self._preview(iw)
            assert gui is not None, "manualroi must keep PreviewDataWidget"
            assert gui.manual_roi is not None
            assert gui.manual_roi.tools_window is gui.top_strip
            assert iw.figure.imgui_windows["top"] is gui.top_strip
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
            assert not gui.top_strip.has("roi")
            gui.sync_manual_roi(False)  # idempotent

            gui.sync_manual_roi(True)
            w2 = gui.manual_roi
            assert w2 is not w and w2.counts == [100]
        finally:
            iw.close()

    def test_toggle_off_and_on_keeps_the_runs(self):
        iw = self._open("preview", shape=(4, 1, 1, 64, 64))
        try:
            gui = self._preview(iw)
            gui.sync_manual_roi(True)
            w = gui.manual_roi
            w._add_derived(make_result(w, [disc(40, 40)], with_traces=True))
            w.promote_derived(0, 0)
            uid = w.store.rois[0].uid

            gui.sync_manual_roi(False)
            gui.sync_manual_roi(True)
            w2 = gui.manual_roi
            assert w2 is not w
            assert [s.name for s in w2.derived] == ["find01"]
            assert w2.counts == [36]
            # promoted state recomputes from the adopted store's sources
            assert w2.promoted_index(0, 0) == 0
            assert uid in w2.trace_sets["find01"].data
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

    def test_roi_tabs_are_in_the_tab_bar_and_render(self):
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
        assert "Traces" in seen, f"Traces tab missing from tab bar: {seen}"
        assert "Runs" not in seen, "the Runs tab was removed"
        assert "Image" in seen and "Run" in seen, "the other tabs must survive"
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


@pytest.fixture
def cwidget():
    """widget over 5D (T, C, Z, Y, X) data -> sliders ('t', 'c', 'z'),
    every (c, z) pair keys its own mask plane, z fastest."""
    from mbo_utilities.gui._ndviewer import MboNDViewer
    from mbo_utilities.gui.manual_roi import ManualRoiWidget

    data = np.random.default_rng(0).random((4, 2, 3, 64, 64)).astype(np.float32)
    iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
    iw.show()
    yield ManualRoiWidget(iw, fpath=None)
    iw.close()


class TestPlaneMapping:
    """Masks key every scrolling dim, not just z: flipping the channel (or
    any extra slider) swaps which masks show."""

    def test_axes_and_store_size(self, cwidget):
        assert cwidget.plane_axes == (("c", 2), ("z", 3))
        assert cwidget.store.nz == 6
        assert cwidget.store.plane_axes == (("c", 2), ("z", 3))

    def test_plane_is_flat_index_with_z_fastest(self, cwidget):
        assert cwidget.z == 0
        cwidget.iw.indices["z"] = 2
        assert cwidget.z == 2  # c 0 keeps plane == z
        cwidget.iw.indices["c"] = 1
        assert cwidget.z == 5
        assert cwidget._plane_pos(5) == {"c": 1, "z": 2}
        assert cwidget._plane_label(5) == "c2·z3"

    def test_stroke_lands_on_the_channel_plane(self, cwidget):
        cwidget.iw.indices["c"] = 1
        cwidget.iw.indices["z"] = 2
        cwidget.add_roi(square(10, 10, 9))
        assert cwidget.store.rois[0].z == 5
        assert cwidget.store.labels[5, 15, 15] == 1
        assert cwidget.store.labels[2].max() == 0  # same z, other channel

    def test_channel_flip_swaps_the_overlay(self, cwidget):
        cwidget.add_roi(square(10, 10, 9))  # plane 0 = (c0, z0)
        assert cwidget.overlay.data.value[..., 3].any()
        cwidget.iw.indices["c"] = 1
        assert not cwidget.overlay.data.value[..., 3].any()
        cwidget.iw.indices["c"] = 0
        assert cwidget.overlay.data.value[..., 3].any()

    def test_selecting_jumps_both_sliders(self, cwidget):
        cwidget.iw.indices["c"] = 1
        cwidget.iw.indices["z"] = 2
        cwidget.add_roi(square(10, 10, 9))
        cwidget.iw.indices["c"] = 0
        cwidget.iw.indices["z"] = 0
        cwidget.select_roi(0)
        assert cwidget.iw.indices["c"] == 1
        assert cwidget.iw.indices["z"] == 2
        assert cwidget.z == 5

    def test_movie_decodes_z_and_channel(self, cwidget):
        movie = cwidget.movie(5)
        assert movie is not None
        assert movie.z == 2 and movie.c == 1
        assert movie.shape == (4, 64, 64)

    def test_legacy_zarr_grows_into_channel_planes(self, tmp_path):
        from mbo_utilities.annotation import LabelsZarr, RoiLabelStore
        from mbo_utilities.gui._ndviewer import MboNDViewer
        from mbo_utilities.gui.manual_roi import ManualRoiWidget, labels_path

        fpath = tmp_path / "movie.tif"
        old = RoiLabelStore(3, 64, 64)
        mask = np.zeros((64, 64), bool)
        mask[10:20, 10:20] = True
        old.add_roi(1, mask)
        LabelsZarr(labels_path(fpath)).save(old, source_path=fpath)

        data = np.random.default_rng(0).random((4, 2, 3, 64, 64)).astype(np.float32)
        iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
        iw.show()
        try:
            w = ManualRoiWidget(iw, fpath=fpath)
            assert w.store.nz == 6
            assert len(w.store.rois) == 1
            # the old z1 plane is plane 1 here: (c0, z1)
            assert w.store.rois[0].z == 1
            assert w.store.labels[1, 15, 15] == 1
        finally:
            iw.close()


class TestGroupBuffer:
    """Ctrl / shift click builds a group; labels and colors apply to all."""

    def _two(self, widget):
        widget.add_roi(square(8, 8, 9))
        widget.add_roi(square(40, 40, 9))
        return widget

    def test_ctrl_click_toggles_and_seeds_from_selection(self, widget):
        self._two(widget)
        widget.select_roi(0)
        widget._pick(45, 45, frozenset({"Ctrl"}))
        assert widget.buffer == [(-1, 0), (-1, 1)]
        widget._pick(45, 45, frozenset({"Ctrl"}))
        assert widget.buffer == [(-1, 0)]

    def test_ctrl_click_through_real_pointer_events(self, widget):
        self._two(widget)
        widget.select_roi(0)
        click(widget, 45, 45, modifiers=("Ctrl",))
        assert widget.buffer == [(-1, 0), (-1, 1)]

    def test_plain_click_drops_the_group(self, widget):
        self._two(widget)
        widget.select_roi(0)
        widget._pick(45, 45, frozenset({"Ctrl"}))
        click(widget, 12, 12)
        assert widget.buffer == []
        assert widget.selected == 0

    def test_label_applies_to_every_member(self, widget):
        widget.store.add_label_name("soma")
        self._two(widget)
        widget.select_roi(0)
        widget.buffer_toggle(-1, 1)
        widget.assign_class(0)
        assert [r.class_index for r in widget.store.rois] == [0, 0]

    def test_group_color_wins_and_resets(self, widget):
        self._two(widget)
        widget.buffer_add(-1, 0)
        widget.buffer_add(-1, 1)
        widget.set_group_color((1.0, 0.0, 0.0))
        assert widget.store.roi_rgb(0) == (255, 0, 0)
        assert widget.store.roi_rgb(1) == (255, 0, 0)
        widget.set_group_color(None)
        assert widget.store.roi_rgb(0) != (255, 0, 0)

    def test_group_color_reaches_derived_rows(self, widget):
        from mbo_utilities.gui.roi_runs import component_color

        widget._add_derived(make_result(widget, [disc(30, 30)]))
        widget.buffer_toggle(0, 0)
        widget.set_group_color((0.0, 1.0, 0.0))
        assert component_color(widget.derived[0], 0) == (0.0, 1.0, 0.0)

    def test_delete_remaps_drawn_members(self, widget):
        self._two(widget)
        widget.buffer_add(-1, 0)
        widget.buffer_add(-1, 1)
        widget.delete_roi(0)
        assert widget.buffer == [(-1, 0)]

    def test_trace_color_matches_mask_color(self, widget):
        widget.add_roi(square(8, 8, 9))
        uid = widget.store.rois[0].uid
        rgb = tuple(v / 255.0 for v in widget.store.roi_rgb(0))
        assert widget._trace_color(("uid", "quick", uid)) == rgb
