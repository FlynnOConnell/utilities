"""The figure's shared top strip: the menu row, the panels features register
on it, and the Signal Quality split (plot on top, table in the right tab)."""

import numpy as np
import pytest

FIGURE_SIZE = (900, 700)


@pytest.fixture
def figure():
    from mbo_utilities.gui._ndviewer import MboNDViewer

    data = np.random.default_rng(0).random((4, 32, 32)).astype(np.float32)
    iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
    iw.show()
    yield iw.figure
    iw.close()


def panel(key, label="P", height=100, right_tab=None, priority=100):
    from mbo_utilities.gui._top_strip import TopPanel

    return TopPanel(key, label, lambda: None, height, right_tab, priority)


class TestTopStrip:
    def test_claims_the_top_edge_and_sizes_to_the_menu_row(self, figure):
        from mbo_utilities.gui._top_strip import MENU_HEIGHT, TopStrip

        strip = TopStrip(figure)
        assert figure.imgui_windows["top"] is strip
        assert strip.size == MENU_HEIGHT
        strip.close()
        assert figure.imgui_windows.get("top") is None

    def test_sizes_to_the_selected_panel_and_back(self, figure):
        from mbo_utilities.gui._top_strip import MENU_HEIGHT, TopStrip

        strip = TopStrip(figure)
        strip.register(panel("a", height=100, priority=10))
        short = strip.size
        assert short > MENU_HEIGHT
        # a taller panel that is not selected does not stretch the strip
        strip.register(panel("b", height=240, priority=20))
        assert strip.size == short
        strip.active = "b"
        strip._resize()
        assert strip.size > short
        strip.unregister("b")
        assert strip.size == short
        strip.unregister("a")
        assert strip.size == MENU_HEIGHT

    def test_registering_the_same_key_replaces(self, figure):
        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        strip.register(panel("a", label="one"))
        strip.register(panel("a", label="two"))
        assert [p.label for p in strip.panels] == ["two"]

    def test_panels_order_by_priority(self, figure):
        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        strip.register(panel("late", priority=50))
        strip.register(panel("early", priority=10))
        assert [p.key for p in strip.panels] == ["early", "late"]

    def test_unregistering_the_active_panel_moves_on(self, figure):
        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        strip.register(panel("a", priority=10))
        strip.register(panel("b", priority=20))
        assert strip.active == "a"
        strip.unregister("a")
        assert strip.active == "b"
        strip.unregister("b")
        assert strip.active is None

    def test_a_taller_window_gives_the_panel_more_room(self, figure):
        """Otherwise every extra pixel goes to the canvas and the strip keeps
        a band of empty space under its cards."""
        from mbo_utilities.gui._top_strip import GROW_MAX, TopStrip

        strip = TopStrip(figure)
        strip.register(panel("a", height=100))
        short = strip.size
        figure.canvas.set_logical_size(FIGURE_SIZE[0], FIGURE_SIZE[1] * 2)
        strip._resize()
        assert strip.size > short
        # ... but never runs away with the window
        figure.canvas.set_logical_size(FIGURE_SIZE[0], FIGURE_SIZE[1] * 10)
        strip._resize()
        assert strip.size <= strip._want_size()
        assert strip.size - short <= 100 * GROW_MAX

    def test_right_focus_is_one_shot(self, figure):
        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        strip._right_focus = "rois"
        assert not strip.take_right_focus("traces")
        assert strip.take_right_focus("rois")
        assert not strip.take_right_focus("rois")

    def test_hooks_run_once_per_frame(self, figure):
        from mbo_utilities.gui._top_strip import TopStrip

        strip = TopStrip(figure)
        calls = []
        hook = lambda: calls.append(1)  # noqa: E731
        strip.add_hook(hook)
        strip.add_hook(hook)  # adding twice must not double it
        for _ in range(2):
            figure.canvas.draw()
        assert calls == [1, 1]
        strip.remove_hook(hook)
        figure.canvas.draw()
        assert calls == [1, 1]


class TestSignalQualitySplit:
    """The plot goes on the top strip, the table stays in the right tab."""

    @staticmethod
    def _gui(shape=(4, 1, 5, 32, 32)):
        from mbo_utilities.arrays.numpy import NumpyArray
        from mbo_utilities.gui.run_gui import _create_image_widget
        from mbo_utilities.gui.widgets.preview_data import PreviewDataWidget

        data = np.random.default_rng(0).random(shape).astype(np.float32)
        iw = _create_image_widget(
            NumpyArray(data, dims="TCZYX"),
            widget="preview",
            figure_kwargs_override={"size": FIGURE_SIZE},
        )
        gui = next(
            w for w in iw.figure.imgui_windows.values()
            if isinstance(w, PreviewDataWidget)
        )
        return iw, gui

    @staticmethod
    def _fake_zstats(gui, n=5):
        rng = np.random.default_rng(1)
        stats = {
            "mean": rng.random(n) * 100,
            "std": rng.random(n) * 10,
            "snr": rng.random(n) * 5,
        }
        gui._zstats = [{(): stats}]
        gui._zstats_done = [True]
        gui.nz = n

    def test_menu_draws_in_the_strip_not_the_right_panel(self, monkeypatch):
        import mbo_utilities.gui.widgets.preview_data as pd

        iw, gui = self._gui()
        try:
            assert gui.top_strip is iw.figure.imgui_windows["top"]
            calls = []
            monkeypatch.setattr(pd, "draw_menu_bar", lambda parent: calls.append(1))
            iw.figure.canvas.draw()
            assert calls == [1], "the menu row draws exactly once a frame"
            # ... and the strip is the only thing drawing it
            gui.top_strip.draw_menu = None
            calls.clear()
            iw.figure.canvas.draw()
            assert calls == []
        finally:
            iw.close()

    def test_panel_appears_once_zstats_are_done(self):
        iw, gui = self._gui()
        try:
            assert not gui.top_strip.has("zstats")
            self._fake_zstats(gui)
            gui._sync_top_panels()
            assert gui.top_strip.has("zstats")
            gui._zstats_done = [False]
            gui._sync_top_panels()
            assert not gui.top_strip.has("zstats")
        finally:
            iw.close()

    def test_panel_follows_the_widgets_menu_toggle(self):
        from mbo_utilities.gui.widgets.widget_toggles import set_widget_enabled

        iw, gui = self._gui()
        try:
            self._fake_zstats(gui)
            gui._sync_top_panels()
            assert gui.top_strip.has("zstats")
            set_widget_enabled("signal_quality", False, persist=False)
            gui._sync_top_panels()
            assert not gui.top_strip.has("zstats")
        finally:
            set_widget_enabled("signal_quality", True, persist=False)
            iw.close()

    def test_the_two_halves_draw_and_pair_with_the_right_tab(self):
        import traceback

        iw, gui = self._gui()
        try:
            self._fake_zstats(gui)
            gui._sync_top_panels()
            spot = next(p for p in gui.top_strip.panels if p.key == "zstats")
            assert spot.right_tab == "signal_quality"

            errors = []

            def body():
                for draw in (gui.draw_stats_plot, gui.draw_stats_section):
                    try:
                        draw()
                    except Exception:
                        errors.append(traceback.format_exc())

            gui.top_strip._update_calls[:] = [body]
            for _ in range(2):
                iw.figure.canvas.draw()
            assert not errors, errors[0]
        finally:
            iw.close()
