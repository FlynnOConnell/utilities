"""Offscreen tests for the width-fitting helpers in gui/_imgui_helpers.py.

``fit_width`` wraps text at the panel edge and collapses its block below a
minimum width; ``fit_label`` ellipsizes one-line labels. They are exercised
inside a real edge window on an ``MboNDViewer`` figure, since imgui only
reports sizes while a frame is being drawn.

``RENDERCANVAS_FORCE_OFFSCREEN=1`` must be set before *any* fastplotlib
import in the process — tests/conftest.py does this for the whole suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_manual_roi import FIGURE_SIZE, _offscreen_selected

pytestmark = pytest.mark.skipif(
    not _offscreen_selected(),
    reason="needs the offscreen rendercanvas",
)


def _draw_in_edge_window(width: int, body) -> list:
    """Draw ``body`` once inside a ``width`` px right edge window; returns
    whatever it appended to the shared list."""
    from fastplotlib.ui import ImguiWindow

    from mbo_utilities.gui._ndviewer import MboNDViewer

    out: list = []
    data = np.zeros((2, 16, 16), np.float32)
    iw = MboNDViewer(data=data, figure_kwargs={"size": FIGURE_SIZE})
    iw.show()
    try:
        iw.figure.add_imgui_window(
            ImguiWindow(update_call=lambda: body(out)),
            location="right",
            size=width,
        )
        for _ in range(2):
            iw.figure.canvas.draw()
    finally:
        iw.close()
    return out


class TestFitWidth:
    def test_wide_panel_runs_the_body_wrapped(self):
        from imgui_bundle import imgui

        from mbo_utilities.gui._imgui_helpers import fit_width

        def body(out):
            with fit_width("tools", min_width=100) as shown:
                out.append(shown)
                if shown:
                    # wrapped text advances the cursor by more than one line
                    y0 = imgui.get_cursor_pos_y()
                    imgui.text("word " * 60)
                    out.append(imgui.get_cursor_pos_y() - y0 > imgui.get_text_line_height() * 1.5)

        out = _draw_in_edge_window(300, body)
        assert out[0] is True
        assert out[1] is True, "long text must wrap, not run off the edge"

    def test_narrow_panel_collapses_and_skips_the_body(self):
        from mbo_utilities.gui._imgui_helpers import fit_width

        def body(out):
            with fit_width("tools", min_width=400) as shown:
                out.append(shown)
                if shown:
                    out.append("drew")

        out = _draw_in_edge_window(120, body)
        assert out == [False, False]

    def test_zero_min_width_never_collapses(self):
        from mbo_utilities.gui._imgui_helpers import fit_width

        def body(out):
            with fit_width() as shown:
                out.append(shown)

        assert _draw_in_edge_window(40, body) == [True, True]


class TestFitLabel:
    def test_fits_and_ellipsizes(self):
        from imgui_bundle import imgui

        from mbo_utilities.gui._imgui_helpers import fit_label

        def body(out):
            full = "a fairly long label for a button"
            out.append(fit_label(full, imgui.calc_text_size(full).x + 1))
            short = fit_label(full, 60)
            out.append(short)
            out.append(imgui.calc_text_size(short).x <= 60)
            out.append(fit_label(full, 1))

        full, short, fits, tiny = _draw_in_edge_window(300, body)[:4]
        assert full == "a fairly long label for a button"
        assert short.endswith("…") and len(short) < len(full)
        assert fits
        assert tiny == "…"
