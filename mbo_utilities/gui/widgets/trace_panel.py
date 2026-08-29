"""A floating panel plotting quick traces: pixels clicked in the viewer and
ROIs sent from the ROI table.

GUI-free callers compute the traces (``roi_workflow.roi_trace`` /
``pixel_trace``); this panel only holds and draws them with implot. Traces
are added by name; adding a name again replaces it, so re-clicking an ROI
refreshes its line instead of stacking duplicates.

The figure's edges are all spoken for in the viewer (tools top, tabs right,
sliders bottom), so this is a movable ``imgui.begin`` window - ``draw()`` is
called from whichever edge window hosts the owner (the ROI tools panel) and
draws nothing while hidden. It opens over the lower part of the canvas on
first show and remembers where the user drags it.
"""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui, implot

__all__ = ["TracePanel", "TRACE_PANEL_SIZE"]

TRACE_PANEL_SIZE = (560, 240)
MAX_TRACES = 12

# tab10, so an ROI's line matches its unclassified overlay hue family
_COLORS = (
    (0.12, 0.47, 0.71),
    (1.00, 0.50, 0.05),
    (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74),
    (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76),
    (0.50, 0.50, 0.50),
    (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81),
)


def _ensure_implot_context():
    if implot.get_current_context() is None:
        implot.create_context()


class TracePanel:
    """Holds named 1-D traces and draws them in a floating window."""

    def __init__(self, figure, title: str = "Traces", size=TRACE_PANEL_SIZE):
        self.figure = figure
        self.title = title
        self.size = size
        self.traces: dict[str, np.ndarray] = {}
        self.status = ""
        self.normalize = False
        self.visible = False
        # the viewer's t: drawn as a guide, and draggable when the owner
        # supplies ``on_scrub(frame)`` so the cursor scrubs the movie
        self.frame_marker: int | None = None
        self.on_scrub = None
        self.draw_count = 0
        self._closed = False
        self._placed = False

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------

    def add(self, name: str, y, *, replace: bool = True) -> None:
        y = np.asarray(y, np.float32).reshape(-1)
        if not replace and name in self.traces:
            k = 2
            while f"{name} ({k})" in self.traces:
                k += 1
            name = f"{name} ({k})"
        self.traces[name] = y
        while len(self.traces) > MAX_TRACES:
            self.traces.pop(next(iter(self.traces)))
        self.status = f"{name}: {y.size} frames"
        self.show()

    def remove(self, name: str) -> None:
        self.traces.pop(name, None)

    def clear(self) -> None:
        self.traces.clear()
        self.status = ""

    # ------------------------------------------------------------------
    # visibility
    # ------------------------------------------------------------------

    def show(self) -> None:
        if not self._closed:
            self.visible = True

    def hide(self) -> None:
        self.visible = False

    def close(self) -> None:
        self.hide()
        self.clear()
        self._closed = True

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------

    def _place(self) -> None:
        """First show: over the lower half of the canvas, clear of the edges."""
        w, h = self.size
        try:
            cw, ch = self.figure.canvas.get_logical_size()
        except Exception:
            cw, ch = 1200, 800
        x = max(8.0, (cw - w) / 2)
        y = max(8.0, ch - h - 90)
        imgui.set_next_window_pos(imgui.ImVec2(x, y), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(w, h), imgui.Cond_.first_use_ever)
        self._placed = True

    def _draw_cursor(self) -> None:
        if self.on_scrub is None:
            implot.plot_inf_lines(
                "##t",
                np.array([float(self.frame_marker)]),
                spec=implot.Spec(line_color=imgui.ImVec4(1, 1, 1, 0.5)),
            )
            return
        moved, frame = implot.drag_line_x(
            0, float(self.frame_marker), imgui.ImVec4(1.0, 0.85, 0.3, 0.9), 1.5
        )[:2]
        if moved:
            self.on_scrub(int(round(frame)))

    def draw(self) -> None:
        """Call every frame from a host update call; a no-op while hidden."""
        if not self.visible:
            return
        if not self._placed:
            self._place()
        expanded, opened = imgui.begin(f"{self.title}##trace_panel", True)
        try:
            if not opened:
                self.hide()
                return
            if not expanded:
                return
            self.draw_count += 1
            if imgui.small_button("clear"):
                self.clear()
            imgui.same_line()
            _, self.normalize = imgui.checkbox("normalize", self.normalize)
            imgui.same_line()
            imgui.text_disabled(self.status)

            if not self.traces:
                imgui.text_disabled("click a pixel, or the trace button of an ROI")
                return
            _ensure_implot_context()
            avail = imgui.get_content_region_avail()
            h = max(80.0, avail.y - 4)
            if implot.begin_plot("##traces", imgui.ImVec2(-1, h), implot.Flags_.no_title):
                try:
                    implot.setup_axes(
                        "frame", "value", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit
                    )
                    for k, (name, y) in enumerate(self.traces.items()):
                        if self.normalize:
                            lo, hi = float(y.min()), float(y.max())
                            y = (y - lo) / (hi - lo) if hi > lo else y * 0
                        r, g, b = _COLORS[k % len(_COLORS)]
                        implot.plot_line(
                            name,
                            np.ascontiguousarray(y, np.float32),
                            spec=implot.Spec(line_color=imgui.ImVec4(r, g, b, 1.0), line_weight=1.5),
                        )
                    if self.frame_marker is not None:
                        self._draw_cursor()
                finally:
                    implot.end_plot()
        finally:
            imgui.end()
