"""The viewer's tab bar, as widgets.

Every tab in the main tab bar is a ``placement = "tab"`` widget with an entry
in the Widgets menu. The bodies live where the work lives — the Preview and
Signal Quality tabs delegate to ``PreviewDataWidget``, Process to the pipelines
package, Cloud to ``gui/_cloud.py`` — so these classes are only the seam
between a tab and its panel.

Manual ROI labelling is split: its controls and trace plot hang off the
figure's top edge (``gui/manual_roi.py``), while its ROI table and trace
table are the ROIs and Traces tabs here, beside Image and Signal Quality.
The top panel's tab selection and these tabs follow each other. Runs report
through the process manager (the status button and its console), not a tab
of their own.

Tab order is ``priority``; the viewer draws them in that order.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui, imgui_ctx

from mbo_utilities.gui.widgets._base import Widget

__all__ = [
    "PreviewTabWidget",
    "RoiTableTabWidget",
    "TraceTableTabWidget",
    "RunTabWidget",
    "SignalQualityTabWidget",
]


class PreviewTabWidget(Widget):
    """The Image tab: every panel widget, stacked."""

    name = "Image"
    tab_label = "Image"
    placement = "tab"
    toggle_key = "preview"
    priority = 10

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return True

    def draw(self) -> None:
        # draw_preview_section opens its own child window
        self.parent.draw_preview_section()


def _strip(parent: Any):
    """The figure's top strip, or None when this parent has none."""
    return getattr(parent, "top_strip", None)


class SignalQualityTabWidget(Widget):
    """The Signal Quality tab: the metric table, once the z-stats have been
    computed. Its plot is the top strip's Signal Quality panel, which has the
    canvas's full width; the two selections follow each other."""

    name = "Signal Quality"
    tab_label = "Signal Quality"
    placement = "tab"
    toggle_key = "signal_quality"
    priority = 20

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return True

    def tab_disabled(self) -> str | None:
        if all(self.parent._zstats_done):
            return None
        return ""

    def wants_focus(self) -> bool:
        strip = _strip(self.parent)
        return strip is not None and strip.take_right_focus("signal_quality")

    def draw(self) -> None:
        strip = _strip(self.parent)
        if strip is not None:
            strip.report_right_tab("signal_quality")
        with imgui_ctx.begin_child(
            "##StatsContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
        ):
            self.parent.draw_stats_section()


class RunTabWidget(Widget):
    """The Process tab: registration / segmentation pipelines."""

    name = "Process"
    tab_label = "Process"
    placement = "tab"
    toggle_key = "run"
    priority = 30

    def __init__(self, parent: Any):
        super().__init__(parent)
        self._has_pipeline: bool | None = None

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return True

    def _pipeline_available(self) -> bool:
        if self._has_pipeline is None:
            from mbo_utilities.gui.widgets.pipelines import any_pipeline_available

            self._has_pipeline = any_pipeline_available()
        return self._has_pipeline

    def tab_disabled(self) -> str | None:
        if self._pipeline_available():
            return None
        # Show each registered pipeline's install command so the user can pick
        # whichever applies to their workflow, instead of hard-coding Suite2p.
        from mbo_utilities.gui.widgets.pipelines import get_available_pipelines

        pipelines = get_available_pipelines()
        if not pipelines:
            return "No pipelines registered.\nInstall with: uv pip install mbo_utilities"
        lines = ["No pipeline is installed.\nInstall one of:"]
        for cls in pipelines:
            lines.append(f"  {cls.name}: {cls.install_command}")
        return "\n".join(lines)

    def wants_focus(self) -> bool:
        """One-shot programmatic focus, used by scripts/capture_docs.py."""
        if getattr(self.parent, "_force_run_tab", False):
            self.parent._force_run_tab = False
            return True
        return False

    def draw(self) -> None:
        from mbo_utilities.gui.widgets.pipelines import draw_run_tab

        with imgui_ctx.begin_child(
            "##RunContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
        ):
            draw_run_tab(self.parent)


class RoiTableTabWidget(Widget):
    """The ROIs tab: the manual-ROI table, with its per-row trace actions.

    The panel itself is built by ``PreviewDataWidget.sync_manual_roi`` when
    the menu entry is switched on; the ROI *controls* stay in the top edge
    window, so only the table lives here.
    """

    name = "ROIs"
    tab_label = "ROIs"
    placement = "tab"
    # hidden when either Manual ROI Labeling or its ROI-table section is off
    toggle_key = "manual_roi.table"
    priority = 45

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return True

    def tab_disabled(self) -> str | None:
        if getattr(self.parent, "manual_roi", None) is None:
            return "Enable Widgets > Manual ROI Labeling to draw ROIs."
        return None

    def wants_focus(self) -> bool:
        # one-shot programmatic focus (menu toggle, --widget manualroi,
        # or the top panel switching to its ROI tab)
        roi = getattr(self.parent, "manual_roi", None)
        if roi is None:
            return False
        if getattr(roi, "focus_tab", False):
            roi.focus_tab = False
            return True
        if getattr(roi, "_focus_right", None) == "rois":
            roi._focus_right = None
            return True
        return False

    def draw(self) -> None:
        roi = getattr(self.parent, "manual_roi", None)
        if roi is None:
            imgui.text_disabled("Manual ROI Labeling is off.")
            return
        roi._right_tab_now = "rois"
        # lazy: manual_roi pulls in masknmf, only needed once it is on
        from mbo_utilities.gui._imgui_helpers import fit_width
        from mbo_utilities.gui.manual_roi import MIN_TAB_WIDTH

        with imgui_ctx.begin_child(
            "##RoiTableContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
        ):
            with fit_width("ROI table", min_width=MIN_TAB_WIDTH) as shown:
                if shown:
                    roi.draw_tab()


class TraceTableTabWidget(Widget):
    """The Traces tab: every collected trace with stats; the selection here
    is what the top panel's Traces tab plots."""

    name = "Traces"
    tab_label = "Traces"
    placement = "tab"
    toggle_key = "manual_roi.traces"
    priority = 46

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return True

    def tab_disabled(self) -> str | None:
        roi = getattr(self.parent, "manual_roi", None)
        if roi is None:
            return "Enable Widgets > Manual ROI Labeling to draw ROIs."
        if not roi.has_traces():
            return "No traces yet: use the trace button on a row of the ROIs tab, or run extract / demix."
        return None

    def wants_focus(self) -> bool:
        roi = getattr(self.parent, "manual_roi", None)
        if roi is not None and getattr(roi, "_focus_right", None) == "traces":
            roi._focus_right = None
            return True
        return False

    def draw(self) -> None:
        roi = getattr(self.parent, "manual_roi", None)
        if roi is None:
            imgui.text_disabled("Manual ROI Labeling is off.")
            return
        roi._right_tab_now = "traces"
        with imgui_ctx.begin_child("##TraceTableContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none):
            roi.draw_trace_table()


