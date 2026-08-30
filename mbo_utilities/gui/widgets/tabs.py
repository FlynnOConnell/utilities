"""The viewer's tab bar, as widgets.

Every tab in the main tab bar is a ``placement = "tab"`` widget with an entry
in the Widgets menu. The bodies live where the work lives — the Preview and
Signal Quality tabs delegate to ``PreviewDataWidget``, Run to the pipelines
package, Cloud to ``gui/_cloud.py`` — so these classes are only the seam
between a tab and its panel.

Manual ROI labelling is split: its controls hang off the figure's top edge
(``gui/manual_roi.py``), while its ROI table and trace viewer are the ROIs
and Traces tabs here, beside Preview and Signal Quality.

Tab order is ``priority``; the viewer draws them in that order.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui, imgui_ctx

from mbo_utilities.gui.widgets._base import Widget

__all__ = [
    "CloudTabWidget",
    "PreviewTabWidget",
    "RoiTableTabWidget",
    "RoiTracesTabWidget",
    "RunTabWidget",
    "SignalQualityTabWidget",
]


class PreviewTabWidget(Widget):
    """The Preview tab: every panel widget, stacked."""

    name = "Preview"
    tab_label = "Preview"
    placement = "tab"
    toggle_key = "preview"
    priority = 10

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return True

    def draw(self) -> None:
        # draw_preview_section opens its own child window
        self.parent.draw_preview_section()


class SignalQualityTabWidget(Widget):
    """The Signal Quality tab: z-stats plots, once they have been computed."""

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

    def draw(self) -> None:
        with imgui_ctx.begin_child(
            "##StatsContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
        ):
            self.parent.draw_stats_section()


class RunTabWidget(Widget):
    """The Run tab: registration / segmentation pipelines."""

    name = "Run"
    tab_label = "Run"
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
        # one-shot programmatic focus (menu toggle, --widget manualroi)
        roi = getattr(self.parent, "manual_roi", None)
        if roi is not None and getattr(roi, "focus_tab", False):
            roi.focus_tab = False
            return True
        return False

    def draw(self) -> None:
        roi = getattr(self.parent, "manual_roi", None)
        if roi is None:
            imgui.text_disabled("Manual ROI Labeling is off.")
            return
        # lazy: manual_roi pulls in masknmf, only needed once it is on
        from mbo_utilities.gui._imgui_helpers import fit_width
        from mbo_utilities.gui.manual_roi import MIN_TAB_WIDTH

        with imgui_ctx.begin_child(
            "##RoiTableContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
        ):
            with fit_width("ROI table", min_width=MIN_TAB_WIDTH) as shown:
                if shown:
                    roi.draw_tab()


class RoiTracesTabWidget(Widget):
    """The Traces tab: one ROI's traces from the manual-ROI widget."""

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
        if not roi.traces:
            return "No traces yet: use the trace button on a row of the ROIs tab, or run extract / demix."
        return None

    def wants_focus(self) -> bool:
        roi = getattr(self.parent, "manual_roi", None)
        if roi is not None and getattr(roi, "focus_traces", False):
            roi.focus_traces = False
            return True
        return False

    def draw(self) -> None:
        roi = getattr(self.parent, "manual_roi", None)
        if roi is None:
            imgui.text_disabled("Manual ROI Labeling is off.")
            return
        with imgui_ctx.begin_child("##RoiTracesContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none):
            roi.draw_traces()


class CloudTabWidget(Widget):
    """The Cloud tab: run the loaded dataset on an ephemeral cloud GPU."""

    name = "Cloud (GPU)"
    tab_label = "Cloud"
    placement = "tab"
    toggle_key = "cloud"
    priority = 90

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return True

    def draw(self) -> None:
        from mbo_utilities.gui._cloud import draw_cloud_tab

        # horizontal scrollbar: over-wide content scrolls, never clips
        with imgui_ctx.begin_child(
            "##CloudContent",
            imgui.ImVec2(0, 0),
            imgui.ChildFlags_.none,
            imgui.WindowFlags_.horizontal_scrollbar,
        ):
            draw_cloud_tab(self.parent, self.parent.fpath)
