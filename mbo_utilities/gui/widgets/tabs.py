"""The viewer's tab bar, as widgets.

Every tab in the main tab bar is a ``placement = "tab"`` widget with an entry
in the Widgets menu. The bodies live where the work lives — the Preview and
Signal Quality tabs delegate to ``PreviewDataWidget``, Run to the pipelines
package, Cloud to ``gui/_cloud.py`` — so these classes are only the seam
between a tab and its panel.

Manual ROI labelling is deliberately not here: it hangs its own top and left
edge windows off the figure (``gui/manual_roi.py``), the way masknmf's
``ClassificationVis`` does.

Tab order is ``priority``; the viewer draws them in that order.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui, imgui_ctx

from mbo_utilities.gui.widgets._base import Widget

__all__ = [
    "CloudTabWidget",
    "PreviewTabWidget",
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
