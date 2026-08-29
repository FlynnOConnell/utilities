"""Time series viewer for standard calcium imaging data.

Renders the Preview / Signal Quality / Run tab bar by delegating to
the parent PreviewDataWidget, which owns the actual state and draw
methods. A "ROIs" tab joins them while the manual ROI widget is on
(``Widgets > ROIs``); BioHPC lives in a popup off the same menu.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imgui_bundle import imgui, imgui_ctx

from mbo_utilities.gui._imgui_helpers import fit_width

from . import BaseViewer

if TYPE_CHECKING:
    from fastplotlib.widgets import ImageWidget

__all__ = ["TimeSeriesViewer"]


class TimeSeriesViewer(BaseViewer):
    """Viewer for time-series calcium imaging data (TZYX)."""

    name = "Time Series Viewer"

    def __init__(
        self,
        image_widget: ImageWidget,
        fpath: str | list[str],
        parent=None,
        **kwargs,
    ):
        super().__init__(image_widget, fpath, parent=parent, **kwargs)
        self._has_pipeline: bool | None = None

    def draw(self) -> None:
        """Draw the main tab bar, delegating to parent PreviewDataWidget."""
        if imgui.begin_tab_bar("MainPreviewTabs"):
            if imgui.begin_tab_item("Preview")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))
                try:
                    self.parent.draw_preview_section()
                finally:
                    imgui.pop_style_var(2)
                imgui.end_tab_item()

            imgui.begin_disabled(not all(self.parent._zstats_done))
            if imgui.begin_tab_item("Signal Quality")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))
                try:
                    with imgui_ctx.begin_child("##StatsContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none):
                        with fit_width():
                            self.parent.draw_stats_section()
                finally:
                    imgui.pop_style_var(2)
                imgui.end_tab_item()
            imgui.end_disabled()

            if self._has_pipeline is None:
                from mbo_utilities.gui.widgets.pipelines import any_pipeline_available
                self._has_pipeline = any_pipeline_available()
            if not self._has_pipeline:
                imgui.begin_disabled()
            # one-shot programmatic focus, used by scripts/capture_docs.py
            run_flags = imgui.TabItemFlags_.none
            if getattr(self.parent, "_force_run_tab", False):
                run_flags = imgui.TabItemFlags_.set_selected
                self.parent._force_run_tab = False
            if imgui.begin_tab_item("Run", None, run_flags)[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))
                try:
                    from mbo_utilities.gui.widgets.pipelines import draw_run_tab
                    with imgui_ctx.begin_child(
                        "##RunContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
                    ):
                        with fit_width():
                            draw_run_tab(self.parent)
                finally:
                    imgui.pop_style_var(2)
                imgui.end_tab_item()
            if not self._has_pipeline:
                imgui.end_disabled()
                if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                    # Show each registered pipeline's install command so the
                    # user can pick whichever applies to their workflow,
                    # instead of hard-coding Suite2p in the tooltip.
                    from mbo_utilities.gui.widgets.pipelines import (
                        get_available_pipelines,
                    )
                    pipelines = get_available_pipelines()
                    if pipelines:
                        lines = ["No pipeline is installed.\nInstall one of:"]
                        for cls in pipelines:
                            lines.append(f"  {cls.name}: {cls.install_command}")
                        imgui.set_tooltip("\n".join(lines))
                    else:
                        imgui.set_tooltip(
                            "No pipelines registered.\n"
                            "Install with: uv pip install mbo_utilities"
                        )

            # the manual ROI widget's table, present only while the widget
            # is on (Widgets > ROIs); its tools sit in the top edge panel
            roi = getattr(self.parent, "manual_roi", None)
            if roi is not None:
                # lazy: manual_roi pulls in masknmf, only needed once it is on
                from mbo_utilities.gui.manual_roi import MIN_TAB_WIDTH

                roi_flags = imgui.TabItemFlags_.none
                if roi.focus_tab:
                    # one-shot programmatic focus (menu toggle, --widget manualroi)
                    roi_flags = imgui.TabItemFlags_.set_selected
                    roi.focus_tab = False
                if imgui.begin_tab_item("ROIs", None, roi_flags)[0]:
                    imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
                    imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))
                    try:
                        with imgui_ctx.begin_child(
                            "##RoiContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none
                        ):
                            with fit_width("ROI table", min_width=MIN_TAB_WIDTH) as shown:
                                if shown:
                                    roi.draw_tab()
                    finally:
                        imgui.pop_style_var(2)
                    imgui.end_tab_item()

            imgui.end_tab_bar()
