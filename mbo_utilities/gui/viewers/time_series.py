"""Time series viewer for standard calcium imaging data.

Renders the main tab bar. Every tab is a ``placement = "tab"`` widget (see
``gui/widgets/tabs.py``), so this viewer only orders them, honours the
Widgets menu, and keeps one tab's failure from taking the GUI down.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imgui_bundle import imgui, imgui_ctx  # noqa: F401  (imgui_ctx: tab bodies)

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
        self._tab_widgets: list | None = None

    def draw(self) -> None:
        """Draw the main tab bar."""
        if imgui.begin_tab_bar("MainPreviewTabs"):
            for widget in self._widgets():
                self._draw_tab(widget)
            imgui.end_tab_bar()

    def _widgets(self) -> list:
        """Supported tab widgets, in priority order (built once)."""
        if self._tab_widgets is None:
            from mbo_utilities.gui.widgets import get_tab_widgets

            self._tab_widgets = get_tab_widgets(self.parent)
        return self._tab_widgets

    def _draw_tab(self, widget) -> None:
        """Draw one tab, unless the Widgets menu has it switched off."""
        from mbo_utilities.gui.widgets import widget_is_visible

        if not widget_is_visible(widget):
            return

        label = widget.tab_label or widget.name
        reason = widget.tab_disabled()
        if reason is not None:
            imgui.begin_disabled()

        flags = (
            imgui.TabItemFlags_.set_selected
            if widget.wants_focus()
            else imgui.TabItemFlags_.none
        )
        if imgui.begin_tab_item(label, None, flags)[0]:
            imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
            imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))
            try:
                widget.draw()
            except Exception as e:
                self.logger.exception(f"{label} tab failed")
                imgui.push_text_wrap_pos(
                    imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
                )
                imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), f"Error: {e}")
                imgui.pop_text_wrap_pos()
            finally:
                imgui.pop_style_var(2)
            imgui.end_tab_item()

        if reason is not None:
            imgui.end_disabled()
            if reason and imgui.is_item_hovered(
                imgui.HoveredFlags_.allow_when_disabled
            ):
                imgui.set_tooltip(reason)
