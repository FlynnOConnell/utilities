"""The figure's top edge, shared by everything that wants the full canvas width.

fastplotlib keeps one imgui window per edge, so the menu row, Manual ROI's
control cards and the Signal Quality plot cannot each own the top. They
register a :class:`TopPanel` here instead: the strip draws the menu row,
then a tab bar over whatever is registered, and shrinks back to the menu row
alone when nothing is. A panel names the right-bar tab it belongs with, so
selecting one selects the other (``report_right_tab`` / ``take_right_focus``).

Work a feature needs every frame regardless of which tab is on top — polling
jobs, keyboard handling, its own floating windows — goes in a frame hook, not
in a panel body.

A grab bar along the strip's bottom edge drags it taller or shorter and
double-clicks shut, the way fastplotlib's right and bottom edge windows work
(it does not draw one for the top edge, so the strip draws its own). Dragging
pins the height until :meth:`TopStrip.reset_size`; until then the strip sizes
itself to whatever panel is showing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui, imgui_ctx

__all__ = ["MENU_HEIGHT", "TopPanel", "TopStrip"]

# the menu row on its own; a registered panel adds its height below it
MENU_HEIGHT = 42
# tab bar + spacing around a panel body
PANEL_PAD = 22
# frames to ignore right-bar reports for after the top switches: the right bar
# redraws its old tab for a frame or two before it follows
SYNC_HOLD = 3

# a tall window gives the strip more room rather than handing every pixel to
# the canvas: the panel takes this share of the canvas height, never less than
# the height it asked for and never more than twice it
GROW_SHARE = 0.28
GROW_MAX = 2.0

# never drag the strip so far down that the canvas has less than this
MIN_RENDER_AREA = 150


@dataclass
class TopPanel:
    """One tab in the strip.

    Parameters
    ----------
    key : str
        Identity, for :meth:`TopStrip.focus` and :meth:`TopStrip.unregister`.
    label : str
        Tab caption.
    draw : callable
        Body, called only while the tab is selected.
    height : int
        Pixels the body wants; the strip sizes itself to the tallest panel.
    right_tab : str, optional
        The right-bar tab this panel pairs with, for two-way selection sync.
    priority : int
        Tab order, lower first.
    """

    key: str
    label: str
    draw: Callable[[], None]
    height: int = 200
    right_tab: str | None = None
    priority: int = 100


class TopStrip(ImguiWindow):
    """The top edge window: a menu row, then the registered panels as tabs."""

    def __init__(self, figure, draw_menu: Callable[[], None] | None = None):
        super().__init__()
        self.figure = figure
        self.draw_menu = draw_menu
        self.panels: list[TopPanel] = []
        self.hooks: list[Callable[[], None]] = []
        self.active: str | None = None
        self._focus: str | None = None
        self._right_now = ""
        self._right_last = ""
        self._right_focus: str | None = None
        self._hold = 0
        self._closed = False
        # our own resize state: ImguiWindow._collapsed drives its edge-window
        # drawing, so shutting the strip must not touch it
        self._shut = False
        self._manual: int | None = None
        self._before_collapse: int | None = None
        self._cursor_set = False
        figure.add_imgui_window(self, location="top", size=self._want_size(), title=None)

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def register(self, panel: TopPanel) -> None:
        """Add or replace the panel with this key."""
        self.panels = sorted(
            [p for p in self.panels if p.key != panel.key] + [panel],
            key=lambda p: (p.priority, p.label),
        )
        if self.active is None:
            self.active = panel.key
        self._resize()

    def unregister(self, key: str) -> None:
        if not any(p.key == key for p in self.panels):
            return
        self.panels = [p for p in self.panels if p.key != key]
        if self.active == key:
            self.active = self.panels[0].key if self.panels else None
        self._resize()

    def has(self, key: str) -> bool:
        return any(p.key == key for p in self.panels)

    def add_hook(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` at the top of every frame, whatever tab is selected."""
        if fn not in self.hooks:
            self.hooks.append(fn)

    def remove_hook(self, fn: Callable[[], None]) -> None:
        if fn in self.hooks:
            self.hooks.remove(fn)

    def close(self) -> None:
        """Give the top edge back to the figure."""
        if self._closed:
            return
        self._closed = True
        if self.figure.imgui_windows.get("top") is self:
            self.figure.remove_imgui_window("top")

    # ------------------------------------------------------------------
    # selection, and its sync with the right bar
    # ------------------------------------------------------------------

    def focus(self, key: str) -> None:
        """Select ``key`` on the next frame."""
        if self.has(key):
            self._focus = key

    def report_right_tab(self, name: str) -> None:
        """Called by a right-bar tab body as it draws, to say it is on top."""
        self._right_now = name

    def take_right_focus(self, name: str) -> bool:
        """True once when the right bar should select ``name`` to follow us."""
        if self._right_focus != name:
            return False
        self._right_focus = None
        return True

    def _panel(self, key: str | None) -> TopPanel | None:
        return next((p for p in self.panels if p.key == key), None)

    @property
    def handle_height(self) -> int:
        """Thickness of the grab bar along the bottom edge."""
        return int(self._separator_thickness)

    @property
    def shut_size(self) -> int:
        """Height of the strip with the panels shut: the menu row and the bar."""
        return MENU_HEIGHT + self.handle_height

    @property
    def collapsed(self) -> bool:
        """True while the panels are shut and only the menu row shows."""
        return self._shut

    def toggle_collapsed(self) -> None:
        """Shut the panels away, or bring them back at the previous height."""
        if self._shut:
            self._shut = False
            self._manual, self._before_collapse = self._before_collapse, None
        else:
            self._before_collapse = self._manual
            self._shut = True
        self._resize()

    def resize_to(self, size: int) -> None:
        """Hold the strip at ``size`` px instead of sizing it to its panel."""
        self._shut = False
        self._manual = max(int(size), self.shut_size)
        self._resize()

    def reset_size(self) -> None:
        """Size the strip to whatever panel is showing again."""
        self._shut = False
        self._manual = self._before_collapse = None
        self._resize()

    def _want_size(self) -> int:
        """Menu row plus the selected panel, unless the user set a height.

        The strip is as tall as what it is *showing*, not as tall as the
        tallest thing registered, and it grows with the window: on a tall
        canvas the panel takes a share of the height instead of leaving a
        band of empty space under its cards. A drag on the grab bar pins the
        height until :meth:`reset_size`.
        """
        if not self.panels:
            return MENU_HEIGHT
        if self._shut:
            return self.shut_size
        if self._manual is not None:
            return self._manual
        panel = self._panel(self.active)
        height = panel.height if panel is not None else max(p.height for p in self.panels)
        try:
            canvas_height = float(self.figure.canvas.get_logical_size()[1])
        except Exception:
            canvas_height = 0.0
        if canvas_height > 0:
            height = max(height, min(canvas_height * GROW_SHARE, height * GROW_MAX))
        return MENU_HEIGHT + int(height) + PANEL_PAD + self.handle_height

    def _resize(self) -> None:
        want = self._want_size()
        if self.size != want:
            self.size = want

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------

    def update(self) -> None:
        # the selected panel sets the height; resize before drawing so the
        # rect the figure laid out this frame is the one we fill
        self._resize()
        for hook in list(self.hooks):
            hook()
        if self.draw_menu is not None:
            self.draw_menu()
        if not self.panels:
            return
        if not self._shut:
            # the tabs live above the grab bar, never under it
            with imgui_ctx.begin_child(
                "##strip_body", imgui.ImVec2(0, -float(self.handle_height))
            ):
                self._draw_tabs()
        self._draw_handle()

    def _draw_handle(self) -> None:
        """The grab bar along the bottom edge: drag to resize, double click
        to shut or reopen the panels."""
        thickness = float(self.handle_height)
        imgui.set_cursor_pos(
            imgui.ImVec2(0.0, imgui.get_window_height() - thickness)
        )
        imgui.invisible_button(
            "##top_resize", imgui.ImVec2(imgui.get_window_width(), thickness)
        )
        hovered, active = imgui.is_item_hovered(), imgui.is_item_active()
        rect_min, rect_max = imgui.get_item_rect_min(), imgui.get_item_rect_max()

        if hovered and imgui.is_mouse_double_clicked(0):
            self.toggle_collapsed()

        if hovered or active:
            if not self._cursor_set:
                self._set_cursor("ns_resize")
                self._cursor_set = True
            imgui.set_tooltip("Drag to resize, double click to expand/collapse")
        elif self._cursor_set:
            self._set_cursor("default")
            self._cursor_set = False

        if active and imgui.is_mouse_dragging(0):
            # the bar is the strip's bottom edge, so dragging down grows it -
            # but never past leaving the canvas too little to render into
            delta = imgui.get_mouse_drag_delta(0).y
            imgui.reset_mouse_drag_delta(0)
            if delta > 0 and self._render_height() - delta < MIN_RENDER_AREA:
                delta = 0.0
            if delta:
                self.resize_to(round(self.size + delta))

        draw = imgui.get_window_draw_list()
        strong = hovered or active
        line = imgui.get_color_u32(
            imgui.ImVec4(0.9, 0.9, 0.9, 1.0) if strong
            else imgui.ImVec4(0.5, 0.5, 0.5, 0.8)
        )
        draw.add_rect_filled(
            rect_min, rect_max,
            imgui.get_color_u32(
                imgui.ImVec4(0.2, 0.2, 0.2, 0.8) if strong
                else imgui.ImVec4(0.15, 0.15, 0.15, 0.6)
            ),
        )
        mid_y = (rect_min.y + rect_max.y) * 0.5
        center_x = (rect_min.x + rect_max.x) * 0.5
        for i in (-1, 0, 1):
            draw.add_circle_filled(
                imgui.ImVec2(center_x + i * 7.0, mid_y), 2, line
            )

    def _set_cursor(self, name: str) -> None:
        try:
            self.figure.canvas.set_cursor(name)
        except Exception:
            pass

    def _render_height(self) -> float:
        """Canvas height left for pygfx, or a big number when unknown."""
        try:
            return float(self.figure.get_pygfx_render_area()[3])
        except Exception:
            return float("inf")

    def _draw_tabs(self) -> None:
        focus, self._focus = self._focus, None
        # a fresh report from the right bar means the user switched over there
        if self._right_now and self._right_now != self._right_last:
            self._right_last = self._right_now
            pair = next(
                (p.key for p in self.panels if p.right_tab == self._right_now), None
            )
            if (
                pair is not None
                and pair != self.active
                and focus is None
                and imgui.get_frame_count() >= self._hold
            ):
                focus = pair

        before = self.active
        if imgui.begin_tab_bar("##top_strip_tabs"):
            active = None
            for panel in self.panels:
                flags = (
                    imgui.TabItemFlags_.set_selected
                    if focus == panel.key
                    else imgui.TabItemFlags_.none
                )
                if imgui.begin_tab_item(panel.label, None, flags)[0]:
                    active = panel.key
                    panel.draw()
                    imgui.end_tab_item()
            imgui.end_tab_bar()
            if active is not None:
                self.active = active

        if self.active != before:
            panel = self._panel(self.active)
            if panel is not None and panel.right_tab and self._right_last != panel.right_tab:
                self._right_focus = panel.right_tab
                self._hold = imgui.get_frame_count() + SYNC_HOLD
