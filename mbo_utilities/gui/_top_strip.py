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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui

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

    def _want_size(self) -> int:
        """Menu row plus the selected panel.

        The strip is as tall as what it is *showing*, not as tall as the
        tallest thing registered, and it grows with the window: on a tall
        canvas the panel takes a share of the height instead of leaving a
        band of empty space under its cards.
        """
        if not self.panels:
            return MENU_HEIGHT
        panel = self._panel(self.active)
        height = panel.height if panel is not None else max(p.height for p in self.panels)
        try:
            canvas_height = float(self.figure.canvas.get_logical_size()[1])
        except Exception:
            canvas_height = 0.0
        if canvas_height > 0:
            height = max(height, min(canvas_height * GROW_SHARE, height * GROW_MAX))
        return MENU_HEIGHT + int(height) + PANEL_PAD

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
