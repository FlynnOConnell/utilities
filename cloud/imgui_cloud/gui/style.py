"""Shared drawing vocabulary: cards, section headers, field rows, buttons.

Sizes are in ``em`` rather than pixels, so the panel keeps its proportions on a
laptop screen and a 4K one. Colors come from the imgui_data_loader theme, which
is the same palette the file dialog and the viewer use.
"""

from __future__ import annotations

from contextlib import contextmanager

from imgui_bundle import hello_imgui, icons_fontawesome_6 as fa
from imgui_bundle import imgui, imgui_ctx
from imgui_data_loader import Theme, pop_button_style, push_button_style, to_vec4

WIDTH_LABEL_EM = 7.5
ROUNDING_CARD = 8.0


def em(size: float) -> float:
    """Pixels for ``size`` text-heights at the current font size."""
    return hello_imgui.em_size(size)


def em2(x: float, y: float) -> imgui.ImVec2:
    """An ``ImVec2`` in text-heights."""
    return hello_imgui.em_to_vec2(x, y)


def icon(name: str) -> str:
    """FontAwesome glyph by constant name, empty when this build lacks it."""
    return getattr(fa, name, "")


def wrapped(text: str, color) -> None:
    """Colored text that wraps at the panel's visible right edge."""
    imgui.push_text_wrap_pos(
        imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
    )
    imgui.text_colored(to_vec4(color), text)
    imgui.pop_text_wrap_pos()


def desc(theme: Theme, text: str) -> None:
    """Indented, dim helper line beneath a control."""
    imgui.indent(em(0.5))
    wrapped(text, theme.text_dim)
    imgui.unindent(em(0.5))


def center_text(text: str, color) -> None:
    """One centered line, falling back to wrapping when it does not fit."""
    width = imgui.get_content_region_avail().x
    width_text = imgui.calc_text_size(text).x
    if width_text >= width:
        wrapped(text, color)
        return
    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (width - width_text) * 0.5)
    imgui.text_colored(to_vec4(color), text)


def section(theme: Theme, glyph: str, title: str, hint: str = "") -> None:
    """An accent-colored section header with a rule under it."""
    imgui.dummy(em2(0, 0.3))
    imgui.text_colored(to_vec4(theme.accent), f"{glyph}  {title}" if glyph else title)
    if hint:
        imgui.same_line()
        imgui.text_colored(to_vec4(theme.text_dim), hint)
    imgui.separator()
    imgui.dummy(em2(0, 0.2))


def field_row(label: str, tooltip: str = "", width_label: float = WIDTH_LABEL_EM):
    """Dim label left of a full-width input, stacking when the panel is narrow."""
    imgui.align_text_to_frame_padding()
    imgui.text(label)
    if tooltip and imgui.is_item_hovered():
        imgui.set_tooltip(tooltip)
    column = max(
        em(width_label),
        imgui.calc_text_size(label).x + imgui.get_style().item_spacing.x * 2,
    )
    if imgui.get_content_region_avail().x - column >= em(8):
        imgui.same_line(column)
    imgui.set_next_item_width(-1)


def status_glyph(theme: Theme, ok, unknown_dim: bool = True):
    """Tick / cross / hollow circle for a tri-state ``ok`` (True, False, None)."""
    if ok is True:
        return icon("ICON_FA_CIRCLE_CHECK"), theme.ok
    if ok is None:
        return icon("ICON_FA_CIRCLE"), theme.text_dim if unknown_dim else theme.warn
    return icon("ICON_FA_CIRCLE_XMARK"), theme.err


@contextmanager
def button_style(theme: Theme, primary: bool = False):
    """The theme's filled (primary) or quiet button, as a block."""
    push_button_style(theme, primary=primary)
    try:
        yield
    finally:
        pop_button_style()


@contextmanager
def card(theme: Theme, name: str, width: float = 0.0):
    """A rounded, bordered panel that grows to fit whatever is drawn in it."""
    imgui.push_style_color(imgui.Col_.child_bg, to_vec4(theme.bg_card))
    imgui.push_style_var(imgui.StyleVar_.child_rounding, ROUNDING_CARD)
    imgui.push_style_var(imgui.StyleVar_.window_padding, em2(0.8, 0.6))
    try:
        with imgui_ctx.begin_child(
            name,
            size=imgui.ImVec2(width, 0),
            child_flags=imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y,
            window_flags=imgui.WindowFlags_.no_scrollbar,
        ):
            yield
    finally:
        imgui.pop_style_var(2)
        imgui.pop_style_color(1)


@contextmanager
def card_centered(theme: Theme, name: str, width: float = 26.0):
    """A card of ``width`` em, centered in whatever space is available."""
    available = imgui.get_content_region_avail().x
    width_card = min(available, em(width))
    imgui.set_cursor_pos_x(
        imgui.get_cursor_pos_x() + max(0.0, (available - width_card) * 0.5)
    )
    with card(theme, name, width=width_card):
        yield
