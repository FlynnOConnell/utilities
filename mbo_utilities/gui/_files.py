"""Picking a path without depending on a native file dialog.

A dialog opens where the process runs, so a viewer driven from a notebook on
another machine has nowhere to draw one, and a linux box without zenity or
kdialog on PATH cannot draw one at all. Every path the user has to give goes
through :func:`draw_path_prompt`: a typed field first, with the native dialog
as a browse shortcut that is disabled when nothing can draw it.

Mirrors ``masknmf.visualization.imgui.files`` so the two GUIs behave the
same way on the same machine.
"""

from __future__ import annotations

import os
import shutil
import socket
import sys

from imgui_bundle import hello_imgui, imgui

__all__ = [
    "NATIVE_DIALOGS",
    "PathPrompt",
    "draw_path_prompt",
    "native_dialogs_available",
    "no_dialog_hint",
]


def native_dialogs_available() -> bool:
    """Whether a file dialog can appear on the machine this process runs on.

    Windows and macOS always have one. Linux needs a display and one of the
    helpers the portable-file-dialogs backend shells out to.
    """
    if sys.platform in ("win32", "darwin"):
        return True
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    return any(
        shutil.which(name) for name in ("zenity", "matedialog", "qarma", "kdialog")
    )


NATIVE_DIALOGS = native_dialogs_available()


def no_dialog_hint() -> str:
    """Tooltip for a disabled browse button."""
    return "no file dialog on this machine; type the path instead"


class PathPrompt:
    """State for one path popup: whether it is open, the path, and what it said.

    Parameters
    ----------
    title : str
        popup title
    path : str
        initial path
    action : str
        label of the button that accepts the path
    hint : str
        line under the title saying what the path is for
    """

    def __init__(self, title: str, path: str = "", action: str = "open", hint: str = ""):
        self.title = title
        self.path = path
        self.action = action
        self.hint = hint
        self.open = False
        self.status = ""

    def start(self, path: str | None = None) -> None:
        """Open the popup, optionally on a different path."""
        if path is not None:
            self.path = str(path)
        self.status = ""
        self.open = True


def draw_path_prompt(prompt: PathPrompt) -> tuple[str | None, bool]:
    """Draw one path popup.

    Returns ``(submitted_path, browse_clicked)``: the path when the action
    button or enter was pressed, else None, and whether the browse shortcut
    was pressed. The caller closes the popup by setting ``prompt.open``, so a
    failed action can leave it up with a message in ``prompt.status``.
    """
    if not prompt.open:
        return None, False
    em = hello_imgui.em_size
    imgui.set_next_window_pos(
        imgui.get_main_viewport().get_center(),
        imgui.Cond_.appearing,
        pivot=imgui.ImVec2(0.5, 0.5),
    )
    imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(em(1.0), em(0.8)))
    opened, prompt.open = imgui.begin(
        f"{prompt.title}###path-prompt-{prompt.title}",
        prompt.open,
        flags=imgui.WindowFlags_.no_saved_settings
        | imgui.WindowFlags_.always_auto_resize
        | imgui.WindowFlags_.no_collapse,
    )
    imgui.pop_style_var()
    submitted, browse = None, False
    if opened:
        imgui.text_disabled(
            f"{prompt.hint or 'read by this process'}, on {socket.gethostname()}"
        )
        imgui.set_next_item_width(em(28))
        entered, prompt.path = imgui.input_text(
            f"##path-{prompt.title}", prompt.path, imgui.InputTextFlags_.enter_returns_true
        )
        if imgui.is_window_appearing():
            imgui.set_keyboard_focus_here(-1)
        if imgui.button(prompt.action, imgui.ImVec2(em(6), 0)) or entered:
            submitted = prompt.path.strip().strip('"')
        imgui.same_line(0, em(0.5))
        if not NATIVE_DIALOGS:
            imgui.begin_disabled()
        if imgui.button("browse", imgui.ImVec2(em(6), 0)):
            browse = True
        if not NATIVE_DIALOGS:
            imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "pick a path in a file dialog" if NATIVE_DIALOGS else no_dialog_hint()
            )
        imgui.same_line(0, em(0.5))
        if imgui.button("close", imgui.ImVec2(em(6), 0)):
            prompt.open = False
        if prompt.status:
            imgui.text_disabled(prompt.status)
    imgui.end()
    return submitted, browse
