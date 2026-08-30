"""Cloud tab - run the loaded dataset on an ephemeral Google Cloud A100.

A thin adapter over the ``imgui_cloud`` package, which owns the whole workflow
(sign-in, upload, provisioning, teardown) and is usable on its own:

    uv pip install git+https://github.com/FlynnOConnell/imgui_cloud.git
    imgui-cloud gui

This file only bridges the viewer to it: it hands over the path of whatever is
loaded and keeps the panel alive between frames. When the package is not
installed the tab explains how to get it instead of raising, mirroring how the
Run tab handles a missing pipeline.
"""

from __future__ import annotations

from pathlib import Path

from imgui_bundle import imgui

__all__ = ["draw_cloud_tab"]

_INSTALL_HINT = "uv pip install git+https://github.com/FlynnOConnell/imgui_cloud.git"


def _dir_for(fpath) -> str:
    """Folder to upload for the loaded dataset (a file yields its parent)."""
    if not fpath:
        return ""
    first = fpath[0] if isinstance(fpath, (list, tuple)) and fpath else fpath
    path = Path(first)
    if path.is_file():
        return str(path.parent)
    return str(path)


def draw_cloud_tab(parent, fpath=None) -> None:
    """Draw the cloud panel for ``parent``, creating it on first use.

    Parameters
    ----------
    parent : object
        Widget that owns the tab; the panel is cached on it so its sign-in and
        run state survive across frames.
    fpath : str | list, optional
        Dataset currently loaded, used to pre-fill the upload folder.
    """
    try:
        from imgui_cloud.gui import draw_cloud_tab as draw_panel
    except ImportError:
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.75, 0.3, 1.0), "imgui_cloud is not installed"
        )
        imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), _INSTALL_HINT)
        if imgui.button("Copy install command"):
            imgui.set_clipboard_text(_INSTALL_HINT)
        return

    state = getattr(parent, "_cloud_panel_state", None)
    if state is None:
        state = {}
        parent._cloud_panel_state = state
    draw_panel(state, dir_input=_dir_for(fpath))
