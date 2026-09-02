"""The viewer as an object, and where it runs.

``DataVis`` is the masknmf-style contract: build with the data, ``show()``,
``close()``. ``run_gui`` sits on top and does the right thing for a terminal
(window + event loop) and a notebook (canvas in the cell, ``DataVis`` back).
The path prompts are what keep the running GUI usable from a kernel on
another machine, where a native dialog has nowhere to open.

``RENDERCANVAS_FORCE_OFFSCREEN=1`` must be set before *any* fastplotlib
import in the process — tests/conftest.py does this for the whole suite.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_manual_roi import FIGURE_SIZE, _offscreen_selected

pytestmark = pytest.mark.skipif(
    not _offscreen_selected(),
    reason="needs the offscreen rendercanvas",
)


def _array(shape=(4, 1, 3, 32, 32)):
    from mbo_utilities.arrays.numpy import NumpyArray

    data = np.random.default_rng(0).random(shape).astype(np.float32)
    return NumpyArray(data, dims="TCZYX")


# ----------------------------------------------------------------------
# where are we
# ----------------------------------------------------------------------


class TestInNotebook:
    """Only a kernel counts; ``ipython`` in a terminal wants a window."""

    @staticmethod
    def _shell(monkeypatch, shell):
        IPython = pytest.importorskip("IPython")
        monkeypatch.setattr(IPython, "get_ipython", lambda: shell)

    def test_no_shell(self, monkeypatch):
        from mbo_utilities.gui._notebook import in_notebook

        self._shell(monkeypatch, None)
        assert in_notebook() is False

    def test_terminal_ipython_is_not_a_notebook(self, monkeypatch):
        from mbo_utilities.gui._notebook import in_notebook

        TerminalInteractiveShell = type("TerminalInteractiveShell", (), {})
        self._shell(monkeypatch, TerminalInteractiveShell())
        assert in_notebook() is False

    def test_kernel_is(self, monkeypatch):
        from mbo_utilities.gui._notebook import in_notebook

        ZMQInteractiveShell = type("ZMQInteractiveShell", (), {})
        self._shell(monkeypatch, ZMQInteractiveShell())
        assert in_notebook() is True

    def test_old_name_still_answers(self, monkeypatch):
        from mbo_utilities.gui import run_gui as rg

        monkeypatch.setattr(rg, "in_notebook", lambda: True)
        assert rg._is_jupyter() is True


class TestFigureKwargsForHere:
    def test_notebook_gets_the_jupyter_canvas_with_room(self, monkeypatch):
        from mbo_utilities.gui import run_gui as rg

        monkeypatch.setattr(rg, "in_notebook", lambda: True)
        assert rg._figure_kwargs_for_here() == {"canvas": "jupyter", "size": (1400, 900)}
        assert rg._figure_kwargs_for_here(size=(800, 600))["size"] == (800, 600)

    def test_offscreen_env_leaves_the_canvas_to_rendercanvas(self, monkeypatch):
        from mbo_utilities.gui import run_gui as rg

        monkeypatch.setattr(rg, "in_notebook", lambda: False)
        monkeypatch.setenv("RENDERCANVAS_FORCE_OFFSCREEN", "1")
        kwargs = rg._figure_kwargs_for_here(size=(640, 480))
        assert "canvas" not in kwargs
        assert kwargs["size"] == (640, 480)


# ----------------------------------------------------------------------
# DataVis
# ----------------------------------------------------------------------


class TestDataVis:
    def test_build_show_close(self):
        from mbo_utilities.gui import DataVis
        from mbo_utilities.gui.widgets.preview_data import PreviewDataWidget

        vis = DataVis(_array(), size=FIGURE_SIZE)
        try:
            # the side widget is on before show, the way masknmf builds
            assert isinstance(vis.widget, PreviewDataWidget)
            assert "built" in repr(vis)
            out = vis.show()
            assert vis.show() is out, "show is idempotent"
            assert "shown" in repr(vis)
            # a frame renders with the widget attached pre-show
            for _ in range(2):
                vis.figure.canvas.draw()
            assert vis.iw is vis.image_widget
            assert tuple(vis.data.shape) == (4, 1, 3, 32, 32)
        finally:
            vis.close()
        assert vis.closed
        assert vis.iw._closed, "close reaches the NDViewer"
        vis.close()  # a second close is a no-op

    def test_widget_none_is_only_the_canvas(self):
        from mbo_utilities.gui import DataVis

        vis = DataVis(_array(), widget="none", size=FIGURE_SIZE)
        try:
            assert vis.widget is None
            assert "NumpyArray" in repr(vis)
        finally:
            vis.close()

    def test_size_reaches_the_canvas(self):
        from mbo_utilities.gui import DataVis

        vis = DataVis(_array(), widget="none", size=(640, 480))
        try:
            w, h = vis.figure.canvas.get_logical_size()
            assert (int(w), int(h)) == (640, 480)
        finally:
            vis.close()

    def test_exported_at_the_top_level(self):
        import mbo_utilities as mbo
        from mbo_utilities.gui.data_vis import DataVis

        assert mbo.DataVis is DataVis


# ----------------------------------------------------------------------
# run_gui on top
# ----------------------------------------------------------------------


class TestRunGuiWhereItRuns:
    @staticmethod
    def _offscreen(monkeypatch):
        from mbo_utilities.gui import run_gui as rg

        monkeypatch.setattr(
            rg, "_figure_kwargs_for_here", lambda size=None: {"size": FIGURE_SIZE}
        )
        return rg

    def test_notebook_displays_and_returns_the_vis(self, monkeypatch):
        from mbo_utilities.gui.data_vis import DataVis

        rg = self._offscreen(monkeypatch)
        monkeypatch.setattr(rg, "in_notebook", lambda: True)
        shown = []
        monkeypatch.setattr(rg, "display_widget", shown.append)
        ran = []
        import fastplotlib as fpl

        monkeypatch.setattr(fpl.loop, "run", lambda: ran.append(True))

        data = np.random.default_rng(1).random((3, 16, 16)).astype(np.float32)
        vis = rg._launch_standard_viewer(data, None, "none", False)
        try:
            assert isinstance(vis, DataVis)
            assert shown == [vis.show()], "the canvas went to the output cell"
            assert ran == [], "the kernel owns the loop"
        finally:
            vis.close()

    def test_terminal_runs_the_loop(self, monkeypatch):
        from mbo_utilities.gui import data_vis

        rg = self._offscreen(monkeypatch)
        monkeypatch.setattr(rg, "in_notebook", lambda: False)
        built = []

        class Recording(data_vis.DataVis):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                built.append(self)

        monkeypatch.setattr(data_vis, "DataVis", Recording)
        ran = []
        import fastplotlib as fpl

        monkeypatch.setattr(fpl.loop, "run", lambda: ran.append(True))

        data = np.random.default_rng(2).random((3, 16, 16)).astype(np.float32)
        try:
            assert rg._launch_standard_viewer(data, None, "none", False) is None
            assert ran == [True]
            assert len(built) == 1 and "shown" in repr(built[0])
        finally:
            for v in built:
                v.close()

    def test_no_path_in_a_notebook_is_an_error(self, monkeypatch):
        from mbo_utilities.gui import run_gui as rg

        monkeypatch.setattr(rg, "in_notebook", lambda: True)
        picked = []
        monkeypatch.setattr(rg, "_select_file", lambda **k: picked.append(k))
        with pytest.raises(ValueError, match="needs a path"):
            rg._run_gui_impl(data_in=None)
        assert picked == [], "the desktop picker was never opened"

    def test_cancelled_unit_picker_opens_nothing(self, monkeypatch):
        rg = self._offscreen(monkeypatch)
        monkeypatch.setattr(rg, "_resolve_mesc_unit", lambda d, u: ({}, False))
        assert rg._launch_standard_viewer("x.mesc", None, "none", False) is None


class TestMescInNotebook:
    """No Qt picker on the kernel's machine: the unit is an argument."""

    @staticmethod
    def _mesc(tmp_path, monkeypatch, units):
        from mbo_utilities.arrays import mesc
        from mbo_utilities.gui import run_gui as rg

        path = tmp_path / "scan.mesc"
        path.write_bytes(b"")
        monkeypatch.setattr(mesc, "list_mesc_units", lambda p: list(units))
        monkeypatch.setattr(rg, "in_notebook", lambda: True)
        monkeypatch.setattr(
            rg, "_prompt_for_mesc_unit", lambda *a: pytest.fail("Qt picker opened")
        )
        return rg, path

    def test_one_unit_opens_without_asking(self, tmp_path, monkeypatch):
        rg, path = self._mesc(tmp_path, monkeypatch, ["MSession_0/MUnit_0"])
        assert rg._resolve_mesc_unit(path, None) == ({"unit": "MSession_0/MUnit_0"}, True)

    def test_many_units_list_what_to_pick(self, tmp_path, monkeypatch):
        rg, path = self._mesc(
            tmp_path, monkeypatch, ["MSession_0/MUnit_0", "MSession_0/MUnit_1"]
        )
        with pytest.raises(ValueError, match="MUnit_1"):
            rg._resolve_mesc_unit(path, None)

    def test_explicit_unit_still_bypasses(self, tmp_path, monkeypatch):
        rg, path = self._mesc(tmp_path, monkeypatch, ["a", "b"])
        assert rg._resolve_mesc_unit(path, 1) == ({"unit": 1}, True)


# ----------------------------------------------------------------------
# opening a path without a native dialog
# ----------------------------------------------------------------------


class TestNativeDialogs:
    def test_desktop_platforms_always_have_one(self, monkeypatch):
        from mbo_utilities.gui import _files

        monkeypatch.setattr(_files.sys, "platform", "win32")
        assert _files.native_dialogs_available() is True

    def test_headless_linux_has_none(self, monkeypatch):
        from mbo_utilities.gui import _files

        monkeypatch.setattr(_files.sys, "platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert _files.native_dialogs_available() is False

    def test_linux_needs_a_dialog_helper(self, monkeypatch):
        from mbo_utilities.gui import _files

        monkeypatch.setattr(_files.sys, "platform", "linux")
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setattr(_files.shutil, "which", lambda name: None)
        assert _files.native_dialogs_available() is False
        monkeypatch.setattr(
            _files.shutil, "which", lambda name: "/usr/bin/zenity" if name == "zenity" else None
        )
        assert _files.native_dialogs_available() is True


class TestPathPrompt:
    def test_start_opens_on_a_path(self):
        from mbo_utilities.gui._files import PathPrompt

        p = PathPrompt("Open file", action="open")
        assert p.open is False
        p.status = "old"
        p.start("/data")
        assert (p.open, p.path, p.status) == (True, "/data", "")
        p.start()
        assert p.path == "/data"

    def test_closed_prompt_draws_nothing(self):
        from mbo_utilities.gui._files import PathPrompt, draw_path_prompt

        assert draw_path_prompt(PathPrompt("x")) == (None, False)

    def test_open_prompt_renders_in_a_frame(self):
        from mbo_utilities.gui._files import PathPrompt, draw_path_prompt
        from tests.test_imgui_helpers import _draw_in_edge_window

        prompt = PathPrompt("Open file", path="/data/x.tif", hint="a file")
        prompt.start()

        def body(out):
            out.append(draw_path_prompt(prompt))

        drawn = _draw_in_edge_window(300, body)
        assert drawn and all(d == (None, False) for d in drawn)
        assert prompt.open, "nothing pressed, so it stays up"


class TestOpenPrompts:
    """The menu / `o` route: a typed path first, native dialog as browse."""

    @staticmethod
    def _parent(tmp_path):
        return SimpleNamespace(
            fpath=str(tmp_path / "a.tif"),
            _file_dialog=None,
            _folder_dialog=None,
            logger=SimpleNamespace(info=lambda *a, **k: None),
        )

    def test_start_opens_the_prompt_on_the_data_dir(self, tmp_path):
        from mbo_utilities.gui._dialogs import start_open_prompt

        (tmp_path / "a.tif").write_bytes(b"")
        parent = self._parent(tmp_path)
        start_open_prompt(parent, "file")
        assert parent._open_file_prompt.open
        assert parent._open_file_prompt.path == str(tmp_path)
        assert not parent._open_folder_prompt.open
        start_open_prompt(parent, "folder")
        assert parent._open_folder_prompt.open
        assert not parent._open_file_prompt.open, "one prompt at a time"

    def test_a_native_dialog_in_flight_blocks_a_second_ask(self, tmp_path):
        from mbo_utilities.gui._dialogs import open_prompts, start_open_prompt

        parent = self._parent(tmp_path)
        parent._file_dialog = object()
        start_open_prompt(parent, "file")
        assert not open_prompts(parent)[0].open

    def test_submitted_path_loads_and_a_missing_one_reports(self, tmp_path, monkeypatch):
        from mbo_utilities.gui import _dialogs

        parent = self._parent(tmp_path)
        loaded = []
        monkeypatch.setattr(_dialogs, "load_new_data", lambda p, path: loaded.append(path))
        monkeypatch.setattr(_dialogs, "add_recent_file", lambda *a, **k: None)
        monkeypatch.setattr(_dialogs, "set_last_dir", lambda *a, **k: None)
        file_prompt, folder_prompt = _dialogs.open_prompts(parent)

        answers = {"file": (str(tmp_path / "missing.tif"), False), "folder": (None, False)}
        monkeypatch.setattr(
            _dialogs,
            "draw_path_prompt",
            lambda prompt: answers["file" if prompt is file_prompt else "folder"],
        )
        file_prompt.start()
        _dialogs.check_file_dialogs(parent)
        assert loaded == []
        assert file_prompt.open and "not found" in file_prompt.status

        answers["file"] = (str(tmp_path), False)
        _dialogs.check_file_dialogs(parent)
        assert loaded == [str(tmp_path)]
        assert not file_prompt.open

    def test_folder_prompt_rejects_a_file(self, tmp_path, monkeypatch):
        from mbo_utilities.gui import _dialogs

        parent = self._parent(tmp_path)
        f = tmp_path / "a.tif"
        f.write_bytes(b"")
        loaded = []
        monkeypatch.setattr(_dialogs, "load_new_data", lambda p, path: loaded.append(path))
        _, folder_prompt = _dialogs.open_prompts(parent)
        monkeypatch.setattr(
            _dialogs,
            "draw_path_prompt",
            lambda prompt: (str(f), False) if prompt is folder_prompt else (None, False),
        )
        folder_prompt.start()
        _dialogs.check_file_dialogs(parent)
        assert loaded == [] and "not a folder" in folder_prompt.status

    def test_browse_launches_the_native_dialog(self, tmp_path, monkeypatch):
        from mbo_utilities.gui import _dialogs

        parent = self._parent(tmp_path)
        file_prompt, _ = _dialogs.open_prompts(parent)
        opened = []

        class FakePfd:
            class opt:
                multiselect = 1

            @staticmethod
            def open_file(title, start, filters, opts):
                opened.append(("file", start))
                return SimpleNamespace(ready=lambda: False)

            @staticmethod
            def select_folder(title, start):
                opened.append(("folder", start))
                return SimpleNamespace(ready=lambda: False)

        monkeypatch.setattr(_dialogs, "pfd", FakePfd)
        monkeypatch.setattr(
            _dialogs,
            "draw_path_prompt",
            lambda prompt: (None, True) if prompt is file_prompt else (None, False),
        )
        file_prompt.start(str(tmp_path))
        _dialogs.check_file_dialogs(parent)
        assert opened == [("file", str(tmp_path))]
        assert parent._file_dialog is not None


class TestLauncherTypedPath:
    """`mbo` with no path: the launcher takes a typed path too."""

    @staticmethod
    def _dialog(monkeypatch):
        from mbo_utilities.gui.widgets import file_dialog as fd

        monkeypatch.setattr(fd, "add_recent_file", lambda *a, **k: None)
        monkeypatch.setattr(fd, "set_last_dir", lambda *a, **k: None)
        exits = []
        monkeypatch.setattr(
            fd.hello_imgui,
            "get_runner_params",
            lambda: SimpleNamespace(app_shall_exit=False)
            if not exits.append(True)
            else None,
        )
        dlg = fd.FileDialog.__new__(fd.FileDialog)
        dlg._typed_path = ""
        dlg._typed_status = ""
        dlg.selected_path = None
        dlg._save_gui_preferences = lambda: None
        return dlg, exits

    def test_missing_path_reports_instead_of_exiting(self, tmp_path, monkeypatch):
        dlg, exits = self._dialog(monkeypatch)
        dlg._typed_path = str(tmp_path / "nope")
        dlg._open_typed_path()
        assert dlg.selected_path is None and "not found" in dlg._typed_status
        assert exits == []

    def test_existing_path_is_the_selection(self, tmp_path, monkeypatch):
        dlg, exits = self._dialog(monkeypatch)
        dlg._typed_path = f'"{tmp_path}"'
        dlg._open_typed_path()
        assert dlg.selected_path == str(tmp_path)
        assert exits == [True]
