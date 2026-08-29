"""Fixtures shared by the suite: an isolated profile store and a headless imgui."""

from __future__ import annotations

import pytest
from imgui_bundle import imgui


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the profile store at a temp dir so no test touches a real ~/.mbo."""
    monkeypatch.setenv("MBO_DIR", str(tmp_path / "mbo"))
    return tmp_path


@pytest.fixture
def context():
    """A renderer-less imgui context, torn down after the test."""
    imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1200, 900)
    io.delta_time = 1.0 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield
    imgui.destroy_context()


def render_frame(*draws) -> None:
    """One frame containing ``draws``, each with its first section forced open."""
    imgui.new_frame()
    imgui.begin("test")
    for draw in draws:
        imgui.set_next_item_open(True)
        draw()
    imgui.end()
    imgui.end_frame()
    imgui.render()


@pytest.fixture
def render():
    """Draw callables inside one headless frame; needs the ``context`` fixture."""
    return render_frame
