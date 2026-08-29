"""Fixtures shared by the suite: an isolated profile store and a headless imgui."""

from __future__ import annotations

from collections import Counter

import pytest
from imgui_bundle import imgui


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the profile store at a temp dir so no test touches a real ~/.mbo."""
    monkeypatch.setenv("MBO_DIR", str(tmp_path / "mbo"))
    return tmp_path


@pytest.fixture
def context():
    """A renderer-less imgui context with its own error checks turned up."""
    made = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1200, 900)
    io.delta_time = 1.0 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.config_debug_highlight_id_conflicts = True
    io.config_error_recovery_enable_assert = True
    yield made
    imgui.destroy_context()


class Renderer:
    """
    Draws in one headless frame and fails on what imgui would complain about.

    imgui only flags conflicting IDs when the mouse happens to be over one of
    them, which never happens without a window, so the ids buttons claim are
    recorded here instead: two identical ones in a frame is the bug that shows
    up in the app as "2 items with conflicting IDs".
    """

    def __init__(self, context):
        self.context = context
        self.button_real = imgui.button
        self.ids_claimed: list = []

    def button(self, label, *args, **kwargs):
        """Stand-in for ``imgui.button`` that records the id it claims."""
        self.ids_claimed.append((imgui.get_id(label), label))
        return self.button_real(label, *args, **kwargs)

    def __call__(self, *draws) -> None:
        """One frame containing ``draws``, each with its first section forced open."""
        self.ids_claimed = []
        imgui.button = self.button
        try:
            imgui.new_frame()
            imgui.begin("test")
            for draw in draws:
                imgui.set_next_item_open(True)
                draw()
            imgui.end()
            imgui.end_frame()
            imgui.render()
        finally:
            imgui.button = self.button_real
        self.check_ids()

    def check_ids(self) -> None:
        """Fail when two buttons in the frame claimed the same id."""
        counts = Counter(claimed for claimed, _ in self.ids_claimed)
        clashing = {label for claimed, label in self.ids_claimed if counts[claimed] > 1}
        if clashing:
            raise AssertionError(
                f"items with conflicting IDs: {sorted(clashing)}. Two widgets in "
                "one scope share a label; give one an imgui.push_id() of its own."
            )


@pytest.fixture
def render(context):
    """Draw callables inside one headless frame, checking imgui's error state."""
    return Renderer(context)
