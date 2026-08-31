"""Registry and menu for the "Widgets" menu-bar entry.

Every toggleable piece of UI is one :class:`WidgetEntry` with a list of
:class:`SubWidget` children. The menu draws a checkbox per entry and per
child; the draw code for each widget asks :func:`widget_enabled` /
:func:`sub_enabled` before rendering.

State is flat (``"scan_phase"``, ``"scan_phase.border"``) and persisted in
preferences, so toggles survive a restart. Anything absent from the stored
mapping falls back to the registry default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from imgui_bundle import imgui

from mbo_utilities.preferences import get_widget_toggles, set_widget_toggles

__all__ = [
    "SubWidget",
    "WidgetEntry",
    "WIDGET_REGISTRY",
    "draw_widgets_menu",
    "get_entry",
    "reset_widget_toggles",
    "set_sub_enabled",
    "set_widget_enabled",
    "sub_enabled",
    "widget_enabled",
]


@dataclass(frozen=True)
class SubWidget:
    """One toggleable section inside a widget."""

    key: str
    label: str
    default: bool = True
    tooltip: str = ""
    # same contract as WidgetEntry.on_toggle, for a section that owns live
    # state of its own (an edge window, say) rather than just a block of draw
    # calls the parent can skip.
    on_toggle: Callable[[Any, bool], None] | None = field(default=None, compare=False)


@dataclass(frozen=True)
class WidgetEntry:
    """A widget in the Widgets menu, plus its subwidgets."""

    key: str
    label: str
    subwidgets: tuple[SubWidget, ...] = ()
    default: bool = True
    tooltip: str = ""
    # called as on_toggle(parent, enabled) right after the user flips the
    # widget checkbox; lets a widget build/tear down live state (the ROI
    # overlay, say) instead of only gating its draw.
    on_toggle: Callable[[Any, bool], None] | None = field(default=None, compare=False)

    def sub(self, key: str) -> SubWidget | None:
        for s in self.subwidgets:
            if s.key == key:
                return s
        return None


def _toggle_manual_roi(parent: Any, enabled: bool) -> None:
    """Create or drop the ROI overlay and its edge windows when toggled."""
    sync = getattr(parent, "sync_manual_roi", None)
    if sync is not None:
        sync(enabled)


WIDGET_REGISTRY: tuple[WidgetEntry, ...] = (
    WidgetEntry(
        key="preview",
        label="Image",
        tooltip="The Image tab and the control panels stacked inside it.",
        subwidgets=(
            SubWidget("mesc_units", "MESc Units"),
            SubWidget("window_functions", "Window Functions"),
            SubWidget("spatial_functions", "Spatial Functions"),
            SubWidget("scan_phase", "Scan-Phase Correction"),
            SubWidget("frame_averaging", "Frame Averaging"),
            SubWidget("summary_images", "Summary Images"),
            SubWidget("align_views", "Align Views"),
            SubWidget("projections", "Projections"),
            SubWidget("tile_grid", "Tile Grid"),
        ),
    ),
    WidgetEntry(
        key="signal_quality",
        label="Signal Quality",
        tooltip="Per-plane z-stats and signal-quality plots.",
    ),
    WidgetEntry(
        key="run",
        label="Run",
        tooltip="Registration / segmentation pipelines.",
    ),
    WidgetEntry(
        key="manual_roi",
        label="Manual ROI Labeling",
        tooltip="Freehand ROI drawing and labelling: control cards and the trace "
                "plot in a top panel; the ROI and trace tables in their own tabs.",
        default=False,
        on_toggle=_toggle_manual_roi,
        subwidgets=(
            SubWidget("tools", "Drawing tools"),
            SubWidget("overlay", "Overlay controls"),
            SubWidget("labels", "Label editor"),
            SubWidget("process", "Process card"),
            SubWidget("table", "ROI table", tooltip="The ROIs tab in this panel."),
            SubWidget("traces", "Trace table", tooltip="The Traces tab: every collected trace with stats."),
        ),
    ),
)

_BY_KEY = {e.key: e for e in WIDGET_REGISTRY}

# lazily loaded from preferences, then kept in sync on every write
_state: dict[str, bool] | None = None


def get_entry(key: str) -> WidgetEntry | None:
    """Return the registry entry for ``key``, or None."""
    return _BY_KEY.get(key)


def _load() -> dict[str, bool]:
    global _state
    if _state is None:
        try:
            _state = get_widget_toggles()
        except Exception:
            _state = {}
    return _state


def _persist() -> None:
    try:
        set_widget_toggles(_load())
    except Exception:
        pass


def _default(key: str) -> bool:
    widget_key, _, sub_key = key.partition(".")
    entry = _BY_KEY.get(widget_key)
    if entry is None:
        return True
    if not sub_key:
        return entry.default
    sub = entry.sub(sub_key)
    return entry.default if sub is None else sub.default


def widget_enabled(key: str) -> bool:
    """Whether ``key`` should be drawn.

    Accepts a widget key (``"preview"``) or a subwidget key
    (``"preview.scan_phase"``); a subwidget is off whenever its parent
    widget is, so callers only need this one check.
    """
    widget_key, _, sub_key = key.partition(".")
    if sub_key and not bool(_load().get(widget_key, _default(widget_key))):
        return False
    return bool(_load().get(key, _default(key)))


def sub_enabled(widget_key: str, sub_key: str) -> bool:
    """Whether subwidget ``sub_key`` of ``widget_key`` should be drawn."""
    return widget_enabled(f"{widget_key}.{sub_key}")


def set_widget_enabled(key: str, value: bool, persist: bool = True) -> None:
    """Turn widget ``key`` on or off."""
    _load()[key] = bool(value)
    if persist:
        _persist()


def set_sub_enabled(
    widget_key: str, sub_key: str, value: bool, persist: bool = True
) -> None:
    """Turn one subwidget on or off."""
    _load()[f"{widget_key}.{sub_key}"] = bool(value)
    if persist:
        _persist()


def _fire(parent: Any, on_toggle, key: str, value: bool) -> None:
    """Run a widget/subwidget toggle callback, logging rather than raising."""
    if parent is None or on_toggle is None:
        return
    try:
        on_toggle(parent, value)
    except Exception:
        logger = getattr(parent, "logger", None)
        if logger is not None:
            logger.exception(f"widget toggle failed for {key}")


def _apply_all(parent: Any, chosen, persist: bool = True) -> None:
    """Set every widget and subwidget to ``chosen(entry_or_sub)`` and fire callbacks."""
    state = _load()
    for entry in WIDGET_REGISTRY:
        was_on = widget_enabled(entry.key)
        value = chosen(entry)
        state[entry.key] = value
        for sub in entry.subwidgets:
            sub_key = f"{entry.key}.{sub.key}"
            sub_was_on = widget_enabled(sub_key)
            sub_value = chosen(sub)
            state[sub_key] = sub_value
            # a subwidget is off whenever its parent is, so only fire its
            # callback once the parent's own state has settled
            if sub.on_toggle and (sub_was_on != sub_value or was_on != value):
                _fire(parent, sub.on_toggle, sub_key, sub_value and value)
        if was_on != value:
            _fire(parent, entry.on_toggle, entry.key, value)
    if persist:
        _persist()


def reset_widget_toggles(parent: Any = None, persist: bool = True) -> None:
    """Restore every widget and subwidget to its registry default."""
    _apply_all(parent, lambda item: item.default, persist)


def _set_all(parent: Any, value: bool) -> None:
    _apply_all(parent, lambda item: value)


def _apply_toggle(parent: Any, entry: WidgetEntry, value: bool) -> None:
    """Flip one widget and let it build or tear down whatever it owns."""
    set_widget_enabled(entry.key, value)
    _fire(parent, entry.on_toggle, entry.key, value)


def _apply_sub_toggle(
    parent: Any, entry: WidgetEntry, sub: SubWidget, value: bool
) -> None:
    """Flip one subwidget and let it build or tear down whatever it owns."""
    set_sub_enabled(entry.key, sub.key, value)
    _fire(parent, sub.on_toggle, f"{entry.key}.{sub.key}", value)


def draw_widgets_menu(parent: Any) -> None:
    """Draw the "Widgets" menu. Call inside an active menu bar."""
    if not imgui.begin_menu("Widgets", True):
        return

    for entry in WIDGET_REGISTRY:
        enabled = widget_enabled(entry.key)

        # a widget with no subwidgets is just a checkbox; only one with
        # children earns a submenu
        if not entry.subwidgets:
            clicked, new_value = imgui.menu_item(entry.label, "", enabled, True)
            if entry.tooltip and imgui.is_item_hovered():
                imgui.set_tooltip(entry.tooltip)
            if clicked and new_value != enabled:
                _apply_toggle(parent, entry, new_value)
            continue

        if not imgui.begin_menu(entry.label, True):
            if entry.tooltip and imgui.is_item_hovered():
                imgui.set_tooltip(entry.tooltip)
            continue
        if entry.tooltip:
            imgui.text_disabled(entry.tooltip)
            imgui.separator()

        clicked, new_value = imgui.menu_item(
            f"Show {entry.label}", "", enabled, True
        )
        if clicked and new_value != enabled:
            _apply_toggle(parent, entry, new_value)
            enabled = new_value

        if entry.subwidgets:
            imgui.separator()
            if not enabled:
                imgui.begin_disabled()
            for sub in entry.subwidgets:
                sub_on = bool(
                    _load().get(f"{entry.key}.{sub.key}", sub.default)
                )
                sub_clicked, sub_value = imgui.menu_item(
                    sub.label, "", sub_on, True
                )
                if sub.tooltip and imgui.is_item_hovered(
                    imgui.HoveredFlags_.allow_when_disabled
                ):
                    imgui.set_tooltip(sub.tooltip)
                if sub_clicked and sub_value != sub_on:
                    _apply_sub_toggle(parent, entry, sub, sub_value)
            if not enabled:
                imgui.end_disabled()
        imgui.end_menu()

    imgui.separator()
    if imgui.menu_item("Enable All", "", False, True)[0]:
        _set_all(parent, True)
    if imgui.menu_item("Disable All", "", False, True)[0]:
        _set_all(parent, False)
    if imgui.menu_item("Reset to Defaults", "", False, True)[0]:
        reset_widget_toggles(parent)

    imgui.separator()
    # windows, not tabs: these open as floating popups over the viewer
    if imgui.menu_item("BioHPC...", "", False, True)[0]:
        parent._show_biohpc = True
    if imgui.menu_item("Cloud (GPU)...", "", False, True)[0]:
        parent._show_cloud = True
    imgui.end_menu()
