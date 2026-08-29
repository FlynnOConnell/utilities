"""
widget registry with auto-discovery.

widgets are automatically discovered from this directory.
each widget module should define a class that inherits from Widget.

to add a new widget:
1. create a new file in widgets/ (e.g. my_widget.py)
2. define a class inheriting from Widget
3. implement is_supported() and draw()
4. the widget will be auto-discovered and shown when is_supported() returns True
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui.widgets.menu_bar import draw_menu_bar

# registry of all discovered widget classes
_WIDGET_CLASSES: list[type[Widget]] = []


def _discover_widgets() -> None:
    """auto-discover widget classes from this package."""
    global _WIDGET_CLASSES

    if _WIDGET_CLASSES:
        return  # already discovered

    package_dir = Path(__file__).parent

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        # skip private modules
        if module_info.name.startswith("_"):
            continue

        try:
            module = importlib.import_module(f".{module_info.name}", package=__name__)

            # find Widget subclasses in module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Widget)
                    and attr is not Widget
                ):
                    _WIDGET_CLASSES.append(attr)
        except Exception:
            # log but don't crash on import errors
            pass

    # sort by priority
    _WIDGET_CLASSES.sort(key=lambda w: w.priority)


def _instantiate(parent: Any, placement: str) -> list[Widget]:
    """Instantiate every supported widget with the given placement."""
    _discover_widgets()

    supported = []
    for widget_cls in _WIDGET_CLASSES:
        if widget_cls.placement != placement:
            continue
        try:
            if widget_cls.is_supported(parent):
                supported.append(widget_cls(parent))
        except Exception:
            # Log the error for debugging
            import traceback
            traceback.print_exc()

    # sort by priority (lower = first)
    supported.sort(key=lambda w: w.priority)
    return supported


def get_supported_widgets(parent: Any) -> list[Widget]:
    """
    Get the panel widgets that are supported for the given parent.

    returns instantiated widgets sorted by priority.
    """
    return _instantiate(parent, "panel")


def get_tab_widgets(parent: Any) -> list[Widget]:
    """Get the supported widgets that render as their own viewer tab."""
    return _instantiate(parent, "tab")


def widget_is_visible(widget: Widget) -> bool:
    """Whether the Widgets menu currently has this widget switched on."""
    if widget.toggle_key is None:
        return True
    from mbo_utilities.gui.widgets.widget_toggles import widget_enabled

    return widget_enabled(widget.toggle_key)


def draw_all_widgets(parent: Any, widgets: list[Widget]) -> None:
    """Draw all supported widgets that the Widgets menu leaves switched on."""
    for widget in widgets:
        if not widget_is_visible(widget):
            continue
        try:
            widget.draw()
        except Exception as e:
            # log error but don't crash the ui
            import traceback
            error_msg = f"Error in {widget.name}: {e}"
            parent.logger.exception(error_msg)
            parent.logger.exception(traceback.format_exc())
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.3, 0.3, 1.0),
                error_msg
            )


def cleanup_all_widgets(widgets: list[Widget]) -> None:
    """Clean up all widgets when the gui is closing.

    calls cleanup() on each widget to release resources.
    """
    for widget in widgets:
        try:
            widget.cleanup()
        except Exception:
            # log but don't crash during cleanup
            pass


__all__ = [
    "Widget",
    "cleanup_all_widgets",
    "draw_all_widgets",
    "draw_menu_bar",
    "get_supported_widgets",
    "get_tab_widgets",
    "widget_is_visible",
]
