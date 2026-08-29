"""
base class for ui widgets that control image display.

widgets are self-contained ui components that:
- check if they should display based on data/parent attributes
- draw imgui controls
- modify image_widget.data or window funcs directly

no processors needed - all manipulation happens via data attributes
or imagewidget's built-in window_funcs/spatial_func apis.
"""

from abc import ABC, abstractmethod
from typing import Any


class Widget(ABC):
    """base class for ui widgets."""

    # human-readable name
    name: str = "Widget"

    # priority for ordering (lower = rendered first)
    priority: int = 100

    # where the widget is drawn:
    #   "panel" -> stacked in the Preview tab's control column
    #   "tab"   -> its own tab in the viewer tab bar, labelled `tab_label`
    placement: str = "panel"

    # tab caption for placement == "tab"; defaults to `name`
    tab_label: str | None = None

    # key in widget_toggles.WIDGET_REGISTRY. When set, the host skips the
    # widget entirely while its Widgets-menu entry is off, so draw() never
    # has to check the top-level toggle itself (subwidget checks still do).
    toggle_key: str | None = None

    def __init__(self, parent: Any):
        self.parent = parent

    @classmethod
    @abstractmethod
    def is_supported(cls, parent: Any) -> bool:
        """
        Check if this widget should show for the given parent.

        override to check data metadata, attributes, etc.
        """
        ...

    @abstractmethod
    def draw(self) -> None:
        """Draw the imgui ui for this widget."""
        ...

    def tab_disabled(self) -> str | None:
        """For placement == "tab": None when selectable, else a reason.

        The reason is shown as the tooltip on the greyed-out tab; return an
        empty string to grey it out without one.
        """
        return None

    def wants_focus(self) -> bool:
        """For placement == "tab": select this tab on the next frame."""
        return False

    def cleanup(self) -> None:
        """Clean up resources when widget is destroyed.

        override in subclasses to release resources like open windows,
        background threads, file handles, etc.
        """
