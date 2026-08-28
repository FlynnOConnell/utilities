"""
base class for pipeline widgets.

each pipeline is self-contained with its own settings dataclass and config ui.
"""

from abc import ABC, abstractmethod
from typing import Any

from mbo_utilities.pipeline_registry import PipelineInfo

# How a pipeline consumes each non-spatial axis. Read by
# ``_selection_ui.draw_selection_table`` to decide what to draw per row.
#
#   "range"      user picks a start:stop range (the historical behaviour)
#   "all"        the pipeline needs the whole axis; row is shown disabled
#   "none"       axis does not apply; row is hidden
#   "select-one" exactly one index; row draws a single-select
AXIS_MODES = ("range", "all", "none", "select-one")

# Reproduces the pre-existing behaviour for every widget that does not
# override it, so suite2p / masknmf / isoview are unaffected.
DEFAULT_AXES_CONSUMED: dict[str, str] = {"T": "range", "Z": "range", "C": "range"}


class PipelineWidget(ABC):
    """base class for pipeline widgets."""

    # human-readable name shown in pipeline selector
    name: str = "Pipeline"

    # whether this pipeline's dependencies are installed
    is_available: bool = False

    # install command to show when not available
    install_command: str = "uv pip install mbo_utilities"

    # file patterns / marker files, registered when the widget is discovered
    # through the ``mbo_utilities.pipelines`` entry-point group
    info: PipelineInfo | None = None

    # per-axis consumption mode; see AXIS_MODES
    axes_consumed: dict[str, str] = DEFAULT_AXES_CONSUMED

    def __init__(self, parent: Any):
        self.parent = parent

    @classmethod
    def axis_mode(cls, axis: str) -> str:
        """
        How this pipeline consumes ``axis`` ("T", "Z" or "C").

        Returns
        -------
        str
            One of :data:`AXIS_MODES`; "range" for an axis the pipeline
            does not declare.
        """
        mode = cls.axes_consumed.get(axis.upper(), "range")
        if mode not in AXIS_MODES:
            raise ValueError(
                f"{cls.__name__}.axes_consumed[{axis!r}] is {mode!r}, "
                f"expected one of {AXIS_MODES}"
            )
        return mode

    def draw(self) -> None:
        """Draw the pipeline widget."""
        self.draw_config()

    @abstractmethod
    def draw_config(self) -> None:
        """Draw the configuration/processing ui."""
        ...

    @classmethod
    def applies_to(cls, arr: Any) -> bool:
        """True iff this pipeline can be run against ``arr``.

        Called by the Run-tab selector BEFORE instantiation, so it
        must be safe to call without spinning up the widget. Override
        for pipelines tied to a specific array type (e.g. Isoview
        consolidator only applies to ``IsoviewArray`` instances).

        ``arr`` may be ``None`` when no data is loaded.

        Default: returns ``True`` (pipeline works on any data).
        """
        return True

    def cleanup(self) -> None:
        """Clean up resources when widget is destroyed.

        override in subclasses to release resources like open windows,
        background threads, file handles, etc.
        """
