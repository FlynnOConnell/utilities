"""``DataVis``: the Miller Brain Studio viewer as an object.

The same viewer ``mbo path/to/data`` opens, built the way masknmf's
``*Vis`` classes are: construct with the data, ``show()`` to put it on
screen, ``close()`` to take it down. ``run_gui`` is the one-call form that
also picks the canvas and size for wherever it is running.

In a notebook, construct and call ``show()`` in the same cell; the canvas is
the cell's output and the kernel's loop drives it, so ``fpl.loop.run()``
must not be called. In a terminal or script, ``show()`` opens a window and
the caller runs ``fpl.loop.run()`` (``run_gui`` does).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["DataVis"]


class DataVis:
    """Preview imaging data of any supported type, with the side widget.

    Parameters
    ----------
    data : str, Path, array, or sequence of paths
        Anything ``imread`` opens, or an array already in memory.
    roi : int or tuple of int, optional
        ROI index(es) for multi-ROI raw files. None shows every ROI.
    widget : str, default "preview"
        ``"preview"`` attaches the side widget, ``"manualroi"`` adds the
        manual ROI tools to it, ``"none"`` shows only the canvas.
    unit : int or str, optional
        Which measurement unit of a ``.mesc`` to open. In a terminal the
        picker asks when this is omitted; in a notebook it is required when
        the file holds more than one.
    size : tuple of int, optional
        Canvas size in pixels. Defaults to the screen's work area for a
        desktop window and to (1400, 900) in a notebook, where the edge
        windows need fixed room.
    figure_kwargs
        Passed on to the figure, e.g. ``canvas="jupyter"``. ``run_gui`` sets
        these itself.

    Examples
    --------
    In a notebook::

        from mbo_utilities import DataVis
        vis = DataVis("path/to/data.tif")
        vis.show()

    then ``vis.close()`` when done. From a script::

        import fastplotlib as fpl
        vis = DataVis("path/to/data.tif")
        vis.show()
        fpl.loop.run()
    """

    def __init__(
        self,
        data: Any,
        roi: int | tuple[int, ...] | None = None,
        widget: bool | str = "preview",
        unit: int | str | None = None,
        size: tuple[int, int] | None = None,
        **figure_kwargs,
    ):
        from mbo_utilities.gui.run_gui import (
            _create_image_widget,
            _figure_kwargs_for_here,
            _load_for_viewer,
        )

        self._source = data
        self._data_array = _load_for_viewer(data, roi=roi, unit=unit)
        kwargs = _figure_kwargs_for_here(size=size)
        kwargs.update(figure_kwargs)
        self._iw = _create_image_widget(
            self._data_array,
            widget=widget,
            figure_kwargs_override=kwargs,
            show=False,
        )
        self._output = None
        self._shown = False
        self._closed = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def show(self, **kwargs):
        """Show the figure. Returns the canvas, which a notebook cell displays
        when it is the last expression."""
        from mbo_utilities.gui.run_gui import _after_show

        if not self._shown:
            # offscreen and desktop canvases return None here; only the
            # notebook canvas is a widget worth handing back
            self._output = self._iw.show(**kwargs)
            self._shown = True
            _after_show(self._iw)
        return self._output

    def close(self) -> None:
        """Stop the side widget's threads and close the figure."""
        if self._closed:
            return
        self._closed = True
        gui = self.widget
        if gui is not None:
            cleanup = getattr(gui, "cleanup", None)
            if cleanup is not None:
                try:
                    cleanup()
                except Exception:  # noqa: BLE001 - a dead thread must not block close
                    pass
        self._iw.close()

    @property
    def closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------
    # what it is made of
    # ------------------------------------------------------------------

    @property
    def iw(self):
        """The ``MboNDViewer`` underneath: sliders, cmap, window functions."""
        return self._iw

    @property
    def image_widget(self):
        return self._iw

    @property
    def figure(self):
        return self._iw.figure

    @property
    def data(self):
        """The array as the viewer sees it (after axial and phase wraps)."""
        return self._data_array

    @property
    def source(self):
        """What the viewer was built from: a path, paths, or an array."""
        return self._source

    @property
    def widget(self):
        """The ``PreviewDataWidget`` on the figure, or None with ``widget="none"``."""
        try:
            from mbo_utilities.gui.widgets.preview_data import PreviewDataWidget
        except ImportError:
            return None
        windows = getattr(self.figure, "imgui_windows", None) or {}
        for w in windows.values():
            if isinstance(w, PreviewDataWidget):
                return w
        return None

    def __repr__(self) -> str:
        src = self._source
        if isinstance(src, (str, Path)):
            what = Path(src).name
        elif isinstance(src, (list, tuple)) and src:
            what = f"{len(src)} files"
        else:
            what = type(src).__name__
        state = "closed" if self._closed else ("shown" if self._shown else "built")
        return f"DataVis({what}, shape={tuple(self._data_array.shape)}, {state})"
