"""EdgeWindow shim over the ndwidget branch's ImguiWindow.

The pre-refactor fastplotlib EdgeWindow took its placement in the
constructor and was attached with ``figure.add_gui``; the ndwidget branch
replaced it with ``ImguiWindow`` + ``figure.add_imgui_window(window,
location=..., size=..., ...)``. Same draw contract either way: the base
manages the window, the subclass implements ``update()``. This shim keeps
the old constructor signature and registers itself with the figure, so
subclasses written for EdgeWindow (PreviewDataWidget, the vendored
ImageWidgetSliders) run unchanged.
"""

from imgui_bundle import imgui

from fastplotlib.ui import ImguiWindow


class EdgeWindow(ImguiWindow):
    def __init__(
        self,
        figure,
        size: int,
        location: str,
        title: str,
        window_flags=imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_resize,
        *args,
        **kwargs,
    ):
        super().__init__()
        figure.add_imgui_window(
            self,
            location=location,
            size=size,
            title=title,
            window_flags=window_flags,
        )

    def update(self):
        raise NotImplementedError

    def get_rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height
