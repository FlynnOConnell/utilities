"""EdgeWindow: ImguiWindow that self-registers on a figure edge."""

from imgui_bundle import imgui

from fastplotlib.ui import ImguiWindow


class EdgeWindow(ImguiWindow):
    def __init__(
        self,
        figure,
        size: int,
        location: str,
        title: str,
        window_flags=imgui.WindowFlags_.no_collapse
        | imgui.WindowFlags_.no_resize
        | imgui.WindowFlags_.no_title_bar,
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

    def draw(self):
        self.draw_window()

    def draw_window(self):
        ImguiWindow.draw(self)

    def update(self):
        raise NotImplementedError

    def get_rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height
