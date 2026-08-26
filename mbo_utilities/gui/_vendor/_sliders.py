import math
import os
from collections import defaultdict
from time import perf_counter

from imgui_bundle import imgui, icons_fontawesome_6 as fa

from mbo_utilities.gui._vendor._edge_window import EdgeWindow


class ImageWidgetSliders(EdgeWindow):
    def __init__(self, figure, size, location, title, image_widget):
        super().__init__(figure=figure, size=size, location=location, title=title)
        self._image_widget = image_widget

        # per-dimension playback state. defaultdicts, not fixed {"t", "z"}
        # keys: the widget supports a third scrollable dim ("c"), and a data
        # swap can add one, which a literal dict would KeyError on.
        self._playing: dict[str, bool] = defaultdict(bool)

        # approximate framerate for playing
        self._fps: dict[str, int] = defaultdict(lambda: 20)
        # framerate converted to frame time
        self._frame_time: dict[str, float] = defaultdict(lambda: 1 / 20)

        # dims whose fps the user typed explicitly — metadata seeding
        # (seed_fps) never overrides these
        self._user_fps: set[str] = set()

        # last timepoint that a frame was displayed from a given dimension
        self._last_frame_time: dict[str, float] = defaultdict(float)

        self._loop = False

        if "RTD_BUILD" in os.environ.keys():
            if os.environ["RTD_BUILD"] == "1":
                self._playing["t"] = True
                self._loop = True

    def seed_fps(self, dim: str, fps) -> None:
        """seed the playback rate for a dim from data metadata.

        ignores None/non-numeric/non-finite/<=0 values, clamps to the
        slider's [1, 50] range, and never overrides an fps the user typed
        themselves.
        """
        if fps is None or dim in self._user_fps:
            return
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            return
        # nan slips past `<= 0` and int(round(nan)) raises; inf overflows
        if fps <= 0 or not math.isfinite(fps):
            return
        value = min(max(int(round(fps)), 1), 50)
        self._fps[dim] = value
        self._frame_time[dim] = 1 / value

    def set_index(self, dim: str, index: int):
        """set the current_index of the ImageWidget"""

        # make sure the max index for this dim is not exceeded
        max_index = self._image_widget._dims_max_bounds[dim] - 1
        if index > max_index:
            if self._loop:
                # loop back to index zero if looping is enabled
                index = 0
            else:
                # if looping not enabled, stop playing this dimension
                self._playing[dim] = False
                return

        # set current_index
        self._image_widget.current_index = {dim: min(index, max_index)}

    def _dim_label(self, dim: str) -> str:
        """Display name for a scrollable axis.

        Falls back to the internal letter when the widget carries no
        `_slider_dim_names` (plain fastplotlib usage).
        """
        names = getattr(self._image_widget, "_slider_dim_names", None) or ()
        dims = self._image_widget.slider_dims
        try:
            return names[dims.index(dim)]
        except (ValueError, IndexError):
            return dim

    def update(self):
        """called on every render cycle to update the GUI elements"""

        # store the new index of the image widget ("t" and "z")
        new_index = dict()

        # flag if the index changed
        flag_index_changed = False

        # reset vmin-vmax using full orig data
        if imgui.button(label=fa.ICON_FA_CIRCLE_HALF_STROKE + fa.ICON_FA_FILM):
            self._image_widget.reset_vmin_vmax()
        if imgui.is_item_hovered(0):
            imgui.set_tooltip("reset contrast limits using full movie/stack")

        # reset vmin-vmax using currently displayed ImageGraphic data
        imgui.same_line()
        if imgui.button(label=fa.ICON_FA_CIRCLE_HALF_STROKE):
            self._image_widget.reset_vmin_vmax_frame()
        if imgui.is_item_hovered(0):
            imgui.set_tooltip("reset contrast limits using current frame")

        # time now
        now = perf_counter()

        # buttons and slider UI elements for each dim
        for dim in self._image_widget.slider_dims:
            imgui.push_id(f"{self._id_counter}_{dim}")

            if self._playing[dim]:
                # show pause button if playing
                if imgui.button(label=fa.ICON_FA_PAUSE):
                    # if pause button clicked, then set playing to false
                    self._playing[dim] = False

                # if in play mode and enough time has elapsed w.r.t. the desired framerate, increment the index
                if now - self._last_frame_time[dim] >= self._frame_time[dim]:
                    self.set_index(dim, self._image_widget.current_index[dim] + 1)
                    self._last_frame_time[dim] = now

            else:
                # we are not playing, so display play button
                if imgui.button(label=fa.ICON_FA_PLAY):
                    # if play button is clicked, set last frame time to 0 so that index increments on next render
                    self._last_frame_time[dim] = 0
                    # set playing to True since play button was clicked
                    self._playing[dim] = True

            imgui.same_line()
            # step back one frame button
            if imgui.button(label=fa.ICON_FA_BACKWARD_STEP) and not self._playing[dim]:
                self.set_index(dim, self._image_widget.current_index[dim] - 1)

            imgui.same_line()
            # step forward one frame button
            if imgui.button(label=fa.ICON_FA_FORWARD_STEP) and not self._playing[dim]:
                self.set_index(dim, self._image_widget.current_index[dim] + 1)

            imgui.same_line()
            # stop button
            if imgui.button(label=fa.ICON_FA_STOP):
                self._playing[dim] = False
                self._last_frame_time[dim] = 0
                self.set_index(dim, 0)

            imgui.same_line()
            # loop checkbox
            _, self._loop = imgui.checkbox(label=fa.ICON_FA_ROTATE, v=self._loop)
            if imgui.is_item_hovered(0):
                imgui.set_tooltip("loop playback")

            imgui.same_line()
            imgui.text("framerate :")
            imgui.same_line()
            imgui.set_next_item_width(100)
            # framerate int entry
            fps_changed, value = imgui.input_int(
                label="fps", v=self._fps[dim], step_fast=5
            )
            if imgui.is_item_hovered(0):
                imgui.set_tooltip(
                    "framerate is approximate and less reliable as it approaches your monitor refresh rate"
                )
            if fps_changed:
                if value < 1:
                    value = 1
                if value > 50:
                    value = 50
                self._fps[dim] = value
                self._frame_time[dim] = 1 / value
                self._user_fps.add(dim)

            val = self._image_widget.current_index[dim]
            vmax = self._image_widget._dims_max_bounds[dim] - 1

            # "t"/"z"/"c" are internal positional labels; show the name the
            # array actually reports for that axis ("Timepoint", "ROI", ...)
            imgui.text(f"{self._dim_label(dim)}: ")
            imgui.same_line()
            # so that slider occupies full width
            imgui.set_next_item_width(self.width * 0.85)

            if "Jupyter" in self._image_widget.figure.canvas.__class__.__name__:
                # until https://github.com/pygfx/wgpu-py/issues/530
                flags = imgui.SliderFlags_.no_input
            else:
                # clamps to min, max if user inputs value outside these bounds
                flags = imgui.SliderFlags_.always_clamp

            # slider for this dimension
            changed, index = imgui.slider_int(
                f"##{dim}", v=val, v_min=0, v_max=vmax, flags=flags
            )

            new_index[dim] = index

            # if the slider value changed for this dimension
            flag_index_changed |= changed

            imgui.pop_id()

        if flag_index_changed:
            # if any slider dim changed set the new index of the image widget
            self._image_widget.current_index = new_index

