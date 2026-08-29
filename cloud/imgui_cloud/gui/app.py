"""
Standalone window: pick data with imgui_data_loader, then run it on an A100.

This is the "no host app" path. It opens the file dialog from imgui_data_loader
to choose the dataset - the same launcher the viewer uses, so the upload starts
where every other data selection in the lab starts - and then hands that path to
:class:`~imgui_cloud.gui.panel.CloudPanel` in a window of its own.
"""

from __future__ import annotations


from imgui_data_loader import (
    ButtonSpec,
    FileDialogConfig,
    FileType,
    JsonPreferenceStore,
    PickKind,
    Theme,
    run_file_dialog,
)


def pick_dataset(theme: Theme | None = None) -> str:
    """
    Open the imgui_data_loader launcher and return the chosen path.

    Returns
    -------
    str
        The selected folder or file; empty when the user quit.
    """
    config = FileDialogConfig(
        title="Run on Google Cloud",
        subtitle="Choose the dataset to upload to an A100 worker",
        buttons=[
            ButtonSpec(
                "Select Folder",
                PickKind.SELECT_FOLDER,
                tooltip="Upload a whole acquisition folder",
            ),
            ButtonSpec(
                "Open File(s)",
                PickKind.OPEN_FILE,
                multiselect=True,
                tooltip="Upload specific files",
            ),
        ],
        filetypes=[
            FileType("Imaging data", "*.tif *.tiff *.h5 *.zarr *.bin"),
            FileType("All Files", "*"),
        ],
        theme=theme or Theme.dark(),
        persistence=JsonPreferenceStore(),
        window_title="imgui_cloud",
    )
    result = run_file_dialog(config)
    return result.path or ""


def run_cloud_app(
    dir_input: str = "", theme: Theme | None = None, pick_first: bool = False
) -> None:
    """
    Launch the cloud panel as its own application.

    The panel opens straight away and shows the sign-in form until the profile
    checks out, so signing in never requires having data selected first. Data is
    chosen inside the panel, on its Data tab.

    Parameters
    ----------
    dir_input : str
        Dataset to start from; may be filled in later from the Data tab.
    theme : imgui_data_loader.Theme, optional
        Palette shared by the dialog and the panel.
    pick_first : bool
        Open the imgui_data_loader launcher before the panel, for the
        "pick data, then send it up" flow.
    """
    from imgui_bundle import hello_imgui, imgui, immapp

    from imgui_cloud.gui.panel import CloudPanel

    theme = theme or Theme.dark()
    if pick_first and not dir_input:
        dir_input = pick_dataset(theme)

    panel = CloudPanel(dir_input=dir_input, theme=theme)

    def gui() -> None:
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(12, 12))
        panel.draw()
        imgui.pop_style_var()

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "imgui_cloud"
    params.app_window_params.window_geometry.size = (900, 720)
    params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_window
    )
    params.callbacks.show_gui = gui
    immapp.run(params)
