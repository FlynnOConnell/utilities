"""
The cloud panel: sign in, pick data, size the box, launch, watch, retrieve.

:class:`CloudPanel` is a plain widget - construct it once, call :meth:`draw`
once per frame from inside whatever window or tab it belongs to. It holds no
imgui context of its own, so the same object backs the standalone app, an
imgui_data_loader dialog, and the viewer's "Cloud" tab.

Everything long-running lives on :class:`~imgui_cloud.run.CloudRun`'s thread;
this file only reads snapshots, which is why the UI stays responsive while a
few hundred gigabytes go up to a bucket.
"""

from __future__ import annotations


import json
import time
import webbrowser
from pathlib import Path

from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui, portable_file_dialogs as pfd
from imgui_data_loader import Theme, pop_button_style, push_button_style, to_vec4

from imgui_cloud import config as config_module
from imgui_cloud import credentials as credentials_module
from imgui_cloud import history, pipelines
from imgui_cloud import run as run_module
from imgui_cloud.gui import setup as setup_module
from imgui_cloud.gui import style
from imgui_cloud.gui.login import LoginPanel
from imgui_cloud.gui.style import field_row

COUNT_ZONES_OFFERED = 6


def number_quota(limit: float) -> str:
    """A quota limit as text, with unread shown as a question mark."""
    return "?" if limit < 0 else f"{limit:.0f}"


class CloudPanel:
    """
    Full cloud workflow as one embeddable widget.

    Parameters
    ----------
    dir_input : str
        Pre-filled local input folder, e.g. the dataset already open in a viewer.
    dir_output : str
        Pre-filled local output folder; defaults to ``<input parent>/results``.
    theme : imgui_data_loader.Theme, optional
        Palette shared with the file dialog and the host app.
    profile_name : str, optional
        Profile to sign in with; the active one when omitted.
    """

    def __init__(
        self,
        dir_input: str = "",
        dir_output: str = "",
        theme: Theme | None = None,
        profile_name: str | None = None,
    ):
        self.theme = theme or Theme.dark()
        self.login = LoginPanel(profile_name=profile_name, theme=self.theme)
        self.config = config_module.default_config(
            dir_input=dir_input,
            dir_output=dir_output
            or (str(Path(dir_input).parent / "results") if dir_input else ""),
            name=Path(dir_input).name if dir_input else "run",
        )
        self.run: run_module.CloudRun | None = None
        self.text_params = "{}"
        self.error_params = ""
        self.records = history.load_all(limit=25)
        self._time_records = 0.0
        self._pending_pick = None
        self._pick_target = ""
        self._tab_next = ""

    def set_input(self, dir_input: str) -> None:
        """Point the panel at a dataset, filling in a matching output folder."""
        self.config.io.input = str(dir_input)
        self.config.io.name = Path(dir_input).name or "run"
        if not self.config.io.output:
            self.config.io.output = str(Path(dir_input).parent / "results")

    def draw(self) -> None:
        """Draw one frame of the whole panel."""
        self._poll_pick()
        if not self.login.signed_in:
            self.login.draw()
            imgui.dummy(imgui.ImVec2(0, 8))
            imgui.separator()
            imgui.text_colored(
                to_vec4(self.theme.text_dim),
                "Sign in to upload data and start a worker.",
            )
            return

        if imgui.begin_tab_bar("##cloudtabs"):
            if imgui.begin_tab_item(f"{fa.ICON_FA_ROCKET}  Run")[0]:
                self._draw_run()
                imgui.end_tab_item()
            if imgui.begin_tab_item(f"{fa.ICON_FA_FOLDER_OPEN}  Data")[0]:
                self._draw_data()
                imgui.end_tab_item()
            if imgui.begin_tab_item(f"{fa.ICON_FA_MICROCHIP}  Machine")[0]:
                self._draw_machine()
                imgui.end_tab_item()
            if imgui.begin_tab_item(f"{fa.ICON_FA_LIST}  Runs")[0]:
                self._draw_runs()
                imgui.end_tab_item()
            flags_account = 0
            if self._tab_next == "account":
                flags_account = imgui.TabItemFlags_.set_selected
                self._tab_next = ""
            if imgui.begin_tab_item(f"{fa.ICON_FA_USER}  Account", None, flags_account)[
                0
            ]:
                self.login.draw()
                imgui.end_tab_item()
            imgui.end_tab_bar()

    def _poll_pick(self) -> None:
        """Collect a native folder picker result into whichever field asked."""
        if self._pending_pick is None or not self._pending_pick.ready():
            return
        result = self._pending_pick.result()
        self._pending_pick = None
        path = result[0] if isinstance(result, list) and result else result
        if not path:
            return
        if self._pick_target == "input":
            self.set_input(path)
        elif self._pick_target == "output":
            self.config.io.output = path

    def _browse(self, target: str, title: str) -> None:
        """Open the OS folder picker for ``target`` (``input`` / ``output``)."""
        start = self.config.io.input or ""
        self._pending_pick = pfd.select_folder(title, start)
        self._pick_target = target

    def _draw_data(self) -> None:
        """Input, output, run name, pipeline and its parameters."""
        theme = self.theme
        field_row("Input folder", "Uploaded to the bucket, then pulled onto the box.")
        changed, value = imgui.input_text("##input", self.config.io.input)
        if changed:
            self.config.io.input = value
        push_button_style(theme, primary=False)
        if imgui.button(
            f"{fa.ICON_FA_FOLDER_OPEN}  Choose input...", imgui.ImVec2(200, 0)
        ):
            self._browse("input", "Select the data to upload")
        pop_button_style()

        field_row("Output folder", "Results are downloaded here when the run finishes.")
        changed, value = imgui.input_text("##output", self.config.io.output)
        if changed:
            self.config.io.output = value
        push_button_style(theme, primary=False)
        if imgui.button(
            f"{fa.ICON_FA_FOLDER_OPEN}  Choose output...", imgui.ImVec2(200, 0)
        ):
            self._browse("output", "Select where results should land")
        pop_button_style()

        field_row(
            "Run name", "Used for the instance name and the dated results folder."
        )
        changed, value = imgui.input_text("##name", self.config.io.name)
        if changed:
            self.config.io.name = value

        if self.config.io.input:
            imgui.text_colored(
                to_vec4(theme.text_dim),
                f"results -> {config_module.resolve_output_dir(self.config)}",
            )

        imgui.dummy(imgui.ImVec2(0, 8))
        imgui.text_colored(to_vec4(theme.accent), "Pipeline")
        imgui.separator()
        field_row(
            "Pipeline", "Runs on the worker after its requirements are installed."
        )
        if imgui.begin_combo("##pipeline", self.config.job.pipeline):
            for name in pipelines.available():
                selected, _ = imgui.selectable(name, name == self.config.job.pipeline)
                if selected:
                    self.config.job.pipeline = name
            imgui.end_combo()
        try:
            spec = pipelines.get(self.config.job.pipeline)
            imgui.text_colored(to_vec4(theme.text_dim), spec.description)
            imgui.text_colored(
                to_vec4(theme.text_dim), f"installs: {', '.join(spec.pip)}"
            )
        except KeyError as e:
            imgui.text_colored(to_vec4(theme.err), str(e))

        imgui.dummy(imgui.ImVec2(0, 6))
        imgui.text_colored(
            to_vec4(theme.text_dim), "Parameters (JSON, forwarded to the pipeline)"
        )
        changed, value = imgui.input_text_multiline(
            "##params", self.text_params, imgui.ImVec2(-1, 120)
        )
        if changed:
            self.text_params = value
            try:
                parsed = json.loads(value or "{}")
                if not isinstance(parsed, dict):
                    raise ValueError("parameters must be a JSON object")
                self.config.job.params = parsed
                self.error_params = ""
            except Exception as e:
                self.error_params = str(e)
        if self.error_params:
            imgui.text_colored(to_vec4(theme.err), self.error_params)

    def _draw_machine(self) -> None:
        """Pick the GPU, then the disks and the lifetime cap."""
        theme = self.theme
        machine = self.config.machine

        style.section(
            theme,
            style.icon("ICON_FA_MICROCHIP"),
            "GPU",
            f"quota read from {self.login.profile.region}",
        )
        self._draw_gpu_table()
        imgui.dummy(style.em2(0, 0.3))
        self._draw_gpu_detail()

        style.section(theme, style.icon("ICON_FA_HARD_DRIVE"), "Disks and lifetime")
        changed, value = imgui.checkbox(
            "Spot (preemptible, ~60% cheaper)", machine.spot
        )
        if changed:
            machine.spot = value
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Google can reclaim a spot box at any time; the run fails and "
                "nothing is lost but the compute already spent."
            )

        field_row(
            "Scratch disk GB",
            "Mounted at /mnt/data; must hold the input and the output.",
        )
        changed, value = imgui.input_int("##datadisk", machine.data_disk_gb, 100, 500)
        if changed:
            machine.data_disk_gb = max(10, value)

        field_row("Boot disk GB", "Holds the OS, CUDA and the installed pipeline.")
        changed, value = imgui.input_int("##bootdisk", machine.boot_disk_gb, 50, 100)
        if changed:
            machine.boot_disk_gb = max(50, value)

        field_row(
            "Max runtime (min)", "Hard cap: the instance deletes itself at this age."
        )
        changed, value = imgui.input_int("##maxrun", machine.max_runtime_min, 30, 120)
        if changed:
            machine.max_runtime_min = max(5, value)

        changed, value = imgui.checkbox(
            "Keep the scratch disk after teardown", machine.keep_data_disk
        )
        if changed:
            machine.keep_data_disk = value
        changed, value = imgui.checkbox(
            "Leave the instance running (debugging)", machine.keep_instance
        )
        if changed:
            machine.keep_instance = value
        if machine.keep_instance:
            style.wrapped(
                f"{fa.ICON_FA_TRIANGLE_EXCLAMATION}  billing continues until you run "
                "'imgui-cloud down --all'",
                theme.warn,
            )

        imgui.dummy(style.em2(0, 0.4))
        imgui.separator()
        rate = machine.cost_per_hour_estimate()
        imgui.text_colored(
            to_vec4(theme.accent),
            f"~${rate:.2f}/hour  ->  ~${rate * machine.max_runtime_min / 60:.2f} "
            f"if it runs the full {machine.max_runtime_min} min",
        )
        imgui.text_colored(
            to_vec4(theme.text_dim), "list-price estimate, us-central1; excludes egress"
        )

    def choose_gpu(self, option: config_module.GpuOption) -> None:
        """Adopt a catalog entry: machine type, accelerator and how many."""
        self.config.machine.machine_type = option.machine_type
        self.config.machine.accelerator_type = option.accelerator_type
        self.config.machine.accelerator_count = option.count

    def _draw_gpu_table(self) -> None:
        """One row per offered box: what it is, what it costs, what quota it needs."""
        flags = (
            imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.sizing_stretch_prop
        )
        if not imgui.begin_table("##gpus", 5, flags):
            return
        imgui.table_setup_column("GPU", imgui.TableColumnFlags_.width_stretch, 2.0)
        imgui.table_setup_column("machine", imgui.TableColumnFlags_.width_stretch, 2.0)
        imgui.table_setup_column("vCPU / RAM")
        imgui.table_setup_column("~$/hour")
        imgui.table_setup_column("quota")
        imgui.table_headers_row()
        for option in config_module.GPUS:
            self._draw_gpu_row(option)
        imgui.end_table()

    def _draw_gpu_row(self, option: config_module.GpuOption) -> None:
        """A selectable row, red in the quota column when the project cannot run it."""
        theme = self.theme
        machine = self.config.machine
        chosen = (
            option.machine_type == machine.machine_type
            and option.accelerator_type == machine.accelerator_type
        )
        limit, _ = config_module.quota_effective(
            self.login.state.quotas, option, machine.spot
        )
        rate = option.usd_per_hour * (0.4 if machine.spot else 1.0)

        imgui.table_next_row()
        imgui.table_next_column()
        imgui.push_id(f"{option.machine_type}-{option.accelerator_type}")
        selected, _ = imgui.selectable(
            option.label, chosen, imgui.SelectableFlags_.span_all_columns
        )
        imgui.pop_id()
        if selected:
            self.choose_gpu(option)
        imgui.table_next_column()
        imgui.text_colored(to_vec4(theme.text_dim), option.machine_type)
        imgui.table_next_column()
        imgui.text_colored(
            to_vec4(theme.text_dim), f"{option.vcpu} / {option.memory_gb} GB"
        )
        imgui.table_next_column()
        imgui.text_colored(to_vec4(theme.text_dim), f"${rate:.2f}")
        imgui.table_next_column()
        if limit < 0:
            imgui.text_colored(to_vec4(theme.text_dim), "?")
        elif limit >= option.count:
            imgui.text_colored(to_vec4(theme.ok), f"{limit:.0f}")
        else:
            imgui.text_colored(to_vec4(theme.err), f"{limit:.0f}")

    def _draw_gpu_detail(self) -> None:
        """Whether the chosen GPU can run, and what to do when it cannot."""
        theme = self.theme
        machine = self.config.machine
        profile = self.login.profile
        option = config_module.gpu_option(
            machine.machine_type, machine.accelerator_type
        )
        if option is None:
            style.wrapped(
                f"{machine.machine_type} is not in the catalog; sent as typed.",
                theme.warn,
            )
            return

        with style.card(theme, "##gpudetail"):
            imgui.text_colored(
                to_vec4(theme.accent), f"{option.label}   ({option.machine_type})"
            )
            limit, metric = config_module.quota_effective(
                self.login.state.quotas, option, machine.spot
            )
            if limit < 0:
                style.wrapped(f"{metric}: not read yet", theme.text_dim)
            elif limit >= option.count:
                style.wrapped(
                    f"{metric} = {limit:.0f} in {profile.region}, needs "
                    f"{option.count}",
                    theme.ok,
                )
            else:
                style.wrapped(
                    f"{metric} = {limit:.0f} in {profile.region}, needs "
                    f"{option.count}",
                    theme.err,
                )
            self._draw_quota_detail(option)
            self._draw_gpu_zones(option)

    def _draw_quota_detail(self, option: config_module.GpuOption) -> None:
        """The numbers behind the verdict, and the request that fixes them."""
        theme = self.theme
        state = self.login.state
        regular = state.quota_for(option.metric_quota)
        spot = state.quota_for(option.metric_quota_spot)
        style.wrapped(
            f"{option.metric_quota}: {number_quota(regular)}   |   "
            f"{option.metric_quota_spot}: {number_quota(spot)}",
            theme.text_dim,
        )
        style.help_marker(
            theme,
            "Preemptible quota is opt-in and most projects have none; spot VMs "
            "then count against the ordinary quota, so a zero next to "
            "PREEMPTIBLE_ is not what stops a run.",
        )
        cap = state.gpus_all_regions
        if cap == 0:
            style.wrapped("GPUS_ALL_REGIONS = 0, so no region works", theme.err)
            style.help_marker(
                theme,
                "That quota is counted across every region at once. While it is "
                "zero no GPU starts anywhere, and switching region cannot help. "
                "Zero is the default for a new project, not a fault.",
            )
        elif cap > 0:
            style.wrapped(f"GPUS_ALL_REGIONS: {cap:.0f}", theme.text_dim)
        self._draw_quota_ask(option)
        self._draw_quota_regions(option)

    def quota_asks(self, option: config_module.GpuOption) -> list:
        """The ``(metric, amount, region)`` increases this GPU still needs."""
        state = self.login.state
        asks = []
        if 0 <= state.gpus_all_regions < option.count:
            asks.append(("GPUS_ALL_REGIONS", float(option.count), ""))
        limit, _ = config_module.quota_effective(
            state.quotas, option, self.config.machine.spot
        )
        if 0 <= limit < option.count:
            asks.append(
                (option.metric_quota, float(option.count), self.login.profile.region)
            )
        return asks

    def _draw_quota_ask(self, option: config_module.GpuOption) -> None:
        """Exactly what to ask Google for, and the button that asks."""
        theme = self.theme
        login = self.login
        asks = self.quota_asks(option)
        if not asks:
            return
        imgui.dummy(style.em2(0, 0.2))
        imgui.text_colored(to_vec4(theme.accent), "Ask Google for")
        for metric, value, region in asks:
            style.wrapped(
                f"    {metric} = {value:.0f}    {region or 'project-wide'}", theme.text
            )
        imgui.begin_disabled(bool(login.busy))
        with style.button_style(theme, primary=True):
            asked = imgui.button(
                f"{fa.ICON_FA_PAPER_PLANE}  Ask Google"
                f"{' for both' if len(asks) > 1 else ''}",
                style.em2(14, 2.0),
            )
        imgui.end_disabled()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Files the increase through the Cloud Quotas API, turning that "
                "API on first. Small increases are granted in minutes."
            )
        if asked:
            login.ask_for_quota(asks)
        imgui.same_line()
        self._draw_quota_link(option.metric_quota)
        for answer in login.quota_answers:
            style.wrapped(f"{fa.ICON_FA_CIRCLE_CHECK}  {answer}", theme.ok)

    def _draw_quota_regions(self, option: config_module.GpuOption) -> None:
        """Regions that already allow this GPU, each one click away."""
        theme = self.theme
        info = self.login.state.info_for(option.metric_quota)
        if info is None:
            return
        regions = info.regions_with(option.count)
        if not regions:
            return
        style.wrapped(f"Regions that allow {option.count}:", theme.ok)
        for region in regions[:COUNT_ZONES_OFFERED]:
            imgui.push_id(f"region-{region}")
            with style.button_style(theme, primary=False):
                taken = imgui.button(region)
            imgui.pop_id()
            if taken:
                self.use_region(region, option)
            imgui.same_line()
        imgui.new_line()

    def use_region(self, region: str, option: config_module.GpuOption) -> None:
        """Move the profile to a zone in ``region`` that carries this GPU."""
        zones = [
            zone
            for zone in self.login.state.zones_for(option.accelerator_type)
            if zone.startswith(f"{region}-")
        ]
        self.use_zone(zones[0] if zones else f"{region}-a")

    def _draw_quota_link(self, metric: str) -> None:
        """The console quota page, and the metric to filter it by."""
        imgui.push_id(f"quotalink-{metric}")
        with style.button_style(self.theme, primary=False):
            clicked = imgui.button(f"{fa.ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE}  Console")
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"the quota page, filtered by hand for {metric}")
        imgui.same_line()
        with style.button_style(self.theme, primary=False):
            copied = imgui.button(f"{fa.ICON_FA_COPY}  Copy {metric}")
        imgui.pop_id()
        if copied:
            imgui.set_clipboard_text(metric)
        if clicked:
            webbrowser.open(
                setup_module.url_console(
                    "/iam-admin/quotas", self.login.profile.project_id
                )
            )

    def use_zone(self, zone: str) -> None:
        """Move the profile to ``zone`` and remember it."""
        self.login.profile.zone = zone
        credentials_module.save_profile(self.login.profile)

    def _draw_gpu_zones(self, option: config_module.GpuOption) -> None:
        """Whether the profile's zone has this GPU, and one click to one that does."""
        theme = self.theme
        profile = self.login.profile
        zones = self.login.state.zones_for(option.accelerator_type)
        if not zones:
            style.wrapped(
                f"no zone list for {option.accelerator_type} yet; "
                f"{profile.zone} is used as set.",
                theme.text_dim,
            )
            return
        if profile.zone in zones:
            style.wrapped(f"{profile.zone} carries it.", theme.text_dim)
            return
        style.wrapped(
            f"{profile.zone} has no {option.gpu}. Zones in this project that do:",
            theme.warn,
        )
        for zone in zones[:COUNT_ZONES_OFFERED]:
            imgui.push_id(f"zone-{zone}")
            with style.button_style(theme, primary=False):
                taken = imgui.button(zone)
            imgui.pop_id()
            if taken:
                self.use_zone(zone)
            imgui.same_line()
        imgui.new_line()

    def use_recommended_gpu(self) -> bool:
        """Switch to the cheapest box this project can actually run."""
        state = self.login.state
        better = config_module.recommend_gpu(
            state.quotas,
            state.zones_by_accelerator,
            self.login.profile.zone,
            self.config.machine.spot,
        )
        if better is None:
            return False
        self.choose_gpu(better)
        zones = state.zones_for(better.accelerator_type)
        if zones and self.login.profile.zone not in zones:
            self.use_zone(zones[0])
        return True

    def _draw_run(self) -> None:
        """Everything one run needs, on one screen, with the button at the end."""
        with style.card(self.theme, "##runcard"):
            self._draw_run_form()
        self._draw_run_state()

    def problems_profile(self) -> list:
        """What the account still owes this run, in the order worth fixing."""
        profile = self.login.profile
        missing = []
        if not profile.project_id:
            missing.append("no project picked yet")
        if not profile.bucket:
            missing.append("no staging bucket yet")
        return missing or list(self.login.status.problems)

    def _draw_run_form(self) -> None:
        """Data, pipeline, GPU, what it will cost, and what is still missing."""
        theme = self.theme
        style.section(theme, style.icon("ICON_FA_ROCKET"), "Start a run")

        field_row("Data", "Folder uploaded to the bucket, then pulled onto the box.")
        changed, value = imgui.input_text("##runinput", self.config.io.input)
        if changed:
            self.config.io.input = value
            if value and not self.config.io.output:
                self.config.io.output = str(Path(value).parent / "results")
        with style.button_style(theme, primary=False):
            if imgui.button(
                f"{fa.ICON_FA_FOLDER_OPEN}  Choose folder", style.em2(12, 1.7)
            ):
                self._browse("input", "Select the data to upload")

        field_row("Pipeline", "Runs on the worker once its requirements are in.")
        if imgui.begin_combo("##runpipeline", self.config.job.pipeline):
            for name in pipelines.available():
                selected, _ = imgui.selectable(name, name == self.config.job.pipeline)
                if selected:
                    self.config.job.pipeline = name
            imgui.end_combo()

        self._draw_gpu_choice()
        self._draw_run_summary()

        problems_account = self.problems_profile()
        problems = self.config.validate() + problems_account
        for problem in problems:
            style.wrapped(f"{fa.ICON_FA_TRIANGLE_EXCLAMATION}  {problem}", theme.err)
        if problems_account:
            with style.button_style(theme, primary=False):
                if imgui.button(
                    f"{fa.ICON_FA_USER}  Fix it on the Account tab", style.em2(16, 1.8)
                ):
                    self._tab_next = "account"
        imgui.dummy(style.em2(0, 0.3))
        running = self.run is not None and not self.run.state.is_terminal
        imgui.begin_disabled(bool(problems) or running)
        with style.button_style(theme, primary=True):
            if imgui.button(f"{fa.ICON_FA_ROCKET}  Launch run", style.em2(18, 2.4)):
                self.launch()
        imgui.end_disabled()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Quota is advice, not a gate: if it looks wrong, launch anyway "
                "and Google's refusal names the quota it wants."
            )

    def launch(self) -> None:
        """Start the run described by the form."""
        self.run = run_module.CloudRun(self.config, profile=self.login.profile)
        self.run.start()

    def _draw_gpu_choice(self) -> None:
        """The GPU in one line, with what its quota says and how to fix it."""
        theme = self.theme
        machine = self.config.machine
        state = self.login.state
        option = config_module.gpu_option(
            machine.machine_type, machine.accelerator_type
        )
        field_row("GPU", "The Machine tab has the whole catalog and the detail.")
        label = option.label if option else machine.machine_type
        if imgui.begin_combo("##gpuchoice", label):
            for entry in config_module.GPUS:
                selected, _ = imgui.selectable(
                    f"{entry.label}   ({entry.machine_type})", entry is option
                )
                if selected:
                    self.choose_gpu(entry)
            imgui.end_combo()
        if option is None:
            return

        limit, metric = config_module.quota_effective(
            state.quotas, option, machine.spot
        )
        if limit < 0:
            style.desc(theme, f"{metric}: not read yet")
            return
        if limit >= option.count:
            style.desc(
                theme, f"{metric} = {limit:.0f} in {self.login.profile.region}"
            )
            return
        style.wrapped(
            f"{metric} = {limit:.0f} in {self.login.profile.region}, and this "
            f"needs {option.count}.",
            theme.err,
        )
        self._draw_gpu_alternative(option)

    def _draw_gpu_alternative(self, option: config_module.GpuOption) -> None:
        """One click to a box with quota, or the honest answer that none has any."""
        theme = self.theme
        state = self.login.state
        better = config_module.recommend_gpu(
            state.quotas,
            state.zones_by_accelerator,
            self.login.profile.zone,
            self.config.machine.spot,
        )
        if better is None:
            style.wrapped("Nothing in the catalog has quota here yet.", theme.warn)
            self._draw_quota_ask(option)
            return
        with style.button_style(theme, primary=False):
            if imgui.button(
                f"{fa.ICON_FA_WAND_MAGIC_SPARKLES}  Use {better.label} instead",
                style.em2(18, 1.8),
            ):
                self.use_recommended_gpu()


    def _draw_run_summary(self) -> None:
        """Where it lands, what it runs on, and roughly what it costs."""
        theme = self.theme
        profile = self.login.profile
        machine = self.config.machine
        rate = machine.cost_per_hour_estimate()
        imgui.dummy(style.em2(0, 0.2))
        if self.config.io.input:
            style.wrapped(
                f"results -> {config_module.resolve_output_dir(self.config)}",
                theme.text_dim,
            )
        style.wrapped(
            f"gs://{profile.bucket}/{profile.prefix}   ·   {profile.zone}   ·   "
            f"{'spot' if machine.spot else 'on demand'}",
            theme.text_dim,
        )
        imgui.text_colored(
            to_vec4(theme.accent),
            f"~${rate:.2f}/hour, and it deletes itself after "
            f"{machine.max_runtime_min} min (~${rate * machine.max_runtime_min / 60:.2f})",
        )

    def _draw_run_state(self) -> None:
        """The live run: phase, progress, instance, cost so far, worker log."""
        run = self.run
        if run is None:
            return
        theme = self.theme
        state = run.state
        imgui.dummy(style.em2(0, 0.4))
        imgui.separator()
        color_phase = {
            history.PHASE_DONE: theme.ok,
            history.PHASE_FAILED: theme.err,
            history.PHASE_CANCELLED: theme.warn,
        }.get(state.phase, theme.accent)
        imgui.text_colored(to_vec4(color_phase), f"{state.phase.upper()}  {run.run_id}")
        style.wrapped(state.message or "...", theme.text)

        if state.phase in (history.PHASE_UPLOADING, history.PHASE_DOWNLOADING):
            imgui.progress_bar(state.fraction, imgui.ImVec2(-1, 0))
        if state.instance_status:
            imgui.text_colored(
                to_vec4(theme.text_dim), f"instance: {state.instance_status}"
            )
        record = state.record
        if record is not None and record.time_started:
            imgui.text_colored(
                to_vec4(theme.text_dim),
                f"elapsed {record.duration_s / 60:.1f} min  |  ~${record.cost_estimate():.2f}",
            )

        if not state.is_terminal:
            with style.button_style(theme, primary=False):
                if imgui.button(
                    f"{fa.ICON_FA_STOP}  Cancel and tear down", style.em2(15, 1.8)
                ):
                    run.cancel()

        if state.log_tail:
            imgui.dummy(style.em2(0, 0.3))
            imgui.text_colored(to_vec4(theme.text_dim), "worker log")
            if imgui.begin_child(
                "##worklog", imgui.ImVec2(0, style.em(12)), imgui.ChildFlags_.borders
            ):
                imgui.text_unformatted(state.log_tail)
                if imgui.get_scroll_y() >= imgui.get_scroll_max_y() - 1.0:
                    imgui.set_scroll_here_y(1.0)
            imgui.end_child()

    def _draw_runs(self) -> None:
        """Run history, with a teardown button for anything still alive."""
        theme = self.theme
        if time.time() - self._time_records > 5.0:
            self.records = history.load_all(limit=25)
            self._time_records = time.time()

        push_button_style(theme, primary=False)
        if imgui.button(f"{fa.ICON_FA_ROTATE}  Refresh", imgui.ImVec2(120, 0)):
            self.records = history.load_all(limit=25)
        imgui.same_line()
        if imgui.button(
            f"{fa.ICON_FA_TRASH}  Tear down all workers", imgui.ImVec2(240, 0)
        ):
            run_module.teardown_orphans(self.login.profile)
            self.records = history.load_all(limit=25)
        pop_button_style()

        imgui.dummy(imgui.ImVec2(0, 6))
        if not self.records:
            imgui.text_colored(to_vec4(theme.text_dim), "no runs yet")
            return

        flags = (
            imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.resizable
        )
        if imgui.begin_table("##runs", 5, flags):
            for column in ("run", "when", "phase", "pipeline", "cost"):
                imgui.table_setup_column(column)
            imgui.table_headers_row()
            for record in self.records:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(record.run_id)
                imgui.table_next_column()
                imgui.text(
                    time.strftime("%m-%d %H:%M", time.localtime(record.time_created))
                )
                imgui.table_next_column()
                color = {
                    history.PHASE_DONE: theme.ok,
                    history.PHASE_FAILED: theme.err,
                }.get(record.phase, theme.text_dim)
                imgui.text_colored(to_vec4(color), record.phase)
                imgui.table_next_column()
                imgui.text(record.pipeline)
                imgui.table_next_column()
                imgui.text(f"${record.cost_estimate():.2f}")
            imgui.end_table()


def draw_cloud_tab(panel_state: dict, dir_input: str = "") -> None:
    """
    Draw the panel from a host that keeps only a dict of widget state.

    Mirrors ``mbo_utilities.gui._biohpc.draw_biohpc_tab``: the host calls this
    each frame and the panel is created on first use, so a viewer needs one
    import and one line to gain the whole workflow.

    Parameters
    ----------
    panel_state : dict
        Any dict owned by the host; the panel is cached under ``"cloud_panel"``.
    dir_input : str
        Dataset to pre-fill the first time the panel is created.
    """
    panel = panel_state.get("cloud_panel")
    if panel is None:
        panel = CloudPanel(dir_input=dir_input)
        panel_state["cloud_panel"] = panel
    elif dir_input and not panel.config.io.input:
        panel.set_input(dir_input)
    panel.draw()
