"""The account panel: sign in with Google, pick a project, let the rest fill itself in.

Everything slow - the browser sign-in, listing projects, enabling APIs, creating
a bucket - runs on a worker thread, so the frame never blocks on the network.
The panel writes to the profile store, never to the app: anything signed in here
is available to the CLI, and vice versa.
"""

from __future__ import annotations


import threading
import time
import webbrowser

from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui, portable_file_dialogs as pfd
from imgui_data_loader import Theme, pop_button_style, push_button_style, to_vec4

from imgui_cloud import account
from imgui_cloud import config as config_module
from imgui_cloud import credentials as credentials_module
from imgui_cloud.gui import setup as setup_module
from imgui_cloud.gui import style
from imgui_cloud.gui.setup import SetupGuide, SetupState
from imgui_cloud.gui.style import field_row

FILTER_JSON = ["JSON", "*.json", "All Files", "*"]


def combo_of(label: str, current: str, options: list, placeholder: str = "") -> str:
    """One-line combo over ``options``; returns the picked value or ``current``."""
    picked = current
    imgui.set_next_item_width(-1)
    if imgui.begin_combo(label, current or placeholder):
        for option in options:
            selected, _ = imgui.selectable(option, option == current)
            if selected:
                picked = option
        imgui.end_combo()
    return picked


class LoginPanel:
    """
    Sign-in and setup for one stored :class:`~imgui_cloud.credentials.CloudProfile`.

    Parameters
    ----------
    profile_name : str, optional
        Profile to edit; the active one when omitted.
    theme : imgui_data_loader.Theme, optional
        Palette, shared with the rest of the app.
    """

    def __init__(self, profile_name: str | None = None, theme: Theme | None = None):
        self.theme = theme or Theme.dark()
        self.profile = credentials_module.load_profile(profile_name)
        self.status = credentials_module.ProfileStatus()
        self.state = SetupState(gcloud=bool(account.path_gcloud()))
        self.guide = SetupGuide(theme=self.theme)
        self.busy = ""
        self.error = ""
        self.warnings: list = []
        self.text_bucket_new = ""
        self.filepath_client_oauth = ""
        self.waiting_console = False
        self.api_pending = ""
        self.quota_asks: list = []
        self.quota_answers: list = []
        self._started = False
        self._time_polled = 0.0
        self._pending_pick = None
        self._pick_target = ""

    @property
    def signed_in(self) -> bool:
        """Whether credentials on this machine work, whatever else is missing."""
        return self.status.credentials_ok or bool(self.state.email)

    @property
    def verified(self) -> bool:
        """Whether the last probe found credentials, bucket and compute all good."""
        return self.status.ok

    @property
    def checking(self) -> bool:
        """Whether a worker thread is busy (the CLI and tests read this)."""
        return bool(self.busy)

    def sign_in(self) -> None:
        """Save the profile and probe the connection on a worker thread."""
        credentials_module.save_profile(self.profile)
        self._start(self._task_verify)

    def sign_out(self) -> None:
        """Forget the in-app sign-in; a gcloud login on this machine is untouched."""
        credentials_module.forget_token_user()
        self.status = credentials_module.ProfileStatus()
        self.state = SetupState(gcloud=bool(account.path_gcloud()))

    def _start(self, task) -> None:
        """Run one bound ``_task_*`` method on a worker thread."""
        if self.busy:
            return
        self.busy = task.__name__.removeprefix("_task_").replace("_", " ")
        threading.Thread(target=self._run, args=(task,), daemon=True).start()

    def _run(self, task) -> None:
        """Worker-thread body: run ``task``, keeping its failure as text."""
        try:
            task()
            self.error = ""
        except Exception as e:
            self.error = str(e)
        finally:
            self.busy = ""

    def _soft(self, label: str, default, call, *args):
        """Run ``call``, recording a failure as a warning and returning ``default``."""
        try:
            return call(*args)
        except Exception as e:
            self.warnings.append(f"{label}: {e}")
            return default

    def _task_signin(self) -> None:
        """Browser sign-in through the Cloud SDK, then read the account."""
        account.login_gcloud()
        self._task_account()

    def _task_signin_console(self) -> None:
        """Hand the sign-in to a console window and watch for it to land."""
        account.login_gcloud(interactive=True)
        self.waiting_console = True

    def _task_signin_oauth(self) -> None:
        """Browser sign-in with a downloaded desktop OAuth client."""
        account.login_oauth_client(self.filepath_client_oauth)
        self._task_account()

    def _task_account(self) -> None:
        """Who is signed in, what projects they have, and what the project holds."""
        credentials, _ = credentials_module.credentials_for(self.profile)
        self.state.email = account.email_of(credentials) or self.state.email
        if not self.profile.user_email:
            self.profile.user_email = self.state.email
        self.state.projects = account.list_projects(credentials)
        self._adopt_project_known()
        self._read_details(credentials)
        credentials_module.save_profile(self.profile)

    def _task_project(self) -> None:
        """Bill API calls to the chosen project, then read what it holds."""
        if self.state.gcloud and not self.profile.filepath_service_account_key:
            self._soft(
                "quota project",
                None,
                account.set_quota_project,
                self.profile.project_id,
            )
        credentials, _ = credentials_module.credentials_for(self.profile)
        self._read_details(credentials)
        credentials_module.save_profile(self.profile)

    def _task_apis(self) -> None:
        """Turn on Compute Engine and Cloud Storage, then re-read the project."""
        credentials, _ = credentials_module.credentials_for(self.profile)
        account.enable_services(credentials, self.profile.project_id)
        self._read_details(credentials)

    def _task_enable_api(self) -> None:
        """Turn on one API by name, then re-read what the project holds."""
        credentials, _ = credentials_module.credentials_for(self.profile)
        account.enable_services(
            credentials, self.profile.project_id, (self.api_pending,)
        )
        self._read_details(credentials)

    def _task_quota(self) -> None:
        """Send the queued quota requests, turning the Quotas API on first."""
        credentials, _ = credentials_module.credentials_for(self.profile)
        account.enable_services(
            credentials, self.profile.project_id, (account.SERVICE_QUOTAS,)
        )
        self.quota_answers = [
            account.request_quota(
                credentials,
                self.profile.project_id,
                info,
                value,
                region=region,
                email=self.profile.user_email,
            )
            for info, value, region in self.quota_asks
        ]
        self._read_details(credentials)

    def _task_bucket(self) -> None:
        """Create the staging bucket in the region the zone sits in."""
        credentials, _ = credentials_module.credentials_for(self.profile)
        account.create_bucket(
            credentials,
            self.profile.project_id,
            self.text_bucket_new,
            self.profile.region,
        )
        self.profile.bucket = self.text_bucket_new
        self._read_details(credentials)
        credentials_module.save_profile(self.profile)

    def _task_verify(self) -> None:
        """Probe credentials, bucket and compute in one go."""
        self.status = credentials_module.check_profile(self.profile)

    def enable_api(self, name: str) -> None:
        """Turn on one API for the chosen project, on a worker thread."""
        self.api_pending = name
        self._start(self._task_enable_api)

    def ask_for_quota(self, asks: list) -> None:
        """Send ``(info, value, region)`` quota requests to Google."""
        self.quota_asks = asks
        self.quota_answers = []
        self._start(self._task_quota)

    def _adopt_project_known(self) -> None:
        """Fill in the number and name of the stored project, or the only project."""
        by_id = {project.project_id: project for project in self.state.projects}
        if not self.profile.project_id and len(self.state.projects) == 1:
            self.choose_project(self.state.projects[0], read_details=False)
        elif self.profile.project_id in by_id:
            self.choose_project(by_id[self.profile.project_id], read_details=False)

    def choose_project(self, project: account.Project, read_details: bool = True):
        """Adopt ``project``, dropping anything that belonged to the previous one."""
        changed = project.project_id != self.profile.project_id
        self.profile.project_id = project.project_id
        self.profile.project_number = project.project_number
        self.profile.project_name = project.display_name
        if not self.profile.service_account_email or changed:
            self.profile.service_account_email = account.service_account_default(
                project.project_number
            )
        if changed:
            self.profile.bucket = ""
            self.state.apis = {}
            self.state.buckets = []
            self.state.quotas = {}
            self.state.quotas_project = {}
            self.state.zones_by_accelerator = {}
        self.text_bucket_new = account.bucket_suggested(project.project_id)
        if read_details:
            self._start(self._task_project)

    def _read_details(self, credentials) -> None:
        """Read the chosen project's APIs, buckets, service accounts and quota."""
        project = self.profile.project_id
        if not project:
            return
        self.warnings = []
        self.state.apis = self._soft(
            "APIs", {}, account.services_state, credentials, project, account.APIS_ALL
        )
        if self.state.apis.get("storage.googleapis.com"):
            self.state.buckets = self._soft(
                "buckets", [], account.list_buckets, credentials, project
            )
            self._choose_bucket_default()
        if self.state.apis.get("compute.googleapis.com"):
            self.state.quotas = self._soft(
                "quota",
                {},
                account.quotas_gpu,
                credentials,
                project,
                self.profile.region,
            )
            self.state.quotas_project = self._soft(
                "project quota", {}, account.quotas_project, credentials, project
            )
            self.state.zones_by_accelerator = self._soft(
                "zones", {}, account.zones_by_accelerator, credentials, project
            )
        if self.state.apis.get(account.SERVICE_QUOTAS):
            self.state.quota_infos = self._soft(
                "quota detail", {}, account.quota_infos, credentials, project
            )
        if self.state.apis.get(account.SERVICE_IAM):
            self.state.accounts_service = self._soft(
                "service accounts",
                [],
                account.list_service_accounts,
                credentials,
                project,
            )
        if self.profile.bucket and self.state.apis_ok:
            self.status = credentials_module.check_profile(self.profile)

    def _choose_bucket_default(self) -> None:
        """Adopt the obvious staging bucket; leave the choice open when unclear."""
        if self.profile.bucket in self.state.buckets:
            return
        matches = [b for b in self.state.buckets if self.profile.prefix in b]
        candidates = matches or self.state.buckets
        self.profile.bucket = candidates[0] if len(candidates) == 1 else ""

    def _poll_pick(self) -> None:
        """Collect the native file picker's result, if one is open."""
        if self._pending_pick is None or not self._pending_pick.ready():
            return
        result = self._pending_pick.result()
        self._pending_pick = None
        paths = result if isinstance(result, list) else ([result] if result else [])
        if not paths:
            return
        if self._pick_target == "key":
            self.profile.filepath_service_account_key = paths[0]
            self._start(self._task_account)
        elif self._pick_target == "oauth":
            self.filepath_client_oauth = paths[0]

    def _browse(self, target: str, title: str) -> None:
        """Open the OS file picker for ``target`` (``key`` / ``oauth``)."""
        self._pending_pick = pfd.open_file(title, "", FILTER_JSON)
        self._pick_target = target

    def _poll_console_signin(self) -> None:
        """Pick up a sign-in finished in the console window, once a second."""
        if not self.waiting_console or self.busy:
            return
        if time.time() - self._time_polled < 1.0:
            return
        self._time_polled = time.time()
        if credentials_module.credentials_available(self.profile):
            self.waiting_console = False
            self._start(self._task_account)

    def draw(self) -> None:
        """Draw one frame of the account panel."""
        self._poll_pick()
        self._poll_console_signin()
        if not self._started:
            self._started = True
            if credentials_module.credentials_available(self.profile):
                self._start(self._task_account)

        theme = self.theme
        imgui.text_colored(to_vec4(theme.accent), f"{fa.ICON_FA_CLOUD}  Google Cloud")
        imgui.same_line()
        if self.state.email:
            imgui.text_colored(
                to_vec4(theme.text_dim),
                f"{self.state.email}  ·  {credentials_module.source_of(self.profile)}",
            )
        else:
            imgui.text_colored(to_vec4(theme.text_dim), "not signed in")
        imgui.separator()

        self._draw_guide()
        self._draw_settings()
        self._draw_status()

    def _draw_guide(self) -> None:
        """The checklist, with the current step's controls under its description."""
        steps = setup_module.steps_for(self.profile, self.status, self.state)
        if not self.guide.begin(steps, open_by_default=not self.status.ok):
            return
        index = setup_module.index_current(steps)
        for i, step in enumerate(steps):
            imgui.push_id(f"step{i}")
            self.guide.draw_title(i, step, current=i == index)
            if i == index:
                imgui.indent(style.em(1.5))
                self.guide.draw_body(step)
                with style.card(self.theme, "##stepcard"):
                    self._draw_controls(step)
                    self.guide.draw_links(step)
                imgui.unindent(style.em(1.5))
            imgui.pop_id()
        if self.busy:
            imgui.text_colored(to_vec4(self.theme.warn), f"{self.busy}...")
        imgui.separator()

    def _draw_controls(self, step) -> None:
        """The buttons and pickers belonging to the step the user is on."""
        imgui.begin_disabled(bool(self.busy))
        if step.action == setup_module.ACTION_INSTALL:
            self._draw_control_install()
        elif step.action == setup_module.ACTION_SIGNIN:
            self._draw_control_signin()
        elif step.action == setup_module.ACTION_PROJECT:
            self._draw_control_project()
        elif step.action == setup_module.ACTION_APIS:
            self._draw_control_apis()
        elif step.action == setup_module.ACTION_BUCKET:
            self._draw_control_bucket()
        elif step.action == setup_module.ACTION_QUOTA:
            self._draw_control_quota()
        elif step.action == setup_module.ACTION_VERIFY:
            self._draw_control_verify()
        imgui.end_disabled()

    def _draw_control_install(self) -> None:
        """Re-check for gcloud, and offer the sign-ins that do not need it."""
        push_button_style(self.theme, primary=False)
        if imgui.button(f"{fa.ICON_FA_ROTATE}  I installed it", style.em2(12, 1.7)):
            self.state.gcloud = bool(account.path_gcloud())
        pop_button_style()
        self._draw_control_signin()

    def _draw_control_signin(self) -> None:
        """One button for the normal path, a tree for the two key-file paths."""
        theme = self.theme
        imgui.begin_disabled(not self.state.gcloud)
        push_button_style(theme, primary=True)
        if imgui.button(
            f"{fa.ICON_FA_RIGHT_TO_BRACKET}  Sign in with Google",
            style.em2(16, 2.0),
        ):
            self._start(self._task_signin)
        pop_button_style()
        imgui.same_line()
        push_button_style(theme, primary=False)
        if imgui.button(
            f"{fa.ICON_FA_TERMINAL}  Sign in in a console", style.em2(14, 2.0)
        ):
            self._start(self._task_signin_console)
        pop_button_style()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Runs the same command in a window of its own, where gcloud can "
                "print the link and say what went wrong."
            )
        imgui.end_disabled()
        if self.state.email:
            imgui.same_line()
            push_button_style(theme, primary=False)
            if imgui.button("Sign out", style.em2(7, 2.0)):
                self.sign_out()
            pop_button_style()
        if self.waiting_console:
            style.wrapped(
                "waiting for the console window; this picks itself up the moment "
                "you finish signing in there.",
                theme.warn,
            )

        if not imgui.tree_node("Other ways to sign in"):
            return
        style.wrapped(
            "A service-account key signs in without a browser - the right choice "
            "for a shared workstation or a headless machine. Create one, download "
            "the JSON, and point this at it.",
            theme.text_dim,
        )
        self._draw_link(
            "Create a key",
            setup_module.url_console(
                "/iam-admin/serviceaccounts", self.profile.project_id
            ),
            "the account -> Keys -> Add key -> Create new key -> JSON",
        )
        field_row("Key file", "Service-account JSON. Blank means the browser sign-in.")
        changed, value = imgui.input_text(
            "##key", self.profile.filepath_service_account_key
        )
        if changed:
            self.profile.filepath_service_account_key = value
        imgui.same_line()
        if imgui.small_button(f"{fa.ICON_FA_FOLDER_OPEN}##browsekey"):
            self._browse("key", "Service-account key")

        imgui.dummy(imgui.ImVec2(0, 6))
        style.wrapped(
            "No Cloud SDK and no key? Create a desktop OAuth client instead, "
            "download its JSON, and sign in through the browser with that.",
            theme.text_dim,
        )
        self._draw_link(
            "Create an OAuth client",
            setup_module.url_console("/auth/clients", self.profile.project_id),
            "Create client -> Application type: Desktop app -> Download JSON",
        )
        field_row("Client JSON", "Downloaded OAuth client for a desktop app.")
        changed, value = imgui.input_text("##oauth", self.filepath_client_oauth)
        if changed:
            self.filepath_client_oauth = value
        imgui.same_line()
        if imgui.small_button(f"{fa.ICON_FA_FOLDER_OPEN}##browseoauth"):
            self._browse("oauth", "OAuth client JSON")
        imgui.begin_disabled(not self.filepath_client_oauth)
        push_button_style(theme, primary=False)
        if imgui.button("Sign in with this client", style.em2(13, 1.7)):
            self._start(self._task_signin_oauth)
        pop_button_style()
        imgui.end_disabled()
        imgui.tree_pop()

    def _draw_control_project(self) -> None:
        """Pick from the listed projects, or take the id out of a console link."""
        theme = self.theme
        labels = [project.label for project in self.state.projects]
        if labels:
            field_row("Project", "Every project this account can see.")
            current = next(
                (
                    p.label
                    for p in self.state.projects
                    if p.project_id == self.profile.project_id
                ),
                "",
            )
            picked = combo_of("##projectpick", current, labels, "select a project")
            if picked != current:
                self.choose_project(self.state.projects[labels.index(picked)])
        else:
            style.wrapped(
                "No projects listed yet - sign in first, or paste the link of the "
                "project you already have open in the console.",
                theme.text_dim,
            )
            field_row("Project id", "Lowercase id, not the display name.")
            changed, value = imgui.input_text("##projectid", self.profile.project_id)
            if changed:
                self.profile.project_id = value

        push_button_style(theme, primary=False)
        if imgui.button(f"{fa.ICON_FA_ROTATE}  Refresh list", style.em2(10, 1.7)):
            self._start(self._task_account)
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_PASTE}  Paste console link", style.em2(13, 1.7)):
            found = account.project_id_in(imgui.get_clipboard_text() or "")
            if found:
                self.choose_project(account.Project(project_id=found))
            else:
                self.error = "no ?project=... in the clipboard"
        pop_button_style()

    def _draw_control_apis(self) -> None:
        """Each API, what turning it on allows, and a button that does it."""
        theme = self.theme
        for api in account.APIS_ALL:
            self._draw_api_row(api)
        imgui.dummy(style.em2(0, 0.2))
        imgui.begin_disabled(not self.profile.project_id)
        with style.button_style(theme, primary=True):
            if imgui.button(
                f"{fa.ICON_FA_TOGGLE_ON}  Turn on the two a run needs",
                style.em2(18, 2.0),
            ):
                self._start(self._task_apis)
        imgui.end_disabled()

    def _draw_api_row(self, api: str) -> None:
        """One API: its state, its name, what it buys, and Enable when it is off."""
        theme = self.theme
        known = self.state.api_ok(api)
        glyph, color = style.status_glyph(theme, known)
        imgui.push_id(api)
        imgui.text_colored(to_vec4(color), glyph)
        imgui.same_line()
        imgui.text_colored(to_vec4(theme.text if known else theme.text_dim), api)
        if api in account.APIS_OPTIONAL:
            imgui.same_line()
            imgui.text_colored(to_vec4(theme.text_dim), "(optional)")
        if known is not True and self.profile.project_id:
            imgui.same_line()
            with style.button_style(theme, primary=False):
                if imgui.button("Enable"):
                    self.enable_api(api)
        style.desc(theme, account.PURPOSE_API.get(api, ""))
        imgui.pop_id()

    def _draw_control_bucket(self) -> None:
        """Pick an existing bucket, or create the suggested one in this region."""
        theme = self.theme
        if self.state.buckets:
            field_row("Bucket", "Existing buckets in this project.")
            picked = combo_of(
                "##bucketpick",
                self.profile.bucket,
                self.state.buckets,
                "select a bucket",
            )
            if picked != self.profile.bucket:
                self.profile.bucket = picked
                credentials_module.save_profile(self.profile)
        field_row("New bucket", f"Created in {self.profile.region}.")
        changed, value = imgui.input_text("##bucketnew", self.text_bucket_new)
        if changed:
            self.text_bucket_new = value
        imgui.begin_disabled(not (self.text_bucket_new and self.profile.project_id))
        push_button_style(theme, primary=True)
        if imgui.button(f"{fa.ICON_FA_PLUS}  Create bucket", style.em2(16, 2.0)):
            self._start(self._task_bucket)
        pop_button_style()
        imgui.end_disabled()

    def _draw_control_quota(self) -> None:
        """Report the A100 limit read back from the region, and re-check it."""
        theme = self.theme
        if self.state.quota_a100 >= 0:
            color = theme.ok if self.state.quota_a100 >= 1 else theme.err
            imgui.text_colored(
                to_vec4(color),
                f"{self.state.quota_a100:.0f} A100 GPUs allowed in "
                f"{self.profile.region}",
            )
        push_button_style(theme, primary=False)
        if imgui.button(f"{fa.ICON_FA_ROTATE}  Re-check quota", style.em2(12, 1.7)):
            self._start(self._task_project)
        pop_button_style()

    def _draw_control_verify(self) -> None:
        """The final probe, and a shortcut back to the whole form."""
        push_button_style(self.theme, primary=True)
        if imgui.button(
            f"{fa.ICON_FA_CIRCLE_CHECK}  Check connection",
            style.em2(16, 2.0),
        ):
            self.sign_in()
        pop_button_style()

    def _draw_link(self, label: str, url: str, note: str) -> None:
        """A console link with the click-path to follow once it opens."""
        push_button_style(self.theme, primary=False)
        if imgui.button(f"{fa.ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE}  {label}"):
            webbrowser.open(url)
        pop_button_style()
        if imgui.is_item_hovered():
            imgui.set_tooltip(url)
        imgui.same_line()
        imgui.text_colored(to_vec4(self.theme.text_dim), note)

    def _draw_settings(self) -> None:
        """Everything the checklist filled in, editable by hand."""
        if not imgui.collapsing_header(f"{fa.ICON_FA_SLIDERS}  All settings"):
            return
        theme = self.theme
        imgui.dummy(style.em2(0, 0.2))
        field_row("Profile", "Named set of settings; switch profiles per project.")
        changed, value = imgui.input_text("##profile", self.profile.name)
        if changed:
            self.profile.name = value

        field_row("Project id", "Owns the instances and the bucket.")
        changed, value = imgui.input_text("##project", self.profile.project_id)
        if changed:
            self.profile.project_id = value
        if self.profile.project_number:
            imgui.text_colored(
                to_vec4(theme.text_dim),
                f"{self.profile.project_name}  ·  number {self.profile.project_number}",
            )

        field_row("Zone", "A100s exist only in some zones.")
        zones = self.state.zones or config_module.ZONES_A100_COMMON
        self.profile.zone = combo_of("##zone", self.profile.zone, zones)

        field_row("Bucket", "Staging bucket for inputs, outputs and logs.")
        changed, value = imgui.input_text("##bucket", self.profile.bucket)
        if changed:
            self.profile.bucket = value

        field_row("Prefix", "Object-name prefix runs are written under.")
        changed, value = imgui.input_text("##prefix", self.profile.prefix)
        if changed:
            self.profile.prefix = value

        field_row("Email", "Recorded on the instance and in the run history.")
        changed, value = imgui.input_text("##email", self.profile.user_email)
        if changed:
            self.profile.user_email = value

        field_row(
            "VM service acct", "Attached to the worker so it can reach the bucket."
        )
        accounts = self.state.accounts_service or [
            account.service_account_default(self.profile.project_number)
        ]
        self.profile.service_account_email = combo_of(
            "##vmsa", self.profile.service_account_email, [a for a in accounts if a]
        )

        field_row("Key file", "Service-account JSON. Blank means the browser sign-in.")
        changed, value = imgui.input_text(
            "##keyfile", self.profile.filepath_service_account_key
        )
        if changed:
            self.profile.filepath_service_account_key = value
        imgui.same_line()
        if imgui.small_button(f"{fa.ICON_FA_FOLDER_OPEN}##browsekey2"):
            self._browse("key", "Service-account key")

        imgui.dummy(imgui.ImVec2(0, 6))
        push_button_style(theme, primary=False)
        if imgui.button("Save", style.em2(7, 1.7)):
            credentials_module.save_profile(self.profile)
        imgui.same_line()
        if imgui.button("Save and check", style.em2(10, 1.7)):
            self.sign_in()
        pop_button_style()

    def _draw_status(self) -> None:
        """The probe result, plus anything that failed while reading the account."""
        theme = self.theme
        imgui.dummy(imgui.ImVec2(0, 4))
        if self.status.ok:
            imgui.text_colored(
                to_vec4(theme.ok), f"{fa.ICON_FA_CIRCLE_CHECK}  {self.status.summary()}"
            )
            imgui.text_colored(
                to_vec4(theme.text_dim),
                f"gs://{self.profile.bucket}/{self.profile.prefix}  |  {self.profile.zone}",
            )
        for problem in self.status.problems:
            style.wrapped(f"{fa.ICON_FA_TRIANGLE_EXCLAMATION}  {problem}", theme.err)
        if self.error:
            style.wrapped(f"{fa.ICON_FA_TRIANGLE_EXCLAMATION}  {self.error}", theme.err)
        for warning in self.warnings:
            style.wrapped(warning, theme.warn)
        for hint in self.hints():
            style.wrapped(f"{fa.ICON_FA_LIGHTBULB}  {hint}", theme.accent)

    def _draw_enable_offer(self, message: str) -> None:
        """An Enable button under any failure that names a switched-off API."""
        api = account.api_disabled_in(message)
        if not api or not self.profile.project_id:
            return
        theme = self.theme
        style.desc(theme, account.PURPOSE_API.get(api, ""))
        imgui.push_id(f"enable-{api}")
        imgui.begin_disabled(bool(self.busy))
        with style.button_style(theme, primary=False):
            if imgui.button(
                f"{fa.ICON_FA_TOGGLE_ON}  Enable {api}", style.em2(20, 1.8)
            ):
                self.enable_api(api)
        imgui.end_disabled()
        imgui.pop_id()

    def hints(self) -> list:
        """Distinct next steps for whatever failed, in the order it failed."""
        found = []
        for message in [self.error, *self.status.problems, *self.warnings]:
            hint = account.hint_for(message)
            if hint and hint not in found:
                found.append(hint)
        return found
