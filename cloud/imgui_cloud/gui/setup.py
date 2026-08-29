"""The first-run checklist: what a new project still needs, in order.

Each step reports its own state from the last sign-in probe and from whatever
the account panel has read back, so the list is a live status board: press a
step's button, and it ticks over. The first unfinished step is the one that
shows its controls.
"""

from __future__ import annotations


import webbrowser
from dataclasses import dataclass, field

from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui
from imgui_data_loader import Theme, pop_button_style, push_button_style, to_vec4


from imgui_cloud import account
from imgui_cloud import config as config_module
from imgui_cloud.credentials import CloudProfile, ProfileStatus
from imgui_cloud.gui.style import em2, wrapped

URL_CONSOLE = account.URL_CONSOLE
APIS_REQUIRED = account.APIS_REQUIRED

ACTION_INSTALL = "install"
ACTION_SIGNIN = "signin"
ACTION_PROJECT = "project"
ACTION_APIS = "apis"
ACTION_BUCKET = "bucket"
ACTION_QUOTA = "quota"
ACTION_VERIFY = "verify"


def url_console(path: str, project: str = "") -> str:
    """A console page, scoped to ``project`` when one is known."""
    return f"{URL_CONSOLE}{path}" + (f"?project={project}" if project else "")


@dataclass
class SetupState:
    """What the account panel has read back from Google, beyond the probe."""

    gcloud: bool = False
    email: str = ""
    projects: list = field(default_factory=list)
    apis: dict = field(default_factory=dict)
    buckets: list = field(default_factory=list)
    accounts_service: list = field(default_factory=list)
    quotas: dict = field(default_factory=dict)
    quotas_project: dict = field(default_factory=dict)
    quota_infos: dict = field(default_factory=dict)
    zones_by_accelerator: dict = field(default_factory=dict)

    @property
    def apis_ok(self) -> bool:
        """Whether the APIs a run cannot do without are on."""
        return all(self.apis.get(api) for api in account.APIS_REQUIRED)

    def api_ok(self, api: str):
        """One API's state: True, False, or None when it has not been read."""
        return self.apis.get(api)

    def info_for(self, metric: str):
        """The Cloud Quotas entry for a metric, or None when unread."""
        return self.quota_infos.get(metric)

    @property
    def quota_a100(self) -> float:
        """Best A100 limit the region reported; -1 when nothing has been read."""
        found = [
            limit
            for metric, limit in self.quotas.items()
            if account.NAME_A100 in metric
        ]
        return max(found, default=-1.0)

    @property
    def zones(self) -> list:
        """Every zone this project can create one of the offered GPUs in."""
        offered = {option.accelerator_type for option in config_module.GPUS}
        found = {
            zone
            for accelerator, zones in self.zones_by_accelerator.items()
            if accelerator in offered
            for zone in zones
        }
        return sorted(found)

    @property
    def gpus_all_regions(self) -> float:
        """Project-wide GPU cap, which gates every region; -1 when unread."""
        return self.quotas_project.get("GPUS_ALL_REGIONS", -1.0)

    def quota_for(self, metric: str) -> float:
        """One metric's limit, -1 when the region's quotas are not known."""
        return self.quotas.get(metric, -1.0)

    def zones_for(self, accelerator_type: str) -> list:
        """Zones this project can create ``accelerator_type`` in."""
        return self.zones_by_accelerator.get(accelerator_type, [])


@dataclass
class Step:
    """One line of the checklist, with the console page and command that do it."""

    title: str
    body: str
    action: str = ""
    done: bool = False
    url: str = ""
    label_url: str = "Open console"
    command: str = ""
    note: str = ""


def steps_for(
    profile: CloudProfile, status: ProfileStatus, state: SetupState | None = None
) -> list[Step]:
    """
    The checklist for ``profile``, with each step's state read from the probes.

    Parameters
    ----------
    profile : imgui_cloud.credentials.CloudProfile
        Settings being edited in the account panel.
    status : imgui_cloud.credentials.ProfileStatus
        Result of the last sign-in probe.
    state : SetupState, optional
        What the panel has listed from the account; defaults to nothing known.

    Returns
    -------
    list of Step
        In the order they have to be done.
    """
    state = state or SetupState()
    project = profile.project_id
    bucket = profile.bucket or "BUCKET-NAME"
    signed_in = status.credentials_ok or bool(state.email)
    steps = []

    if not state.gcloud and not signed_in:
        steps.append(
            Step(
                title="Install the Google Cloud SDK",
                body="The one-click sign-in uses gcloud, Google's own CLI.",
                action=ACTION_INSTALL,
                url=account.URL_INSTALL_GCLOUD,
                label_url="Download",
                command=account.command_install_gcloud(),
                note="Or sign in with a key file, below.",
            )
        )

    steps += [
        Step(
            title="Sign in with Google",
            body="Opens your browser. No API key is involved.",
            action=ACTION_SIGNIN,
            done=signed_in,
            command="gcloud auth application-default login",
            note=(
                f"signed in as {state.email}"
                if state.email
                else "Tick every box on Google's consent page, or it refuses."
            ),
        ),
        Step(
            title="Pick your project",
            body="Everything below fills itself in from the project you pick.",
            action=ACTION_PROJECT,
            done=bool(project),
            url=url_console("/projectselector2/home/dashboard"),
            label_url="Projects",
            note=(
                f"{profile.project_name or project}  ·  id {project}  ·  number "
                f"{profile.project_number}"
                if project and profile.project_number
                else "Needs a billing account linked."
            ),
        ),
        Step(
            title="Turn on Compute Engine and Cloud Storage",
            body="New projects have every API off.",
            action=ACTION_APIS,
            done=state.apis_ok if state.apis else status.compute_ok,
            url=url_console("/apis/library/compute.googleapis.com", project),
            label_url="API library",
            command=(
                "gcloud services enable "
                + " ".join(APIS_REQUIRED)
                + (f" --project={project}" if project else "")
            ),
            note="Compute Engine takes a minute to come up.",
        ),
        Step(
            title="Choose the staging bucket",
            body="One per project, in the region you run in.",
            action=ACTION_BUCKET,
            done=status.bucket_ok
            or bool(profile.bucket and profile.bucket in state.buckets),
            url=url_console("/storage/browser", project),
            label_url="Buckets",
            command=(
                f"gcloud storage buckets create gs://{bucket}"
                + (f" --project={project}" if project else "")
                + f" --location={profile.region}"
            ),
            note=f"gs://{bucket} must be unused by anyone, anywhere.",
        ),
        Step(
            title="Request A100 quota",
            body="A new project gets zero GPUs until you ask. The Machine tab asks.",
            action=ACTION_QUOTA,
            done=state.quota_a100 >= 1,
            url=url_console("/iam-admin/quotas", project),
            label_url="Quotas",
            note=(
                f"{profile.region}: {state.quota_a100:.0f} A100 allowed"
                if state.quota_a100 >= 0
                else "Approval is usually minutes, sometimes a day."
            ),
        ),
        Step(
            title="Check the connection",
            body="Probes credentials, bucket and compute in one go.",
            action=ACTION_VERIFY,
            done=status.ok,
        ),
    ]
    return steps


def index_current(steps: list[Step]) -> int:
    """Index of the first unfinished step, or one past the end."""
    return next((i for i, step in enumerate(steps) if not step.done), len(steps))


class SetupGuide:
    """Draws the checklist chrome; the panel draws each step's own controls."""

    def __init__(self, theme: Theme | None = None):
        self.theme = theme or Theme.dark()
        self.copied = ""

    def begin(self, steps: list[Step], open_by_default: bool) -> bool:
        """Open the collapsing header, returning whether the body is visible."""
        done = sum(1 for step in steps if step.done)
        flags = imgui.TreeNodeFlags_.default_open if open_by_default else 0
        header = f"{fa.ICON_FA_LIST_CHECK}  Setup  ({done}/{len(steps)})"
        return imgui.collapsing_header(header, flags)

    def draw_title(self, index: int, step: Step, current: bool) -> None:
        """The numbered title line, dim until the step is current or done."""
        theme = self.theme
        imgui.dummy(em2(0, 0.2))
        if step.done:
            glyph, color = fa.ICON_FA_CIRCLE_CHECK, theme.ok
        elif current:
            glyph, color = fa.ICON_FA_CIRCLE_ARROW_RIGHT, theme.accent
        else:
            glyph, color = fa.ICON_FA_CIRCLE, theme.text_dim
        imgui.text_colored(to_vec4(color), glyph)
        imgui.same_line()
        imgui.text_colored(
            to_vec4(theme.text if (current or step.done) else theme.text_dim),
            f"{index + 1}. {step.title}",
        )

    def draw_body(self, step: Step) -> None:
        """What the step is for, in a sentence or two."""
        wrapped(step.body, self.theme.text_dim)

    def draw_links(self, step: Step) -> None:
        """The console button, the copyable command and the note under a step."""
        theme = self.theme
        if step.url:
            push_button_style(theme, primary=False)
            if imgui.button(
                f"{fa.ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE}  {step.label_url}"
            ):
                webbrowser.open(step.url)
            pop_button_style()
            if imgui.is_item_hovered():
                imgui.set_tooltip(step.url)
            if step.command:
                imgui.same_line()
        if step.command:
            push_button_style(theme, primary=False)
            if imgui.button(f"{fa.ICON_FA_COPY}  Copy command"):
                imgui.set_clipboard_text(step.command)
                self.copied = step.command
            pop_button_style()
            if imgui.is_item_hovered():
                imgui.set_tooltip(step.command)
            if self.copied == step.command:
                imgui.same_line()
                imgui.text_colored(to_vec4(theme.ok), "copied")
        if step.note:
            wrapped(step.note, theme.warn if not step.done else theme.text_dim)
