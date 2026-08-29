"""Sign in to Google, then read back the projects, buckets and quota it can see."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

from imgui_cloud import credentials as credentials_module

URL_CONSOLE = "https://console.cloud.google.com"
URL_INSTALL_GCLOUD = "https://cloud.google.com/sdk/docs/install"
API_RESOURCE_MANAGER = "https://cloudresourcemanager.googleapis.com/v3"
API_SERVICE_USAGE = "https://serviceusage.googleapis.com/v1"
API_IAM = "https://iam.googleapis.com/v1"
API_USERINFO = "https://www.googleapis.com/oauth2/v3/userinfo"
API_QUOTAS = "https://cloudquotas.googleapis.com/v1"

SERVICE_COMPUTE = "compute.googleapis.com"
APIS_REQUIRED = (SERVICE_COMPUTE, "storage.googleapis.com")
SERVICE_IAM = "iam.googleapis.com"
SERVICE_QUOTAS = "cloudquotas.googleapis.com"
APIS_OPTIONAL = (SERVICE_IAM, SERVICE_QUOTAS)
APIS_ALL = APIS_REQUIRED + APIS_OPTIONAL

# What turning each one on actually buys, in the panel's own words.
PURPOSE_API = {
    "compute.googleapis.com": (
        "Creates the worker, attaches its disk, and deletes it when the run "
        "ends. Nothing can launch without it."
    ),
    "storage.googleapis.com": (
        "Moves the dataset up to the staging bucket and the results back down."
    ),
    "iam.googleapis.com": (
        "Lists the service accounts a worker can run as. Without it the panel "
        "falls back to the project's default compute account, which is usually "
        "the right one anyway."
    ),
    "cloudquotas.googleapis.com": (
        "Reads your real GPU limits region by region, and asks Google to raise "
        "them from this panel instead of the console."
    ),
}

JUSTIFICATION_QUOTA = (
    "Running a microscopy analysis pipeline on a short-lived GPU worker."
)

PATTERN_API_DISABLED = re.compile(r"([a-z][a-z0-9-]*\.googleapis\.com)")
SUFFIX_METRIC_GPU = "_GPUS"
NAME_A100 = "A100"

TIMEOUT_CALL_S = 60.0
TIMEOUT_LOGIN_S = 300.0
TIMEOUT_ENABLE_S = 240.0

COMMANDS_INSTALL_GCLOUD = {
    "win32": "winget install --id Google.CloudSDK -e",
    "darwin": "brew install --cask google-cloud-sdk",
}
COMMAND_INSTALL_GCLOUD_LINUX = "curl https://sdk.cloud.google.com | bash"

PATTERN_PROJECT = re.compile(r"(?:[?&]project=|/projects/)([a-z][a-z0-9-]{4,29})")

FLAGS_NO_CONSOLE = getattr(subprocess, "CREATE_NO_WINDOW", 0)
FLAGS_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

HINTS = (
    (
        "was not consented",
        "Google's consent page lists each permission with its own tick box, and "
        "they start unticked. Press Select all (or tick every box) so Cloud "
        "Platform access is granted, then sign in again.",
    ),
    (
        "scope is required",
        "Google's consent page lists each permission with its own tick box, and "
        "they start unticked. Press Select all (or tick every box) so Cloud "
        "Platform access is granted, then sign in again.",
    ),
    (
        "problem with web authentication",
        "The browser could not hand the sign-in back to this machine. Use "
        "Sign in in a terminal, where gcloud can print the link itself.",
    ),
    (
        "do not currently have an active account",
        "Nothing is signed in on this machine yet: press Sign in with Google.",
    ),
    (
        "invalid_grant",
        "The saved sign-in has expired or been revoked; sign in again.",
    ),
    (
        "has not been used in project",
        "That API is off for this project - Turn both on, on the API step, enables it.",
    ),
    (
        "billing account",
        "The project has no billing account linked, so APIs cannot be enabled: "
        f"{URL_CONSOLE}/billing/linkedaccount",
    ),
)


def hint_for(message: str) -> str:
    """What to do about a gcloud or API failure, empty when there is nothing to add."""
    lowered = (message or "").lower()
    return next((hint for needle, hint in HINTS if needle in lowered), "")


@dataclass
class Project:
    """One Google Cloud project the signed-in account can see."""

    project_id: str = ""
    project_number: str = ""
    display_name: str = ""

    @property
    def label(self) -> str:
        """``Display name (project-id)``, or just the id when there is no name."""
        if self.display_name and self.display_name != self.project_id:
            return f"{self.display_name}  ({self.project_id})"
        return self.project_id


def api_disabled_in(message: str) -> str:
    """The API a "has not been used / is disabled" failure names, else empty."""
    lowered = (message or "").lower()
    if "has not been used in project" not in lowered and "is disabled" not in lowered:
        return ""
    match = PATTERN_API_DISABLED.search(lowered)
    return match.group(1) if match else ""


def path_gcloud() -> str:
    """Path of the ``gcloud`` executable, empty when the Cloud SDK is absent."""
    return shutil.which("gcloud") or ""


def command_install_gcloud() -> str:
    """One-liner that installs the Cloud SDK on this platform."""
    return COMMANDS_INSTALL_GCLOUD.get(sys.platform, COMMAND_INSTALL_GCLOUD_LINUX)


def run_gcloud(*args: str, timeout: float = TIMEOUT_CALL_S) -> str:
    """
    Run ``gcloud`` with ``args`` and return its stdout.

    Raises
    ------
    RuntimeError
        If the SDK is missing or the command exits non-zero; the message is
        gcloud's own stderr, which is what a user needs to act on.
    """
    path = path_gcloud()
    if not path:
        raise RuntimeError("the Google Cloud SDK (gcloud) is not installed")
    done = subprocess.run(
        [path, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        creationflags=FLAGS_NO_CONSOLE,
    )
    if done.returncode != 0:
        message = (done.stderr or done.stdout).strip()
        raise RuntimeError(message or f"gcloud {' '.join(args)} failed")
    return done.stdout


def login_gcloud(interactive: bool = False) -> None:
    """
    Browser sign-in through the Cloud SDK, writing application default credentials.

    Parameters
    ----------
    interactive : bool
        Run it in a console of its own and return immediately, instead of
        waiting on a hidden one. gcloud can then print the link, and say what
        went wrong, where the person signing in can actually read it.
    """
    if not interactive:
        run_gcloud("auth", "application-default", "login", timeout=TIMEOUT_LOGIN_S)
        return
    path = path_gcloud()
    if not path:
        raise RuntimeError("the Google Cloud SDK (gcloud) is not installed")
    subprocess.Popen(
        [path, "auth", "application-default", "login"],
        creationflags=FLAGS_NEW_CONSOLE,
    )


def login_oauth_client(filepath_client_secret: str) -> None:
    """
    Browser sign-in with a desktop OAuth client downloaded from the console.

    The refresh token is written to the profile store, not to gcloud's own
    credential file, so this never disturbs an existing SDK sign-in.
    """
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        raise RuntimeError(
            "this sign-in needs google-auth-oauthlib: uv pip install google-auth-oauthlib"
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        filepath_client_secret, scopes=credentials_module.SCOPES_LOGIN
    )
    credentials_module.save_token_user(flow.run_local_server(port=0))


def set_quota_project(project_id: str) -> None:
    """Bill API calls made with application default credentials to ``project_id``."""
    run_gcloud("auth", "application-default", "set-quota-project", project_id)


def session_for(credentials):
    """Authorized ``requests`` session for the Google REST endpoints."""
    from google.auth.transport.requests import AuthorizedSession

    return AuthorizedSession(credentials)


def message_of(response) -> str:
    """The API's own error message for a failed response."""
    try:
        return response.json()["error"]["message"]
    except Exception:
        return f"{response.status_code} {response.reason} from {response.url}"


def body_of(response) -> dict:
    """Parsed JSON body, raising the API's error message on a failure status."""
    if response.status_code >= 400:
        raise RuntimeError(message_of(response))
    return response.json()


def projects_from_api(credentials) -> list[Project]:
    """Active projects, straight from the Cloud Resource Manager API."""
    session = session_for(credentials)
    found: list[Project] = []
    token = ""
    while True:
        params = {"query": "state:ACTIVE", "pageSize": 200}
        if token:
            params["pageToken"] = token
        body = body_of(
            session.get(
                f"{API_RESOURCE_MANAGER}/projects:search",
                params=params,
                timeout=TIMEOUT_CALL_S,
            )
        )
        found += [
            Project(
                project_id=item.get("projectId", ""),
                project_number=item.get("name", "").rsplit("/", 1)[-1],
                display_name=item.get("displayName", ""),
            )
            for item in body.get("projects", [])
        ]
        token = body.get("nextPageToken", "")
        if not token:
            return sorted(found, key=lambda project: project.label.lower())


def projects_from_gcloud() -> list[Project]:
    """Active projects, via the Cloud SDK; works when the REST API is not enabled."""
    raw = json.loads(
        run_gcloud(
            "projects",
            "list",
            "--format=json",
            "--filter=lifecycleState:ACTIVE",
        )
        or "[]"
    )
    found = [
        Project(
            project_id=item.get("projectId", ""),
            project_number=str(item.get("projectNumber", "")),
            display_name=item.get("name", ""),
        )
        for item in raw
    ]
    return sorted(found, key=lambda project: project.label.lower())


def list_projects(credentials) -> list[Project]:
    """Every active project the account can see, with the SDK as a fallback."""
    try:
        return projects_from_api(credentials)
    except Exception:
        if not path_gcloud():
            raise
        return projects_from_gcloud()


def email_of(credentials) -> str:
    """Signed-in address: the user's own, or a service account's."""
    email_service_account = getattr(credentials, "service_account_email", "")
    if isinstance(email_service_account, str) and email_service_account:
        return email_service_account
    response = session_for(credentials).get(API_USERINFO, timeout=TIMEOUT_CALL_S)
    if response.status_code >= 400:
        return ""
    return response.json().get("email", "")


def services_state(credentials, project: str, services=APIS_REQUIRED) -> dict:
    """Whether each API in ``services`` is enabled on ``project``."""
    session = session_for(credentials)
    state = {}
    for service in services:
        body = body_of(
            session.get(
                f"{API_SERVICE_USAGE}/projects/{project}/services/{service}",
                timeout=TIMEOUT_CALL_S,
            )
        )
        state[service] = body.get("state") == "ENABLED"
    return state


def enable_services(
    credentials, project: str, services=APIS_REQUIRED, timeout: float = TIMEOUT_ENABLE_S
) -> None:
    """
    Turn ``services`` on for ``project`` and wait for the operation to finish.

    Raises
    ------
    RuntimeError
        If the API refuses (usually no billing account or no permission), or
        the operation is still running after ``timeout`` seconds.
    """
    session = session_for(credentials)
    operation = body_of(
        session.post(
            f"{API_SERVICE_USAGE}/projects/{project}/services:batchEnable",
            json={"serviceIds": list(services)},
            timeout=TIMEOUT_CALL_S,
        )
    )
    deadline = time.time() + timeout
    while not operation.get("done"):
        if time.time() > deadline:
            raise RuntimeError("enabling the APIs timed out; check the console")
        time.sleep(2.0)
        operation = body_of(
            session.get(
                f"{API_SERVICE_USAGE}/{operation['name']}", timeout=TIMEOUT_CALL_S
            )
        )
    if "error" in operation:
        raise RuntimeError(operation["error"].get("message", "could not enable APIs"))


def list_buckets(credentials, project: str) -> list:
    """Bucket names in ``project``."""
    from google.cloud import storage

    client = storage.Client(project=project, credentials=credentials)
    return sorted(bucket.name for bucket in client.list_buckets())


def create_bucket(credentials, project: str, bucket: str, location: str) -> None:
    """Create ``bucket`` in ``location``; bucket names are globally unique."""
    from google.cloud import storage

    client = storage.Client(project=project, credentials=credentials)
    client.create_bucket(bucket, location=location)


def list_service_accounts(credentials, project: str) -> list:
    """Service-account addresses in ``project``."""
    body = body_of(
        session_for(credentials).get(
            f"{API_IAM}/projects/{project}/serviceAccounts",
            params={"pageSize": 100},
            timeout=TIMEOUT_CALL_S,
        )
    )
    return sorted(account.get("email", "") for account in body.get("accounts", []))


def service_account_default(project_number: str) -> str:
    """The Compute Engine default service account for a project number."""
    if not project_number:
        return ""
    return f"{project_number}-compute@developer.gserviceaccount.com"


def quotas_region(credentials, project: str, region: str) -> dict:
    """Every quota limit in ``region``, keyed by metric name.

    GPUs are what the panel shows, but the disk metrics live here too and are
    what actually refuses most first runs.
    """
    from google.cloud import compute_v1

    client = compute_v1.RegionsClient(credentials=credentials)
    return {
        q.metric: q.limit for q in client.get(project=project, region=region).quotas
    }


@dataclass
class QuotaInfo:
    """One Cloud Quotas entry: what Google calls it, and its limit per region."""

    quota_id: str = ""
    metric: str = ""
    display_name: str = ""
    dimensions: tuple = ()
    limits: dict = field(default_factory=dict)

    @property
    def is_regional(self) -> bool:
        """Whether this quota is counted per region rather than project-wide."""
        return "region" in self.dimensions

    def limit_in(self, region: str) -> float:
        """Effective limit in ``region``, falling back to the default entry."""
        if region in self.limits:
            return self.limits[region]
        return self.limits.get("", -1.0)

    @property
    def binding(self) -> bool:
        """Whether this entry is the one that actually caps a launch."""
        return self.is_regional or not self.dimensions

    def regions_with(self, needed: float) -> list:
        """Regions whose limit already covers ``needed``."""
        return sorted(
            region
            for region, limit in self.limits.items()
            if region and limit >= needed
        )


def quota_info_of(item: dict) -> QuotaInfo:
    """One quotaInfos entry, with its per-region limits flattened."""
    limits = {}
    for entry in item.get("dimensionsInfos", []):
        value = entry.get("details", {}).get("value")
        if value is None:
            continue
        limits[entry.get("dimensions", {}).get("region", "")] = float(value)
    return QuotaInfo(
        quota_id=item.get("quotaId", ""),
        metric=item.get("metric", "").rsplit("/", 1)[-1].upper(),
        display_name=item.get("quotaDisplayName", "")
        or item.get("metricDisplayName", ""),
        dimensions=tuple(item.get("dimensions", [])),
        limits=limits,
    )


def quota_infos(credentials, project: str, service: str = SERVICE_COMPUTE) -> dict:
    """
    GPU entries from the Cloud Quotas API, keyed by quota id.

    One metric has several entries - Compute publishes an A100 quota per region
    *and* per zone - so they cannot be keyed by metric: the per-zone one reads
    as unlimited while the per-region one is the zero that blocks the launch.
    Use :func:`info_gating` to pick the one that decides.

    This is the modern, authoritative view: it carries the quota id needed to
    request an increase, and the effective limit in every region, which is what
    answers "would another region work?".
    """
    session = session_for(credentials)
    found: dict = {}
    token = ""
    while True:
        params = {"pageSize": 100}
        if token:
            params["pageToken"] = token
        body = body_of(
            session.get(
                f"{API_QUOTAS}/projects/{project}/locations/global/services/"
                f"{service}/quotaInfos",
                params=params,
                timeout=TIMEOUT_CALL_S,
            )
        )
        for item in body.get("quotaInfos", []):
            info = quota_info_of(item)
            if info.metric.endswith(SUFFIX_METRIC_GPU) or info.metric.startswith(
                "GPUS_"
            ):
                found[info.quota_id] = info
        token = body.get("nextPageToken", "")
        if not token:
            return found


def info_gating(infos: dict, metric: str) -> QuotaInfo | None:
    """
    The entry that actually limits ``metric``, or None when it is not offered.

    Prefers the per-region quota, which is the one Compute enforces; the
    per-zone twin of the same metric is usually unlimited and asking for more
    of it changes nothing.
    """
    matches = [info for info in infos.values() if info.metric == metric]
    binding = [info for info in matches if info.binding]
    return next(iter(binding or matches), None)


def request_quota(
    credentials,
    project: str,
    info: QuotaInfo,
    value: float,
    region: str = "",
    email: str = "",
    justification: str = JUSTIFICATION_QUOTA,
) -> str:
    """
    Ask Google to raise one quota, and report back in one line.

    Requests under Google's auto-approval threshold are granted in minutes;
    larger ones go to a human. One preference exists per quota and region, so
    asking twice reports the request already in flight rather than failing.
    """
    body = {
        "service": SERVICE_COMPUTE,
        "quotaId": info.quota_id,
        "quotaConfig": {"preferredValue": str(int(value))},
    }
    if info.is_regional and region:
        body["dimensions"] = {"region": region}
    if email:
        body["contactEmail"] = email
    if justification:
        body["justification"] = justification
    response = session_for(credentials).post(
        f"{API_QUOTAS}/projects/{project}/locations/global/quotaPreferences",
        json=body,
        timeout=TIMEOUT_CALL_S,
    )
    if response.status_code == 409:
        return f"{info.metric}: already requested, still with Google"
    answer = body_of(response)
    detail = answer.get("quotaConfig", {}).get("stateDetail", "")
    asked = f"{info.metric} = {int(value)}"
    return f"{asked}: requested{f' - {detail}' if detail else ''}"


def quotas_project(credentials, project: str) -> dict:
    """Project-wide compute quotas, keyed by metric; carries GPUS_ALL_REGIONS."""
    from google.cloud import compute_v1

    client = compute_v1.ProjectsClient(credentials=credentials)
    return {q.metric: q.limit for q in client.get(project=project).quotas}


def zones_by_accelerator(credentials, project: str) -> dict:
    """Zones offering each accelerator type, keyed by accelerator id."""
    from google.cloud import compute_v1

    client = compute_v1.AcceleratorTypesClient(credentials=credentials)
    found: dict = {}
    for zone, scoped in client.aggregated_list(project=project):
        name_zone = zone.rsplit("/", 1)[-1]
        for item in scoped.accelerator_types or []:
            found.setdefault(item.name, []).append(name_zone)
    return {name: sorted(zones) for name, zones in found.items()}


def bucket_suggested(project_id: str, suffix: str = "imgui-cloud") -> str:
    """A legal, unlikely-to-be-taken staging bucket name for ``project_id``."""
    if not project_id:
        return ""
    return f"{project_id}-{suffix}"[:63].strip("-")


def project_id_in(text: str) -> str:
    """Project id inside a pasted console URL, empty when there is none."""
    match = PATTERN_PROJECT.search(text or "")
    return match.group(1) if match else ""
