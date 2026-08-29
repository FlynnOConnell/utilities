"""Sign-in state: the Google Cloud identity and destinations a run needs.

Profiles live in ``~/.mbo/settings/cloud_profiles.json`` with owner-only
permissions. Key *contents* are never copied there - only the path - so the
secret stays wherever the user put it.
"""

from __future__ import annotations


import json
import os
import stat
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SCOPES_CLOUD = ["https://www.googleapis.com/auth/cloud-platform"]
SCOPES_LOGIN = SCOPES_CLOUD + [
    "https://www.googleapis.com/auth/userinfo.email",
    "openid",
]


def dir_settings() -> Path:
    """Directory holding the profile store, honoring ``MBO_DIR`` / ``MBO_USER``."""
    dir_explicit = os.environ.get("MBO_DIR")
    if dir_explicit:
        base = Path(dir_explicit).expanduser()
    elif os.environ.get("MBO_USER"):
        base = Path(os.environ["MBO_USER"]).expanduser() / ".mbo"
    else:
        base = Path.home() / ".mbo"
    dir_out = base / "settings"
    dir_out.mkdir(parents=True, exist_ok=True)
    return dir_out


def filepath_profiles() -> str:
    """Path of the JSON profile store."""
    return str(dir_settings() / "cloud_profiles.json")


def filepath_token_user() -> str:
    """Path of the token written by the in-app Google sign-in."""
    return str(dir_settings() / "google_token.json")


def save_token_user(credentials) -> None:
    """Store an OAuth user token, owner-readable only."""
    path = Path(filepath_token_user())
    path.write_text(credentials.to_json())
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def forget_token_user() -> None:
    """Delete the in-app token; a gcloud sign-in on this machine is left alone."""
    Path(filepath_token_user()).unlink(missing_ok=True)


@dataclass
class CloudProfile:
    """
    One named set of Google Cloud connection settings.

    Parameters
    ----------
    name : str
        Profile key in the store.
    project_id : str
        Google Cloud project that owns the instances and the bucket.
    project_number : str
        Numeric id of that project, filled in when a project is picked.
    project_name : str
        Display name of that project, shown so ids never have to be recognised.
    zone : str
        Zone the worker is created in, e.g. ``us-central1-a``. A100s exist only
        in some zones; ``imgui-cloud zones`` lists the ones that have them.
    bucket : str
        GCS bucket used to stage inputs, outputs and logs. No ``gs://`` prefix.
    prefix : str
        Object-name prefix inside the bucket under which runs are written.
    filepath_service_account_key : str
        Path to a service-account JSON key. Empty means Application Default
        Credentials.
    user_email : str
        Who the run belongs to; recorded as an instance label and in history.
    service_account_email : str
        Service account attached to the worker VM so it can reach the bucket.
        Empty means the project's default compute service account.
    """

    name: str = "default"
    project_id: str = ""
    project_number: str = ""
    project_name: str = ""
    zone: str = "us-central1-a"
    bucket: str = ""
    prefix: str = "imgui-cloud"
    filepath_service_account_key: str = ""
    user_email: str = ""
    service_account_email: str = ""

    def uri_run(self, run_id: str) -> str:
        """``gs://`` URI of one run's staging directory."""
        return f"gs://{self.bucket}/{self.prefix}/{run_id}"

    @property
    def region(self) -> str:
        """Region the zone sits in (``us-central1-a`` -> ``us-central1``)."""
        return self.zone.rsplit("-", 1)[0]


@dataclass
class ProfileStatus:
    """Result of :func:`check_profile` - what is filled in, and what works."""

    ok: bool = False
    credentials_ok: bool = False
    bucket_ok: bool = False
    compute_ok: bool = False
    identity: str = ""
    problems: list = field(default_factory=list)

    def summary(self) -> str:
        """One-line human-readable status."""
        if self.ok:
            return f"signed in as {self.identity}"
        return "; ".join(self.problems) or "not signed in"


def load_profiles() -> dict:
    """All stored profiles, keyed by name. A missing store yields ``{}``."""
    path = Path(filepath_profiles())
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    names_valid = {f.name for f in fields(CloudProfile)}
    return {
        name: CloudProfile(**{k: v for k, v in body.items() if k in names_valid})
        for name, body in raw.get("profiles", {}).items()
    }


def load_profile(name: str | None = None) -> CloudProfile:
    """
    Load one profile by name, or the active one when ``name`` is None.

    Returns an empty profile (all defaults) when the store has nothing yet, so
    the sign-in panel always has something to render.
    """
    profiles = load_profiles()
    if name is None:
        name = active_profile_name()
    if name in profiles:
        return profiles[name]
    return CloudProfile(name=name or "default")


def save_profile(profile: CloudProfile, make_active: bool = True) -> None:
    """Write ``profile`` into the store, with owner-only file permissions."""
    if not profile.name:
        raise ValueError("profile.name must not be empty")
    path = Path(filepath_profiles())
    raw = json.loads(path.read_text()) if path.exists() else {"profiles": {}}
    raw.setdefault("profiles", {})[profile.name] = asdict(profile)
    if make_active:
        raw["active"] = profile.name
    path.write_text(json.dumps(raw, indent=2))
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def delete_profile(name: str) -> None:
    """Remove ``name`` from the store; unsets it as active if it was."""
    path = Path(filepath_profiles())
    if not path.exists():
        return
    raw = json.loads(path.read_text())
    raw.get("profiles", {}).pop(name, None)
    if raw.get("active") == name:
        raw["active"] = next(iter(raw.get("profiles", {})), "")
    path.write_text(json.dumps(raw, indent=2))
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def active_profile_name() -> str:
    """Name of the profile last saved as active, or ``"default"``."""
    path = Path(filepath_profiles())
    if not path.exists():
        return "default"
    return json.loads(path.read_text()).get("active", "default") or "default"


def set_active_profile(name: str) -> None:
    """Mark ``name`` as the profile the CLI and GUI open with."""
    path = Path(filepath_profiles())
    raw = json.loads(path.read_text()) if path.exists() else {"profiles": {}}
    raw["active"] = name
    path.write_text(json.dumps(raw, indent=2))
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def filepath_adc() -> str:
    """Where gcloud writes application default credentials on this platform."""
    explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit:
        return explicit
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path.home() / ".config"
    return str(base / "gcloud" / "application_default_credentials.json")


def credentials_available(profile: CloudProfile) -> bool:
    """Whether this machine can sign in without another trip to the browser."""
    if profile.filepath_service_account_key:
        return Path(profile.filepath_service_account_key).expanduser().exists()
    return Path(filepath_token_user()).exists() or Path(filepath_adc()).exists()


def credentials_for(profile: CloudProfile):
    """
    Google credentials for ``profile``: the key file, the in-app sign-in, else ADC.

    Returns
    -------
    tuple
        ``(credentials, project_id)``. The project falls back to whatever the
        credentials themselves carry when the profile leaves it blank.

    Raises
    ------
    FileNotFoundError
        If a key path is set but does not exist - a typo there otherwise turns
        into a silent ADC fallback that talks to the wrong project.
    """
    if profile.filepath_service_account_key:
        path_key = Path(profile.filepath_service_account_key).expanduser()
        if not path_key.exists():
            raise FileNotFoundError(f"service-account key not found: {path_key}")
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(
            str(path_key), scopes=SCOPES_CLOUD
        )
        project = profile.project_id or json.loads(path_key.read_text()).get(
            "project_id", ""
        )
        return credentials, project

    path_token = Path(filepath_token_user())
    if path_token.exists():
        from google.oauth2.credentials import Credentials

        credentials = Credentials.from_authorized_user_file(
            str(path_token), scopes=SCOPES_LOGIN
        )
        return credentials, profile.project_id

    import google.auth

    credentials, project_adc = google.auth.default(scopes=SCOPES_CLOUD)
    return credentials, (profile.project_id or project_adc or "")


def source_of(profile: CloudProfile) -> str:
    """Where :func:`credentials_for` will get its credentials from."""
    if profile.filepath_service_account_key:
        return "service-account key"
    if Path(filepath_token_user()).exists():
        return "Google sign-in"
    return "gcloud application default credentials"


def identity_of(credentials) -> str:
    """Best-effort account string for a credentials object."""
    for attr in ("service_account_email", "signer_email", "quota_project_id"):
        value = getattr(credentials, attr, None)
        if isinstance(value, str) and value:
            return value
    return "application default credentials"


def check_profile(profile: CloudProfile) -> ProfileStatus:
    """
    Sign in and probe: can we authenticate, read the bucket, list instances?

    Every failure is collected rather than raised, so the panel can show all of
    them at once. This is what the GUI "Sign in" button calls.
    """
    status = ProfileStatus()
    if not profile.project_id:
        status.problems.append("project id is empty")
    if not profile.bucket:
        status.problems.append("bucket is empty")

    try:
        credentials, project = credentials_for(profile)
    except Exception as e:
        status.problems.append(f"credentials: {e}")
        return status
    status.credentials_ok = True
    status.identity = identity_of(credentials)
    project = profile.project_id or project

    if profile.bucket:
        from google.cloud import storage

        try:
            storage.Client(project=project, credentials=credentials).get_bucket(
                profile.bucket
            )
            status.bucket_ok = True
        except Exception as e:
            status.problems.append(f"bucket gs://{profile.bucket}: {e}")

    if project:
        from google.cloud import compute_v1

        try:
            client = compute_v1.InstancesClient(credentials=credentials)
            request = compute_v1.ListInstancesRequest(
                project=project, zone=profile.zone, max_results=1
            )
            next(iter(client.list(request=request)), None)
            status.compute_ok = True
        except Exception as e:
            status.problems.append(f"compute {profile.zone}: {e}")

    status.ok = status.credentials_ok and status.bucket_ok and status.compute_ok
    return status
