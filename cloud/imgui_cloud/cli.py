"""``imgui-cloud`` command group: sign in, describe a run, launch it, clean up."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click

from imgui_cloud import account
from imgui_cloud import config as config_module
from imgui_cloud import credentials as credentials_module
from imgui_cloud import (
    gcs,
    history,
    instance as instance_module,
    pipelines,
    run as run_module,
)


def _mark(ok: bool) -> str:
    """Green "ok" / red "no" for one check in the sign-in report."""
    return click.style("ok", fg="green") if ok else click.style("no", fg="red")


def _projects_visible(profile) -> list:
    """Projects the stored credentials can see; empty when not signed in yet."""
    if not credentials_module.credentials_available(profile):
        return []
    try:
        credentials, _ = credentials_module.credentials_for(profile)
        return account.list_projects(credentials)
    except Exception as e:
        click.echo(click.style(f"could not list projects: {e}", fg="yellow"))
        return []


def _prompt_project(profile) -> None:
    """Pick a project by number from the signed-in account, or type an id."""
    projects = _projects_visible(profile)
    for i, project in enumerate(projects, 1):
        click.echo(f"  {i:>2}. {project.label}")
    answer = click.prompt(
        "project (number or id)" if projects else "project id",
        default=profile.project_id,
        show_default=bool(profile.project_id),
    ).strip()
    chosen = next(
        (p for p in projects if p.project_id == answer),
        account.Project(project_id=answer),
    )
    if answer.isdigit() and 1 <= int(answer) <= len(projects):
        chosen = projects[int(answer) - 1]
    profile.project_id = chosen.project_id
    profile.project_number = chosen.project_number
    profile.project_name = chosen.display_name


def _echo_status(status: credentials_module.ProfileStatus) -> None:
    """Print a sign-in probe result, one line per check."""
    click.echo(f"  credentials  {_mark(status.credentials_ok)}  {status.identity}")
    click.echo(f"  bucket       {_mark(status.bucket_ok)}")
    click.echo(f"  compute      {_mark(status.compute_ok)}")
    for problem in status.problems:
        click.echo(click.style(f"  ! {problem}", fg="yellow"))


@click.group("imgui-cloud")
@click.version_option(package_name="imgui_cloud")
def main():
    """Run pipelines on ephemeral Google Cloud A100 workers."""


@main.command(
    "login", short_help="Enter or verify your Google Cloud connection settings."
)
@click.option(
    "-p", "--profile", "profile_name", default=None, help="Profile name to edit."
)
@click.option("--project", default=None, help="Google Cloud project id.")
@click.option("--zone", default=None, help="Zone the worker is created in.")
@click.option("--bucket", default=None, help="Staging bucket (no gs:// prefix).")
@click.option(
    "--key",
    "filepath_key",
    default=None,
    help="Service-account JSON key path; omit to use gcloud ADC.",
)
@click.option("--email", default=None, help="Your email, recorded on runs.")
@click.option(
    "--service-account", default=None, help="Service account attached to the VM."
)
@click.option(
    "--browser",
    "use_browser",
    is_flag=True,
    help="Sign in with Google in your browser before anything else.",
)
@click.option(
    "--gui", "use_gui", is_flag=True, help="Fill this in in the window instead."
)
@click.option("--show", is_flag=True, help="Print the stored profile and exit.")
@click.option(
    "--check/--no-check", default=True, help="Probe the connection after saving."
)
def login(
    profile_name,
    project,
    zone,
    bucket,
    filepath_key,
    email,
    service_account,
    use_browser,
    use_gui,
    show,
    check,
):
    """
    Store the project, zone, bucket and credentials a run needs.

    ``--browser`` signs in with Google first and then lists your projects to
    pick from by number, which is the shortest path from nothing to a working
    profile. With no options this prompts for each field, showing the current
    value as the default; ``--gui`` opens the same form as a window instead.
    The service-account key is stored as a *path*: the key itself never enters
    the profile file.
    """
    if use_browser:
        if not account.path_gcloud():
            raise click.ClickException(
                f"the Google Cloud SDK is not installed: "
                f"{account.URL_INSTALL_GCLOUD}  ({account.command_install_gcloud()})"
            )
        click.echo("opening your browser; finish signing in there...")
        account.login_gcloud()

    if use_gui:
        from imgui_cloud.gui.app import run_cloud_app

        run_cloud_app()
        _echo_status(
            credentials_module.check_profile(
                credentials_module.load_profile(profile_name)
            )
        )
        return

    profile = credentials_module.load_profile(profile_name)
    if show:
        click.echo(json.dumps(profile.__dict__, indent=2))
        _echo_status(credentials_module.check_profile(profile))
        return

    given = dict(
        project_id=project,
        zone=zone,
        bucket=bucket,
        filepath_service_account_key=filepath_key,
        user_email=email,
        service_account_email=service_account,
    )
    if all(v is None for v in given.values()):
        _prompt_project(profile)
        profile.zone = click.prompt("zone", default=profile.zone)
        profile.bucket = click.prompt(
            "bucket (no gs://)",
            default=profile.bucket or account.bucket_suggested(profile.project_id),
        )
        profile.filepath_service_account_key = click.prompt(
            "service-account key path (blank = gcloud ADC)",
            default=profile.filepath_service_account_key,
            show_default=bool(profile.filepath_service_account_key),
        )
        profile.user_email = click.prompt("your email", default=profile.user_email)
        profile.service_account_email = click.prompt(
            "VM service account (blank = project default)",
            default=profile.service_account_email
            or account.service_account_default(profile.project_number),
            show_default=True,
        )
    else:
        for field_name, value in given.items():
            if value is not None:
                setattr(profile, field_name, value)
    if profile_name:
        profile.name = profile_name

    credentials_module.save_profile(profile)
    click.echo(
        f"saved profile {profile.name!r} to {credentials_module.filepath_profiles()}"
    )
    if check:
        _echo_status(credentials_module.check_profile(profile))


@main.command("profiles", short_help="List stored profiles, or switch the active one.")
@click.argument("name", required=False)
@click.option(
    "--delete", "do_delete", is_flag=True, help="Delete NAME instead of activating it."
)
def profiles(name, do_delete):
    """Show every stored profile; with NAME, make it active (or delete it)."""
    if name and do_delete:
        credentials_module.delete_profile(name)
        click.echo(f"deleted profile {name!r}")
        return
    if name:
        credentials_module.set_active_profile(name)
        click.echo(f"active profile is now {name!r}")
        return
    active = credentials_module.active_profile_name()
    for profile_name, profile in credentials_module.load_profiles().items():
        marker = "*" if profile_name == active else " "
        click.echo(
            f"{marker} {profile_name:<16} {profile.project_id}  {profile.zone}  gs://{profile.bucket}"
        )


@main.command("init", short_help="Write a commented cloud.toml next to your data.")
@click.argument("data_path", required=False, type=click.Path())
@click.option(
    "-o", "--config", "filepath_config", default="cloud.toml", type=click.Path()
)
@click.option("-O", "--output", "dir_output", default=None, type=click.Path())
@click.option("--pipeline", default="masknmf", help="Pipeline the run should execute.")
@click.option(
    "--name", default=None, help="Run name; defaults to the input folder name."
)
@click.option("--overwrite/--no-overwrite", default=False)
def init(data_path, filepath_config, dir_output, pipeline, name, overwrite):
    """Create a starter run config for DATA_PATH."""
    dir_input = Path(data_path or ".").resolve()
    dir_output = dir_output or str(dir_input.parent / "results")
    try:
        written = config_module.write_template(
            filepath_config,
            dir_input=str(dir_input),
            dir_output=str(dir_output),
            name=name or dir_input.name,
            pipeline=pipeline,
            overwrite=overwrite,
        )
    except FileExistsError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"wrote {written}")
    click.echo(f"edit it, then: imgui-cloud run {written}")


@main.command(
    "run", short_help="Upload, provision an A100, run the pipeline, bring it back."
)
@click.argument("filepath_config", type=click.Path(exists=True), default="cloud.toml")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the plan and the startup script; create nothing.",
)
@click.option("--input", "input_", default=None, help="Override [io] input.")
@click.option("--output", default=None, help="Override [io] output.")
@click.option("--name", default=None, help="Override [io] name.")
@click.option("--pipeline", default=None, help="Override [job] pipeline.")
@click.option("--machine-type", default=None, help="Override [machine] machine_type.")
@click.option("--spot/--no-spot", default=None, help="Override [machine] spot.")
@click.option(
    "--keep-instance", is_flag=True, default=None, help="Leave the VM up afterwards."
)
@click.option(
    "--max-runtime-min", type=int, default=None, help="Override the hard lifetime cap."
)
@click.option(
    "-p", "--profile", "profile_name", default=None, help="Profile to run under."
)
def run_command(
    filepath_config,
    dry_run,
    input_,
    output,
    name,
    pipeline,
    machine_type,
    spot,
    keep_instance,
    max_runtime_min,
    profile_name,
):
    """Execute the run described by FILEPATH_CONFIG, printing progress as it goes."""
    config = config_module.from_toml(filepath_config)
    config_module.merge_overrides(
        config,
        {
            "io.input": input_,
            "io.output": output,
            "io.name": name,
            "job.pipeline": pipeline,
            "machine.machine_type": machine_type,
            "machine.spot": spot,
            "machine.keep_instance": True if keep_instance else None,
            "machine.max_runtime_min": max_runtime_min,
        },
    )
    profile = credentials_module.load_profile(profile_name)
    problems = config.validate()
    if problems:
        raise click.ClickException("; ".join(problems))
    if not profile.bucket or not profile.project_id:
        message = (
            f"profile {profile.name!r} has no bucket/project; run: imgui-cloud login"
        )
        if not dry_run:
            raise click.ClickException(message)
        click.echo(click.style(f"! {message}", fg="yellow"))

    cloud_run = run_module.CloudRun(config, profile=profile)
    click.echo(f"run id      {cloud_run.run_id}")
    click.echo(f"input       {config.io.input}")
    click.echo(f"output      {config_module.resolve_output_dir(config)}")
    click.echo(f"staging     {cloud_run.uri_run}")
    click.echo(
        f"machine     {config.machine.machine_type} "
        f"({'spot' if config.machine.spot else 'on-demand'}) in {profile.zone}"
    )
    click.echo(f"pipeline    {config.job.pipeline}")
    click.echo(
        f"est. cost   ~${config.machine.cost_per_hour_estimate():.2f}/hour, "
        f"hard cap {config.machine.max_runtime_min} min"
    )

    if dry_run:
        from imgui_cloud import startup

        spec = pipelines.get(config.job.pipeline)
        click.echo("\n--- job.json ---")
        click.echo(
            json.dumps(
                pipelines.build_job(
                    spec, "/mnt/data/input", "/mnt/data/output", config.job.params
                ),
                indent=2,
            )
        )
        click.echo("\n--- startup-script ---")
        click.echo(
            startup.build_startup_script(
                uri_run=cloud_run.uri_run,
                run_id=cloud_run.run_id,
                requirements=pipelines.requirements(spec, config.job.pip),
                env=config.job.env,
                self_delete=not config.machine.keep_instance,
            )
        )
        return

    phase_last = ""
    cloud_run.start()
    try:
        while cloud_run.is_alive():
            state = cloud_run.state
            line = f"[{state.phase}] {state.message}"
            if line != phase_last:
                click.echo(line)
                phase_last = line
            time.sleep(1.0)
    except KeyboardInterrupt:
        click.echo("\ncancelling; the instance will be deleted...")
        cloud_run.cancel()
    state = cloud_run.wait()
    click.echo(f"[{state.phase}] {state.message}")
    if state.phase != history.PHASE_DONE:
        if state.log_tail:
            click.echo("\n--- worker log (tail) ---")
            click.echo(state.log_tail)
        sys.exit(1)


@main.command("ls", short_help="List recent runs.")
@click.option("-n", "--limit", default=20, show_default=True)
@click.option("--live", is_flag=True, help="Only runs that have not finished.")
def ls(limit, live):
    """Show run history, newest first."""
    records = history.load_live() if live else history.load_all(limit=limit)
    if not records:
        click.echo("no runs recorded")
        return
    for record in records:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(record.time_created))
        click.echo(
            f"{record.run_id:<34} {stamp}  {record.phase:<12} "
            f"{record.pipeline:<10} {record.machine_type:<16} ~${record.cost_estimate():.2f}"
        )


@main.command("status", short_help="Show one run's state and the worker's status.")
@click.argument("run_id")
def status(run_id):
    """Print everything known about RUN_ID, including the live instance state."""
    record = history.load(run_id)
    if record is None:
        raise click.ClickException(f"no such run: {run_id}")
    click.echo(json.dumps(record.__dict__, indent=2, default=str))
    profile = credentials_module.load_profile(record.profile_name)
    creds, project = credentials_module.credentials_for(profile)
    live = instance_module.worker_status(
        creds, record.project_id or project, record.zone, record.instance_name
    )
    click.echo(f"instance    {record.instance_name}: {live}")


@main.command("logs", short_help="Print (or follow) a run's worker log.")
@click.argument("run_id")
@click.option("-f", "--follow", is_flag=True, help="Keep polling until the run ends.")
@click.option(
    "--serial", is_flag=True, help="Read the VM serial console instead of GCS."
)
def logs(run_id, follow, serial):
    """Show the worker log that the box streams into the bucket."""
    record = history.load(run_id)
    if record is None:
        raise click.ClickException(f"no such run: {run_id}")
    profile = credentials_module.load_profile(record.profile_name)
    creds, project = credentials_module.credentials_for(profile)
    project = record.project_id or project

    if serial:
        click.echo(
            instance_module.serial_output(
                creds, project, record.zone, record.instance_name
            )
        )
        return

    client = gcs.client_for(profile, credentials=creds, project=project)
    name_log = f"{record.prefix}/{record.run_id}/logs/worker.log"
    seen = 0
    while True:
        text = gcs.read_text(client, record.bucket, name_log) or ""
        if len(text) > seen:
            click.echo(text[seen:], nl=False)
            seen = len(text)
        if not follow:
            break
        state = gcs.read_text(
            client, record.bucket, f"{record.prefix}/{record.run_id}/status/state.txt"
        )
        if state in (gcs.STATE_DONE, gcs.STATE_FAILED):
            break
        time.sleep(10)


@main.command("down", short_help="Delete workers this tool created (safety valve).")
@click.argument("run_id", required=False)
@click.option(
    "--all", "do_all", is_flag=True, help="Delete every managed worker in the zone."
)
@click.option("-p", "--profile", "profile_name", default=None)
@click.option("--yes", is_flag=True, help="Do not ask.")
def down(run_id, do_all, profile_name, yes):
    """Tear down a specific RUN_ID's worker, or every worker with --all."""
    profile = credentials_module.load_profile(profile_name)
    creds, project = credentials_module.credentials_for(profile)
    project = profile.project_id or project

    if do_all:
        workers = instance_module.list_workers(creds, project, profile.zone)
        if not workers:
            click.echo(f"no managed workers in {profile.zone}")
            return
        click.echo("\n".join(f"  {w.name}  {w.status}" for w in workers))
        if not yes and not click.confirm(f"delete {len(workers)} instance(s)?"):
            return
        for name in run_module.teardown_orphans(profile):
            click.echo(f"deleted {name}")
        return

    if not run_id:
        raise click.ClickException("give a RUN_ID or pass --all")
    record = history.load(run_id)
    if record is None:
        raise click.ClickException(f"no such run: {run_id}")
    if not yes and not click.confirm(
        f"delete {record.instance_name} in {record.zone}?"
    ):
        return
    deleted = instance_module.delete_worker(
        creds, project, record.zone, record.instance_name, wait=True
    )
    record.phase = history.PHASE_CANCELLED if not record.is_terminal else record.phase
    record.message = "torn down manually"
    history.save(record)
    click.echo("deleted" if deleted else "instance was already gone")


@main.command("pipelines", short_help="List pipelines a worker can run.")
def list_pipelines():
    """Show every registered pipeline and what it installs."""
    for name in pipelines.available():
        spec = pipelines.get(name)
        click.echo(f"{name:<12} {spec.description}")
        click.echo(f"{'':<12} entry: {spec.entry}")
        click.echo(f"{'':<12} pip:   {', '.join(spec.pip) or '-'}")


@main.command("projects", short_help="List the projects this account can see.")
@click.option("-p", "--profile", "profile_name", default=None)
def projects(profile_name):
    """Print every active project, with its id and number."""
    profile = credentials_module.load_profile(profile_name)
    visible = _projects_visible(profile)
    if not visible:
        click.echo("no projects visible; sign in first: imgui-cloud login --browser")
        return
    for project in visible:
        marker = "*" if project.project_id == profile.project_id else " "
        click.echo(
            f"{marker} {project.project_id:<32} {project.project_number:<16} "
            f"{project.display_name}"
        )


@main.command("zones", short_help="List zones in the project that have A100s.")
@click.option("-p", "--profile", "profile_name", default=None)
@click.option("--accelerator", default="nvidia-tesla-a100", show_default=True)
def zones(profile_name, accelerator):
    """Ask Compute Engine which zones advertise the accelerator."""
    profile = credentials_module.load_profile(profile_name)
    creds, project = credentials_module.credentials_for(profile)
    found = instance_module.zones_with_a100(
        creds, profile.project_id or project, accelerator
    )
    click.echo("\n".join(found) or f"no zones advertise {accelerator}")


@main.command("gui", short_help="Open the standalone sign-in + run window.")
@click.argument("data_path", required=False, type=click.Path())
@click.option(
    "--pick",
    is_flag=True,
    help="Open the imgui_data_loader launcher first, then the panel.",
)
def gui(data_path, pick):
    """
    Launch the imgui panel as its own application.

    Everything the terminal commands do lives here too: signing in, choosing
    data, sizing the box, launching, watching the worker log and tearing down.
    The sign-in form is what the window opens on until the profile checks out.
    """
    from imgui_cloud.gui.app import run_cloud_app

    run_cloud_app(dir_input=data_path or "", pick_first=pick)


if __name__ == "__main__":
    main()
