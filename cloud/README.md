# imgui_cloud

Run a pipeline on a Google Cloud A100 that does not exist before you press the
button and is gone when the results land.

One run is: upload a folder to a bucket → create an A100 VM with a scratch disk
attached → the box pulls the data, installs the pipeline, runs it → results go
back to the bucket → they download to your machine → the box deletes itself.
No SSH session, no VM to remember, no `gcloud` incantations.

It is usable three ways, all on the same code:

```bash
imgui-cloud login                 # project, zone, bucket, credentials
imgui-cloud init /data/raw        # writes cloud.toml
imgui-cloud run cloud.toml        # upload, provision, run, retrieve, tear down
```

```python
from imgui_cloud import CloudConfig, CloudRun

config = CloudConfig()
config.io.input = "/data/raw"
config.io.output = "/data/results"
config.job.pipeline = "masknmf"
config.job.params = {"planes": [1, 2, 3]}

run = CloudRun(config)
run.start()
run.wait()
```

```python
from imgui_cloud.gui import CloudPanel  # embed in any imgui app

panel = CloudPanel(dir_input="/data/raw")  # once
panel.draw()  # every frame
```

### The window

Nothing here needs the terminal. `imgui-cloud gui` (or `imgui-cloud login --gui`)
opens the panel on a **setup checklist** that walks the whole first run, one
step at a time, each with its console page one button away:

1. **Install the Google Cloud SDK** — shown only when `gcloud` is missing, with
   the download page and the one-liner for this platform.
2. **Sign in with Google** — one button opens the browser. No API key is
   involved: an API key cannot authorise Compute Engine or Cloud Storage on
   your behalf. A service-account key or a desktop OAuth client works instead,
   both with a Create link and a Browse field on that step.
3. **Pick your project** — a dropdown of every project the account can see.
   Picking one fills in the id, the number, the display name, the buckets, the
   service accounts, the zones that really have A100s and the region's A100
   quota. Paste a console link instead and the id is read out of the URL.
4. **Turn on Compute Engine and Cloud Storage** — one button, through the
   Service Usage API; the step waits for the operation to finish.
5. **Choose the staging bucket** — an existing one, or create the suggested
   `<project-id>-imgui-cloud` in the region your zone sits in.
6. **Request A100 quota** — read back from the region, so a zero shows up here
   instead of at launch.
7. **Check the connection** — credentials, bucket and compute, each reported.

Everything it fills in stays editable by hand under **All settings**. The
profile it writes is the same one the CLI reads, in either direction.

Once signed in the tabs are: **Run**, **Data** (input/output folders, pipeline,
JSON parameters), **Machine**, **Runs** (history + tear down everything),
**Account** (back to sign-in).

**Run** is the whole thing on one screen: the data folder, the pipeline, the
GPU, where results land, what it costs per hour, and a Launch button. Anything
missing is listed under it with a button that jumps to the tab that fixes it.
The tabs open as soon as you are signed in - a missing bucket is a line to fix,
not a locked door.

**Machine** is a GPU picker rather than a machine-type string. Every box on
offer is a row - A100 40GB x1/2/4, A100 80GB x1/2, H100 80GB x8, L4 x1/2, T4,
V100 - with its vCPU/RAM, an hourly estimate that follows the spot switch, and
the live quota that actually gates it. That last part has a trap in it:

> Preemptible quota is **opt-in**. Most projects have none, and Compute Engine
> then bills spot VMs against the ordinary quota. A zero next to
> `PREEMPTIBLE_NVIDIA_A100_GPUS` therefore says nothing about whether the box
> can start, so it is only believed when it is above zero.

The panel prints both numbers and the project-wide `GPUS_ALL_REGIONS` cap, tells
you which one it used, and offers the cheapest box that does have quota.

A **new project has zero GPU quota everywhere** - that is Google's default, not
a fault, and `GPUS_ALL_REGIONS = 0` is the giveaway: it is counted across every
region at once, so switching region cannot help until it is raised. When the
Cloud Quotas API is on, the panel reads your real limits region by region, says
exactly which increases this run needs, and asks for them from here:

```
Ask Google for
    GPUS_ALL_REGIONS  = 1    project-wide
    NVIDIA_A100_GPUS  = 1    in us-central1
[ Ask Google for both ]   [ Request quota ]   [ Copy metric ]
```

Increases under Google's auto-approval threshold land in minutes. Regions that
already allow the GPU are listed as buttons that move your zone there. Quota
never disables Launch either: if the numbers look wrong, launch anyway and
Google's refusal names the quota and the amount it wants.

### APIs

The API step lists every API the panel uses, what turning it on allows, and an
Enable button per row - and any failure that says *"has not been used in project
... or it is disabled"* grows the same button underneath it, wherever it appears.

| API | what it allows | |
| --- | --- | --- |
| `compute.googleapis.com` | creates, watches and deletes the worker | required |
| `storage.googleapis.com` | moves data up and results back | required |
| `iam.googleapis.com` | lists service accounts a worker can run as | optional |
| `cloudquotas.googleapis.com` | reads real GPU limits and requests increases in-app | optional |

`imgui-cloud gui --pick` puts the
[imgui_data_loader](https://github.com/FlynnOConnell/imgui_data_loader) launcher
in front of it for the "pick data, then send it up" flow; `imgui-cloud gui
/data/raw` starts from a known folder. In `mbo_utilities` the same panel is the
**Cloud** tab of the viewer (enable it under *Widgets → Cloud (GPU)*).

## Install

```bash
uv pip install imgui_cloud    # or: uv pip install ./cloud from this repo
imgui-cloud login --browser   # or press "Sign in with Google" in the panel
```

## What a run costs

An `a2-highgpu-1g` (1× A100 40GB) is roughly **$3.67/hour** on demand and
**~$1.47/hour** on spot, which is the default. The panel shows a live estimate
before you launch and the running total after.

Three independent mechanisms stop the meter, because a closed laptop must never
cost money:

1. the worker's startup script deletes the instance when the job ends,
2. the orchestrator deletes it on success, failure **and** cancellation,
3. the instance carries `max_run_duration` with `instance_termination_action =
   DELETE`, so it removes itself at `machine.max_runtime_min` regardless.

The scratch disk is created with `auto_delete`, so it dies with the VM unless
`keep_data_disk = true`. `imgui-cloud down --all` is the manual sweep; it only
ever touches instances labelled `managed-by=imgui-cloud`.

## Sign-in

`imgui-cloud login` (or the Account tab) stores a profile in
`~/.mbo/settings/cloud_profiles.json`, mode 0600:

| field | meaning |
| --- | --- |
| `project_id` | project that owns the instances and the bucket |
| `project_number` | its numeric id, filled in when you pick the project |
| `project_name` | its display name, so ids never have to be recognised |
| `zone` | where the A100 is created (`imgui-cloud zones` lists valid ones) |
| `bucket` | staging bucket, no `gs://` |
| `filepath_service_account_key` | path to a JSON key; blank falls back to gcloud ADC |
| `user_email` | recorded as an instance label and in the run history |
| `service_account_email` | attached to the VM so it can reach the bucket |

**The key file's contents are never copied into the profile** — only its path.
Set several profiles (`--profile lab2`) and switch with `imgui-cloud profiles`.

The service account needs `roles/compute.instanceAdmin.v1`,
`roles/iam.serviceAccountUser`, and `roles/storage.objectAdmin` on the bucket.

`imgui-cloud login --browser` signs in and then lists your projects to pick from
by number; `imgui-cloud projects` prints them with their ids and numbers.

Where each thing lives in the console, if you would rather click (`ID` is your
project id — the checklist links carry it for you):

| what | where |
| --- | --- |
| your projects | `console.cloud.google.com/projectselector2/home/dashboard` |
| a new project | `console.cloud.google.com/projectcreate` |
| billing, without which the APIs stay off | `console.cloud.google.com/billing/linkedaccount?project=ID` |
| the two APIs | `console.cloud.google.com/apis/library/compute.googleapis.com?project=ID` |
| buckets | `console.cloud.google.com/storage/browser?project=ID` |
| A100 quota (filter for `NVIDIA_A100_GPUS`) | `console.cloud.google.com/iam-admin/quotas?project=ID` |
| a service-account key | `console.cloud.google.com/iam-admin/serviceaccounts?project=ID` → the account → Keys → Add key → Create new key → JSON |
| a desktop OAuth client | `console.cloud.google.com/auth/clients?project=ID` → Create client → Desktop app → Download JSON |

## What runs on the box

`[job] pipeline` names an entry in `imgui_cloud.pipelines`:

| pipeline | entry point |
| --- | --- |
| `masknmf` | `mbo_utilities.masknmf:run_volume` |
| `suite2p` | `lbm_suite2p_python:run_volume` |

The worker installs the spec's requirements plus anything in `[job] pip`, then
calls the entry point with the staged input path, the output path, and
`[job.params]`. A package adds its own pipeline without touching this one:

```toml
[project.entry-points."imgui_cloud.pipelines"]
myproc = "my_package.cloud:SPEC"    # a PipelineSpec
```

Only `worker_main.py` (standard library, ~90 lines) is copied to the VM;
imgui_cloud itself is never installed there, so a GUI dependency can never break
a run.

## Bucket layout

```
gs://<bucket>/<prefix>/<run_id>/
    input/            what was uploaded
    output/           what the worker produced
    logs/worker.log   streamed every 15s while the job runs
    status/state.txt  STARTED | RUNNING | DONE | FAILED
    job.json          the resolved pipeline call
    worker_main.py    the script the box executes
```

Everything the client knows comes from those objects, so a run survives the
laptop sleeping, and `imgui-cloud logs <run_id> -f` works from any machine.

## Commands

| command | does |
| --- | --- |
| `login` / `profiles` | sign in, switch or delete profiles |
| `init` | write a commented `cloud.toml` |
| `run --dry-run` | print the job and the startup script; create nothing |
| `ls` / `status` / `logs -f` | run history, one run's state, its worker log |
| `down --all` | delete every managed worker in the zone |
| `pipelines` / `zones` | what can run, and where an A100 is available |
| `gui` | the standalone window (`--pick` to choose data first) |

## Debugging a failed run

```bash
imgui-cloud logs <run_id>              # worker log from the bucket
imgui-cloud logs <run_id> --serial     # serial console, for boot failures
imgui-cloud run cloud.toml --keep-instance   # leave the box up (then: down --all)
```

## Tests

```bash
uv pip install -e ".[dev]"
pytest
```

No test touches Google Cloud: the transfer, orchestration and lifecycle tests
run against in-memory fakes, and the instance-shape tests assert on the resource
that *would* be sent.
