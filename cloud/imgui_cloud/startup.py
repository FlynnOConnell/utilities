"""
The startup script the worker runs the moment it boots.

One bash script does the whole box-side lifecycle: wait for the GPU driver,
format and mount the attached scratch disk, pull the run's inputs from GCS,
install the pipeline, run it, push the outputs back, unmount, and delete the
instance. State (``STARTED`` / ``RUNNING`` / ``DONE`` / ``FAILED``) and the
growing log are written to GCS throughout, which is how the client follows a
run without ever holding an SSH connection.

The self-delete at the end is deliberate redundancy. Three independent things
stop the meter: this script, the orchestrator's teardown, and the instance's own
``max_run_duration``. A closed laptop is not allowed to cost money.
"""

from __future__ import annotations


import shlex

DEVICE_NAME_DATA = "imgui-cloud-data"
DIR_DATA = "/mnt/data"

_TEMPLATE = r"""#!/bin/bash
set -uo pipefail

LOG=/var/log/imgui-cloud.log
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

GS_RUN="__GS_RUN__"
DATA="__DIR_DATA__"
DEV="/dev/disk/by-id/google-__DEVICE_NAME__"
SELF_DELETE=__SELF_DELETE__

echo "[startup] run __RUN_ID__ starting at $(date -Is)"

push_log() { gcloud storage cp "$LOG" "$GS_RUN/logs/worker.log" >/dev/null 2>&1 || true; }
state() { echo -n "$1" | gcloud storage cp - "$GS_RUN/status/state.txt" >/dev/null 2>&1 || true; }

teardown() {
    echo "[startup] tearing down at $(date -Is)"
    kill "$LOGGER_PID" >/dev/null 2>&1 || true
    push_log
    cd /
    for _ in 1 2 3 4 5; do
        umount "$DATA" >/dev/null 2>&1 && break
        sleep 3
    done
    if [ "$SELF_DELETE" = "1" ]; then
        NAME=$(curl -s -H "Metadata-Flavor: Google" \
            http://metadata.google.internal/computeMetadata/v1/instance/name)
        ZONE=$(curl -s -H "Metadata-Flavor: Google" \
            http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)
        echo "[startup] deleting instance $NAME in $ZONE"
        gcloud --quiet compute instances delete "$NAME" --zone "$ZONE" || shutdown -h now
    fi
}

fail() {
    echo "[startup] FAILED: $1"
    state FAILED
    teardown
    exit 1
}

( while true; do sleep 15; push_log; done ) &
LOGGER_PID=$!

state STARTED

echo "[startup] waiting for the GPU driver"
for _ in $(seq 1 90); do
    nvidia-smi >/dev/null 2>&1 && break
    sleep 10
done
nvidia-smi || echo "[startup] warning: no GPU visible; the job may run on CPU"

echo "[startup] mounting scratch disk $DEV"
for _ in $(seq 1 60); do
    [ -e "$DEV" ] && break
    sleep 2
done
[ -e "$DEV" ] || fail "scratch disk $DEV never appeared"
if ! blkid "$DEV" >/dev/null 2>&1; then
    echo "[startup] formatting $DEV (ext4)"
    mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard "$DEV" \
        || fail "could not format $DEV"
fi
mkdir -p "$DATA"
mount -o discard,defaults "$DEV" "$DATA" || fail "could not mount $DEV on $DATA"
chmod 1777 "$DATA"
mkdir -p "$DATA/input" "$DATA/output"
df -h "$DATA"

echo "[startup] fetching the job payload"
gcloud storage cp "$GS_RUN/worker_main.py" "$DATA/worker_main.py" || fail "no worker_main.py in $GS_RUN"
gcloud storage cp "$GS_RUN/job.json" "$DATA/job.json" || fail "no job.json in $GS_RUN"

echo "[startup] downloading inputs"
gcloud storage rsync --recursive "$GS_RUN/input" "$DATA/input" || fail "input download failed"
du -sh "$DATA/input"

PY=/opt/conda/bin/python
[ -x "$PY" ] || PY=$(command -v python3)
echo "[startup] python: $PY ($($PY --version 2>&1))"

__PIP_BLOCK__

__ENV_BLOCK__
export MBO_INPUT="$DATA/input"
export MBO_OUTPUT="$DATA/output"

state RUNNING
echo "[startup] running the pipeline"
"$PY" "$DATA/worker_main.py" "$DATA/job.json"
RC=$?
echo "[startup] pipeline exited with $RC"

echo "[startup] uploading outputs"
gcloud storage rsync --recursive "$DATA/output" "$GS_RUN/output" || fail "output upload failed"
du -sh "$DATA/output"

if [ "$RC" -eq 0 ]; then
    state DONE
else
    state FAILED
fi
teardown
exit "$RC"
"""


def _pip_block(requirements: list, python_var: str = '"$PY"') -> str:
    """Bash that installs ``requirements``, failing the run if it cannot."""
    if not requirements:
        return 'echo "[startup] no extra requirements"'
    quoted = " ".join(shlex.quote(r) for r in requirements)
    return (
        'echo "[startup] installing: ' + quoted.replace('"', "'") + '"\n'
        f"{python_var} -m pip install --upgrade pip >/dev/null 2>&1\n"
        f'{python_var} -m pip install {quoted} || fail "pip install failed"'
    )


def _env_block(env: dict | None) -> str:
    """Bash exporting each ``env`` entry, quoted."""
    if not env:
        return ""
    return "\n".join(f"export {k}={shlex.quote(str(v))}" for k, v in env.items())


def build_startup_script(
    uri_run: str,
    run_id: str,
    requirements: list | None = None,
    env: dict | None = None,
    self_delete: bool = True,
    device_name: str = DEVICE_NAME_DATA,
    dir_data: str = DIR_DATA,
) -> str:
    """
    Render the worker's startup script.

    Parameters
    ----------
    uri_run : str
        ``gs://bucket/prefix/run_id`` - the run's staging directory.
    run_id : str
        Identifier echoed into the log.
    requirements : list of str, optional
        pip requirements installed before the pipeline is imported.
    env : dict, optional
        Environment variables exported for the pipeline.
    self_delete : bool
        Delete the instance when the job finishes. False leaves the box up for
        debugging (and billing).
    device_name : str
        ``device_name`` given to the attached scratch disk; determines the
        ``/dev/disk/by-id/google-*`` path.
    dir_data : str
        Mount point for the scratch disk.

    Returns
    -------
    str
        A complete bash script, ready to hand to instance metadata.
    """
    script = _TEMPLATE
    for token, value in (
        ("__GS_RUN__", uri_run.rstrip("/")),
        ("__RUN_ID__", run_id),
        ("__DIR_DATA__", dir_data),
        ("__DEVICE_NAME__", device_name),
        ("__SELF_DELETE__", "1" if self_delete else "0"),
        ("__PIP_BLOCK__", _pip_block(list(requirements or []))),
        ("__ENV_BLOCK__", _env_block(env)),
    ):
        script = script.replace(token, value)
    return script
