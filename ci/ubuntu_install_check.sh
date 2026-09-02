#!/usr/bin/env bash
# The install paths an Ubuntu user follows, and a smoke test of what they get.
#
#   ubuntu_install_check.sh one_step    PYTHON       fresh venv, `pip install git+utilities` alone
#   ubuntu_install_check.sh one_cmd     pip|uv PY    both specs in a single install command
#   ubuntu_install_check.sh two_step    PYTHON       masknmf from git first, then utilities
#   ubuntu_install_check.sh uv_two_step PYVER        the same through `uv venv` / `uv pip`
#   ubuntu_install_check.sh qt_libs     VENV         what Qt's xcb plugin cannot find on this box
#   ubuntu_install_check.sh smoke       VENV         imports, CLI, Qt, offscreen render, convert, masknmf
#   ubuntu_install_check.sh pytest      VENV SRCDIR  the repo test suite against the installed package
#
# Every check is recorded as a row in $GITHUB_STEP_SUMMARY; the script exits
# non-zero when any check that should pass did not.
#
# Env: UTIL_REF (default main), MASKNMF_REF (default imgui-traces),
#      VENV (default $HOME/venv), ARTIFACTS (default $HOME/artifacts).
set -uo pipefail

UTIL_REF="${UTIL_REF:-main}"
UTIL_URL="git+https://github.com/FlynnOConnell/utilities@${UTIL_REF}"
MASKNMF_REF="${MASKNMF_REF:-imgui-traces}"
MASKNMF_REQ="masknmf[multisession,classification] @ git+https://github.com/apasarkar/masknmf-toolbox.git@${MASKNMF_REF}"
# the ndwidget commit masknmf pins; naming it explicitly is what lets pip/uv satisfy
# utilities' bare `fastplotlib[imgui]` with the git tree instead of the PyPI wheel
FPL_REF="${FPL_REF:-b2132e3d11b9e2bd641e0bbfc0bbee3d413d1d88}"
FPL_REQ="fastplotlib[imgui,notebook] @ git+https://github.com/fastplotlib/fastplotlib@${FPL_REF}"
VENV="${VENV:-$HOME/venv}"
ARTIFACTS="${ARTIFACTS:-$HOME/artifacts}"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/stdout}"
WORK="${WORK:-$HOME/work}"
mkdir -p "$ARTIFACTS" "$WORK"
# never run from the checkout: `python -c` puts the cwd on sys.path and the repo's
# own mbo_utilities/ would shadow the installed package
cd "$WORK"

export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

FAILED=()

heading() {
  printf '\n### %s\n\n| result | check | time | detail |\n|---|---|---|---|\n' "$1" >> "$SUMMARY"
  echo; echo "=================== $1 ==================="
}

# _record MARK NAME SECONDS DETAIL
_record() {
  printf '| %s | %s | %ss | %s |\n' "$1" "$2" "$3" "${4//|/\\|}" >> "$SUMMARY"
  echo ">>> $1  $2  (${3}s)  $4"
}

# _run LOGFILE CMD... ; echoes the exit code
_run() {
  local log=$1; shift
  echo "::group::$*"
  local rc=0
  "$@" >"$log" 2>&1 || rc=$?
  cat "$log"
  echo "::endgroup::"
  return $rc
}

# check NAME CMD...   -> PASS when CMD exits 0
check() {
  local name=$1; shift
  local log; log=$(mktemp)
  local t0=$SECONDS rc=0
  _run "$log" "$@" || rc=$?
  local dt=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then
    _record PASS "$name" "$dt" "$(tail -n 1 "$log" | cut -c1-160)"
  else
    FAILED+=("$name")
    _record FAIL "$name" "$dt" "rc=$rc: $(grep -iE 'error|Traceback|not found|No matching|requires|because|conflict' "$log" | tail -n 3 | tr '\n' ' ' | cut -c1-300)"
    { echo; echo "<details><summary>FAIL: $name (last 30 lines)</summary>"; echo; echo '```'; tail -n 30 "$log" | cut -c1-300; echo '```'; echo '</details>'; } >> "$SUMMARY"
  fi
  rm -f "$log"
  return 0
}

# expect_fail NAME CMD... -> PASS when CMD fails; records the error line
expect_fail() {
  local name=$1; shift
  local log; log=$(mktemp)
  local t0=$SECONDS rc=0
  _run "$log" "$@" || rc=$?
  local dt=$((SECONDS - t0))
  if [ $rc -ne 0 ]; then
    _record "FAILS (as predicted)" "$name" "$dt" "$(grep -iE 'ERROR|requires a different Python|No matching distribution|Because|not found' "$log" | head -n 3 | tr '\n' ' ' | cut -c1-300)"
  else
    FAILED+=("$name (was expected to fail but succeeded)")
    _record "UNEXPECTED PASS" "$name" "$dt" "$(tail -n 1 "$log" | cut -c1-160)"
  fi
  rm -f "$log"
  return 0
}

finish() {
  echo
  if [ ${#FAILED[@]} -eq 0 ]; then
    echo "all checks in this scenario passed" | tee -a "$SUMMARY"
    exit 0
  fi
  {
    echo
    echo "**failed:**"
    for f in "${FAILED[@]}"; do echo "- $f"; done
  } >> "$SUMMARY"
  printf 'FAILED: %s\n' "${FAILED[@]}"
  exit 1
}

pyinfo() {
  local py=$1
  "$py" -c 'import sys, platform; print("python", sys.version.split()[0], "|", platform.platform(), "| glibc", platform.libc_ver()[1])'
}

# ---------------------------------------------------------------------------
scenario_one_step() {
  local python=$1
  heading "one-step: pip install ${UTIL_URL} (python: $($python --version 2>&1))"
  check "interpreter" pyinfo "$python"
  rm -rf "$VENV"
  check "python -m venv" "$python" -m venv "$VENV"
  check "pip upgrade" "$VENV/bin/python" -m pip install -q --upgrade pip
  check "pip version" "$VENV/bin/python" -m pip --version
  if [ "${EXPECT:-fail}" = "pass" ]; then
    check "pip install $UTIL_URL" "$VENV/bin/python" -m pip install "$UTIL_URL"
  else
    expect_fail "pip install $UTIL_URL" "$VENV/bin/python" -m pip install "$UTIL_URL"
  fi
  finish
}

# uv one-step: `uv venv --python PYVER` (whatever interpreter uv picks) then `uv pip install git+utilities` alone
scenario_uv_one_step() {
  local pyver=$1
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    export PATH="$HOME/.local/bin:$PATH"
  fi
  heading "uv one-step: uv venv --python $pyver; uv pip install ${UTIL_URL}"
  check "uv version" uv --version
  rm -rf "$VENV"
  check "uv venv --python $pyver" uv venv --python "$pyver" "$VENV"
  check "interpreter" pyinfo "$VENV/bin/python"
  if [ "${EXPECT:-fail}" = "pass" ]; then
    check "uv pip install $UTIL_URL" uv pip install --python "$VENV/bin/python" "$UTIL_URL"
  else
    expect_fail "uv pip install $UTIL_URL" uv pip install --python "$VENV/bin/python" "$UTIL_URL"
  fi
  uv pip install --python "$VENV/bin/python" pip >/dev/null 2>&1 || true
  check "disk after install" df -h "$HOME"
  finish
}

# one command, both specs: pip resolves utilities' bare `masknmf[...]` against the URL given beside it
scenario_one_cmd() {
  local tool=$1 python=$2 with_fpl=${3:-}
  local specs=("$MASKNMF_REQ" "$UTIL_URL")
  if [ "$with_fpl" = fpl ]; then
    specs=("$FPL_REQ" "$MASKNMF_REQ" "$UTIL_URL")
    heading "one command, three specs ($tool): install \"$FPL_REQ\" \"$MASKNMF_REQ\" $UTIL_URL"
  else
    heading "one command, two specs ($tool): install \"$MASKNMF_REQ\" $UTIL_URL"
  fi
  rm -rf "$VENV"
  if [ "$tool" = uv ]; then
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
      export PATH="$HOME/.local/bin:$PATH"
    fi
    check "uv venv --python $python" uv venv --python "$python" "$VENV"
    check "interpreter" pyinfo "$VENV/bin/python"
    check "uv pip install ${#specs[@]} specs" uv pip install --python "$VENV/bin/python" "${specs[@]}"
    uv pip install --python "$VENV/bin/python" pip >/dev/null 2>&1 || true
  else
    check "interpreter" pyinfo "$python"
    check "python -m venv" "$python" -m venv "$VENV"
    check "pip upgrade" "$VENV/bin/python" -m pip install -q --upgrade pip
    check "pip install ${#specs[@]} specs" "$VENV/bin/python" -m pip install "${specs[@]}"
  fi
  check "disk after install" df -h "$HOME"
  finish
}

scenario_two_step() {
  local python=$1
  heading "two-step: masknmf@${MASKNMF_REF} from git, then ${UTIL_URL} (python: $($python --version 2>&1))"
  check "interpreter" pyinfo "$python"
  rm -rf "$VENV"
  check "python -m venv" "$python" -m venv "$VENV"
  check "pip upgrade" "$VENV/bin/python" -m pip install -q --upgrade pip
  check "pip install masknmf[multisession,classification]@$MASKNMF_REF" "$VENV/bin/python" -m pip install "$MASKNMF_REQ"
  check "pip install utilities@$UTIL_REF" "$VENV/bin/python" -m pip install "$UTIL_URL"
  check "disk after install" df -h "$HOME"
  finish
}

scenario_uv_two_step() {
  local pyver=$1
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    export PATH="$HOME/.local/bin:$PATH"
  fi
  heading "uv two-step: uv venv --python $pyver; uv pip install masknmf@${MASKNMF_REF}, then ${UTIL_URL}"
  check "uv version" uv --version
  rm -rf "$VENV" "$VENV-onestep"
  check "uv venv --python $pyver" uv venv --python "$pyver" "$VENV"
  check "interpreter" pyinfo "$VENV/bin/python"
  uv venv --python "$pyver" "$VENV-onestep" >/dev/null 2>&1
  expect_fail "uv pip install $UTIL_URL alone" uv pip install --python "$VENV-onestep/bin/python" "$UTIL_URL"
  check "uv pip install masknmf[multisession,classification]@$MASKNMF_REF" uv pip install --python "$VENV/bin/python" "$MASKNMF_REQ"
  check "uv pip install utilities@$UTIL_REF" uv pip install --python "$VENV/bin/python" "$UTIL_URL"
  # pip itself, so the smoke scenario's `pip check` / `pip list` work in a uv venv
  uv pip install --python "$VENV/bin/python" pip >/dev/null 2>&1 || true
  check "disk after install" df -h "$HOME"
  finish
}

_qt_plugin_missing_libs() {
  local py=$1
  local dir; dir=$("$py" -c "import PyQt6, os; print(os.path.dirname(PyQt6.__file__))")
  local plugin="$dir/Qt6/plugins/platforms/libqxcb.so"
  echo "plugin: $plugin"
  [ -f "$plugin" ] || { echo "xcb plugin not shipped"; return 1; }
  local missing
  missing=$( (ldd "$plugin"; ldd "$dir/Qt6/lib/libQt6Gui.so.6" 2>/dev/null; ldd "$dir/Qt6/lib/libQt6Widgets.so.6" 2>/dev/null) | grep "not found" | awk '{print $1}' | sort -u)
  if [ -n "$missing" ]; then
    echo "shared libraries Qt cannot find on this machine:"
    echo "$missing" | sed 's/^/  /'
    return 1
  fi
  echo "every shared library Qt's xcb plugin needs is present"
}

scenario_qt_libs() {
  local venv=$1 py="$1/bin/python"
  heading "Qt runtime libs on this machine (before any extra apt packages)"
  check "system libs Qt xcb plugin links against" _qt_plugin_missing_libs "$py"
  check "PyQt6 QApplication, QT_QPA_PLATFORM=offscreen" env QT_QPA_PLATFORM=offscreen "$py" -c "from PyQt6.QtWidgets import QApplication; QApplication([]); print('offscreen QApplication ok')"
  if command -v xvfb-run >/dev/null 2>&1; then
    check "PyQt6 QApplication on xcb under Xvfb" xvfb-run -a env QT_QPA_PLATFORM=xcb "$py" -c "from PyQt6.QtWidgets import QApplication; QApplication([]); print('xcb QApplication ok')"
  fi
  finish
}

_write_synthetic() {
  local py=$1
  "$py" - "$WORK" <<'EOF'
import sys, numpy as np, tifffile
from pathlib import Path
work = Path(sys.argv[1]); work.mkdir(parents=True, exist_ok=True)
rng = np.random.RandomState(0)
t, y, x = 40, 64, 72
data = rng.normal(400, 60, (t, y, x)).astype(np.float32)
yy, xx = np.mgrid[:y, :x]
for i, (cy, cx) in enumerate([(16, 20), (40, 50), (30, 30)]):
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 < 30
    sig = 300 * (0.5 + 0.5 * np.sin(np.arange(t) / 3.0 + i))
    data[:, mask] += sig[:, None]
tifffile.imwrite(work / "synthetic.tif", data.clip(0, 4095).astype(np.int16), imagej=True, metadata={"axes": "TYX"})
print("wrote", work / "synthetic.tif", data.shape)
EOF
}

scenario_smoke() {
  local venv=$1 py="$1/bin/python" mbo="$1/bin/mbo"
  heading "smoke: installed package on $(. /etc/os-release && echo "$PRETTY_NAME")"
  check "interpreter" pyinfo "$py"
  check "import mbo_utilities, __version__" "$py" -c "import mbo_utilities as m; print('mbo_utilities.__version__ =', m.__version__)"
  check "mbo --version" "$mbo" --version
  check "mbo --help" "$mbo" --help
  check "mbo formats" "$mbo" formats
  check "mbo --check-install" "$mbo" --check-install
  check "pip check (declared dependency conflicts)" "$py" -m pip check
  check "key package versions" bash -c "$py -m pip list --format=freeze | grep -iE '^(fastplotlib|masknmf|pygfx|wgpu|imgui-bundle|rendercanvas|torch|torchvision|numpy|opencv-[a-z-]+|PyQt6|PyQt6-Qt6|imagecodecs|numba|llvmlite|zarr|tifffile|dask|scikit-image|roicat|utilities|mbo-utilities|glfw|cvxpy|jupyterlab|setuptools)=='"
  "$py" -m pip list --format=json > "$ARTIFACTS/pip-list.json" 2>/dev/null || true
  "$py" -m pip freeze > "$ARTIFACTS/pip-freeze.txt" 2>/dev/null || true
  check "where fastplotlib / masknmf came from" "$py" - <<'EOF'
import importlib.metadata as m, json
for n in ("fastplotlib", "masknmf", "utilities", "mbo-utilities"):
    try:
        d = m.distribution(n)
    except m.PackageNotFoundError:
        print(f"{n}: not installed"); continue
    du = d.read_text("direct_url.json")
    src = json.loads(du) if du else {}
    vcs = src.get("vcs_info", {})
    print(f"{n} {d.version}: {src.get('url', 'PyPI')} {vcs.get('requested_revision', '')} {vcs.get('commit_id', '')[:12]}")
EOF
  check "import fastplotlib.widgets.nd_widget (needs the ndwidget branch)" "$py" -c "import fastplotlib.widgets.nd_widget as w; print(w.__file__)"
  check "import masknmf" "$py" -c "import masknmf, masknmf.compression.preprocessing, masknmf.demixing, masknmf.visualization.classification_vis; print(masknmf.__file__)"
  check "import torch" "$py" -c "import torch; print(torch.__version__, 'cuda available:', torch.cuda.is_available())"
  check "import every mbo_utilities module" "$py" - <<'EOF'
import importlib, pkgutil, sys, os
os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")
import mbo_utilities
bad = []
for info in pkgutil.walk_packages(mbo_utilities.__path__, "mbo_utilities."):
    if ".hpc" in info.name or "isoview" in info.name:
        continue
    try:
        importlib.import_module(info.name)
    except Exception as e:  # noqa: BLE001
        bad.append(f"{info.name}: {type(e).__name__}: {e}")
print(f"imported {sum(1 for _ in pkgutil.walk_packages(mbo_utilities.__path__, 'mbo_utilities.'))} modules, {len(bad)} failed")
print("\n".join(bad))
sys.exit(1 if bad else 0)
EOF
  check "import GUI (offscreen): run_gui, manual_roi, file_dialog" env RENDERCANVAS_FORCE_OFFSCREEN=1 "$py" -c "import mbo_utilities.gui.run_gui; from mbo_utilities.gui.manual_roi import roi_widgets_available; import mbo_utilities.gui.widgets.file_dialog; print('roi widgets available:', roi_widgets_available())"
  check "wgpu adapters (python -m wgpu.diagnostics)" "$py" -m wgpu.diagnostics
  check "fastplotlib offscreen render" env RENDERCANVAS_FORCE_OFFSCREEN=1 "$py" -c "import numpy as np, fastplotlib as fpl; fig = fpl.Figure(size=(300, 300)); fig[0, 0].add_image(np.random.rand(64, 64).astype('f4')); fig.show(); fig.canvas.draw(); print('offscreen render ok, adapter:', fig.canvas.__class__.__module__)"
  check "DataVis offscreen show/draw/close" env RENDERCANVAS_FORCE_OFFSCREEN=1 "$py" -c "
import numpy as np
from mbo_utilities import DataVis
v = DataVis(np.random.randint(0, 4000, (6, 1, 64, 64), dtype='int16'), size=(1000, 800))
v.show(); v.iw.figure.canvas.draw(); v.iw.figure.canvas.draw(); v.close(); print('DataVis ok')"
  check "PyQt6 QApplication offscreen" env QT_QPA_PLATFORM=offscreen "$py" -c "from PyQt6.QtWidgets import QApplication; QApplication([]); print('qt ok')"
  check "rendercanvas.pyqt6 imports (the Linux desktop canvas)" env QT_QPA_PLATFORM=offscreen "$py" -c "from rendercanvas.pyqt6 import RenderCanvas; print(RenderCanvas)"

  check "write synthetic ScanImage-free tiff" _write_synthetic "$py"
  check "mbo info synthetic.tif" "$mbo" info "$WORK/synthetic.tif"
  check "mbo convert -> .zarr" "$mbo" convert "$WORK/synthetic.tif" "$WORK/out_zarr" -e .zarr --overwrite
  check "mbo convert -> .tiff" "$mbo" convert "$WORK/synthetic.tif" "$WORK/out_tiff" -e .tiff --overwrite
  check "mbo convert -> .h5" "$mbo" convert "$WORK/synthetic.tif" "$WORK/out_h5" -e .h5 --overwrite
  check "mbo convert -> .bin (suite2p)" "$mbo" convert "$WORK/synthetic.tif" "$WORK/out_bin" -e .bin --overwrite
  check "outputs readable and equal to the source" "$py" - "$WORK" <<'EOF'
import sys, numpy as np, tifffile
from pathlib import Path
import mbo_utilities as mbo
work = Path(sys.argv[1])
src = tifffile.imread(work / "synthetic.tif")
for d in ("out_zarr", "out_tiff", "out_h5", "out_bin"):
    files = [p for p in (work / d).rglob("*") if p.suffix in (".zarr", ".tif", ".tiff", ".h5", ".bin")]
    target = files[0] if files else (work / d)
    arr = mbo.imread(target)
    back = np.asarray(arr[:]).squeeze()
    ok = back.shape == src.shape and np.array_equal(back, src)
    print(f"{d}: {target.name} shape={back.shape} dtype={back.dtype} equal={ok}")
    if not ok:
        sys.exit(1)
EOF
  check "python imread/imwrite roundtrip TZYX -> zarr" "$py" - "$WORK" <<'EOF'
import sys, numpy as np
from pathlib import Path
import mbo_utilities as mbo
out = Path(sys.argv[1]) / "rt"
data = np.random.RandomState(3).randint(0, 4096, size=(8, 3, 32, 40), dtype=np.int16)
arr = mbo.imread(data, dims="TZYX")
mbo.imwrite(arr, out, ext=".zarr", overwrite=True)
back = mbo.imread(next(out.rglob("*.zarr")))
b = np.asarray(back[:])
print("back", b.shape, back.dtype)
assert b.squeeze().shape == data.shape and np.array_equal(b.squeeze(), data)
print("roundtrip ok")
EOF
  check "mbo roi-run --register masknmf --process none (CPU, tiny movie)" timeout 1500 "$mbo" roi-run "$WORK/synthetic.tif" -o "$WORK/roi_out" --register masknmf --process none
  check "registered plane dir opens" "$py" - "$WORK" <<'EOF'
import sys
from pathlib import Path
import mbo_utilities as mbo
out = Path(sys.argv[1]) / "roi_out"
listing = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
print("\n".join(listing[:40]))
planes = sorted(p for p in out.iterdir() if p.is_dir())
print("plane dirs:", [p.name for p in planes])
if planes:
    arr = mbo.imread(planes[0])
    print("opened", planes[0].name, arr.shape, arr.dtype)
EOF
  finish
}

scenario_pytest() {
  local venv=$1 src=$2 py="$1/bin/python"
  heading "pytest: $src against the installed package"
  check "install pytest" "$py" -m pip install -q pytest pytest-timeout
  local log="$ARTIFACTS/pytest.log"
  local t0=$SECONDS rc=0
  # the tests import `tests.*`, so the source root must be the cwd; move its
  # mbo_utilities/ aside so the installed package is the one under test
  [ -d "$src/mbo_utilities" ] && mv "$src/mbo_utilities" "$src/_mbo_utilities_source"
  (
    cd "$src" && RENDERCANVAS_FORCE_OFFSCREEN=1 QT_QPA_PLATFORM=offscreen \
      timeout 3000 "$py" -m pytest tests -q -rfEs --timeout=900 -p no:cacheprovider \
      --junitxml="$ARTIFACTS/junit.xml" --ignore=tests/local
  ) 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  local dt=$((SECONDS - t0))
  local tail; tail=$(grep -E '^(=+ .*(passed|failed|error|skipped).* =+|FAILED|ERROR) ' "$log" | tail -n 40)
  if [ $rc -eq 0 ]; then
    _record PASS "pytest tests/" "$dt" "$(tail -n 1 "$log" | cut -c1-200)"
  else
    FAILED+=("pytest tests/ (rc=$rc)")
    _record FAIL "pytest tests/" "$dt" "rc=$rc: $(tail -n 1 "$log" | cut -c1-200)"
  fi
  {
    echo
    echo '<details><summary>pytest short summary</summary>'
    echo
    echo '```'
    echo "$tail"
    echo '```'
    echo '</details>'
  } >> "$SUMMARY"
  finish
}

case "${1:-}" in
  one_step)    scenario_one_step "$2" ;;
  one_cmd)     scenario_one_cmd "$2" "$3" "${4:-}" ;;
  uv_one_step) scenario_uv_one_step "$2" ;;
  two_step)    scenario_two_step "$2" ;;
  uv_two_step) scenario_uv_two_step "$2" ;;
  qt_libs)     scenario_qt_libs "$2" ;;
  smoke)       scenario_smoke "$2" ;;
  pytest)      scenario_pytest "$2" "$3" ;;
  *) echo "usage: $0 {one_step PYTHON|one_cmd pip|uv PY|two_step PYTHON|uv_two_step PYVER|qt_libs VENV|smoke VENV|pytest VENV SRCDIR}"; exit 2 ;;
esac
