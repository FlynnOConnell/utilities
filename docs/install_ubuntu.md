---
orphan: true
---

# Installing on Ubuntu

Tested on Ubuntu 22.04 and 24.04 (GitHub Actions runners and bare `ubuntu:24.04`
containers), Python 3.12 and 3.13. Ubuntu 26.04 ships Python 3.14, which masknmf
does not support yet; install Python 3.13 there (see below).

## 1. System packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-tk git \
    libegl1 libgl1 libglib2.0-0t64 libx11-xcb1 libxkbcommon-x11-0 \
    libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-shape0 libxcb-util1 libxcb-xkb1 libxcb-xinerama0 \
    libdbus-1-3 libfontconfig1 mesa-vulkan-drivers
```

(On Ubuntu 22.04 the glib package is `libglib2.0-0`.)

- `python3-tk`: masknmf's dependency `roicat` imports tkinter at import time; the
  system Python has no tkinter without it (`ModuleNotFoundError: No module named 'tkinter'`).
- `libegl1`, `libxcb-*`, `libxkbcommon-x11-0`: Qt 6 needs these to load at all
  (`ImportError: libEGL.so.1`) and to open a window on X11/XWayland. A stock Ubuntu
  Desktop has most of them; a server or minimal install does not.
- `mesa-vulkan-drivers`: a software Vulkan adapter, so the viewer still renders on a
  machine without a working GPU driver.
- Optional: `zenity` (GNOME has it) for the native file dialog; without it the
  viewer's typed-path fields still work.

## 2. Install

```bash
python3 -m venv ~/mbo && source ~/mbo/bin/activate
pip install --upgrade pip
pip install git+https://github.com/FlynnOConnell/utilities
```

The fastplotlib and masknmf git trees this fork needs are declared in `pyproject.toml`,
so this one command is enough. The install pulls PyTorch's CUDA build (~4 GB on disk).

With uv, name all three git trees in one command (uv rejects git URLs that only appear
transitively):

```bash
uv venv --python 3.12 ~/mbo && source ~/mbo/bin/activate
uv pip install \
  "fastplotlib[imgui,notebook] @ git+https://github.com/fastplotlib/fastplotlib@b2132e3d11b9e2bd641e0bbfc0bbee3d413d1d88" \
  "masknmf[multisession,classification] @ git+https://github.com/apasarkar/masknmf-toolbox.git@imgui-traces" \
  git+https://github.com/FlynnOConnell/utilities
```

## 3. Check

```bash
mbo --check-install
mbo /path/to/data.tif
```

`mbo --check-install` reports the PyTorch build against your NVIDIA driver and prints
the exact wheel to install if the default CUDA build cannot see your card.

## Ubuntu 26.04 / Python 3.14

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13
uv venv --python 3.13 ~/mbo && source ~/mbo/bin/activate
uv pip install pip
pip install git+https://github.com/FlynnOConnell/utilities
```
