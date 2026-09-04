"""Framework-agnostic ROI annotation model.

``RoiLabelStore`` holds everything the manual ROI GUI edits — a per-plane
label volume, per-ROI records (plane, area, class, note) and the
user-defined class-label set — with no GUI imports, so the same model can
back a widget, a script, or a batch tool.

The volume is ``(P, Y, X)`` uint16: one plane per combination of the
data's scrolling dims (plain z for most data — arrays without depth get a
single plane). When channels or other scroll axes key their own masks,
``plane_axes`` records the layout ``((dim, size), ...)`` with z last, so
``nz`` is the product and a z-only store keeps ``plane == z``. T is always
shared. ROI ``i`` owns label value ``i + 1``; 0 is background. ROIs can
never overlap: pixels already claimed by another ROI are dropped from a new
one. Deleting an ROI renumbers the labels above it, so values stay
contiguous ``1..N``.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "UNLABELED",
    "CLASS_COLORS",
    "ROI_COLORS",
    "class_color",
    "RoiRecord",
    "RoiLabelStore",
]

UNLABELED = -1

# one color per class label (tab10, matched to masknmf's classification GUI
# so shared label sets look the same in both tools), float rgb in 0-1
CLASS_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.12, 0.47, 0.71),
    (1.00, 0.50, 0.05),
    (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74),
    (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76),
    (0.50, 0.50, 0.50),
    (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81),
)


def class_color(index: int) -> tuple[float, float, float]:
    """rgb in 0-1 for a class label index (wraps past the palette end)"""
    return CLASS_COLORS[index % len(CLASS_COLORS)]


def _make_roi_colors() -> np.ndarray:
    # one fully saturated color per unclassified ROI. hues are shuffled so
    # consecutive ROIs contrast, and saturating them keeps fills readable
    # over gnuplot2, which owns most of the pale end of the spectrum
    hues = np.random.default_rng(0).permutation(180)
    return np.array(
        [
            [int(round(c * 255)) for c in colorsys.hsv_to_rgb(h / 180.0, 1.0, 1.0)]
            for h in hues
        ],
        dtype=np.uint8,
    )


# (180, 3) uint8 fill colors for ROIs with no class label; index with
# ``roi_index % len(ROI_COLORS)``
ROI_COLORS = _make_roi_colors()


@dataclass
class RoiRecord:
    """Per-ROI metadata; the pixels live in the store's label volume."""

    z: int
    area: int
    class_index: int = UNLABELED
    note: str = ""
    uid: int = 0  # persistent id, never reused; 0 = unassigned
    source: str = ""  # "" = drawn by hand; "<run name>:<row>" = promoted
    color: tuple[int, int, int] | None = None  # explicit group color, uint8 rgb


class RoiLabelStore:
    """Label volume + per-ROI records + the class-label name set.

    Mutations record which z-planes changed in ``dirty_planes`` so a
    persistence layer can write incrementally (see ``ngff.LabelsZarr``);
    the saver is expected to clear the set after writing.
    """

    def __init__(
        self,
        nz: int,
        ny: int,
        nx: int,
        label_names: tuple[str, ...] = (),
        min_pixels: int = 1,
        labels: np.ndarray | None = None,
        rois: list[RoiRecord] | None = None,
        next_uid: int = 0,
    ):
        if labels is None:
            labels = np.zeros((int(nz), int(ny), int(nx)), np.uint16)
        else:
            labels = np.asarray(labels, np.uint16)
            if labels.shape != (int(nz), int(ny), int(nx)):
                raise ValueError(f"labels shape {labels.shape} != ({nz}, {ny}, {nx})")
        self.labels = labels
        self.rois: list[RoiRecord] = list(rois) if rois else []
        if self.rois and int(labels.max(initial=0)) != len(self.rois):
            raise ValueError(
                f"label volume holds {int(labels.max(initial=0))} labels but "
                f"{len(self.rois)} records were given"
            )
        self.label_names: tuple[str, ...] = tuple(str(n) for n in label_names)
        # what the plane axis means: ((dim name, size), ...) over the data's
        # scrolling dims with z last, so nz == product; empty = plain z planes
        self.plane_axes: tuple[tuple[str, int], ...] = ()
        self.min_pixels = int(min_pixels)
        self.dirty_planes: set[int] = set()
        self.next_uid = max(
            int(next_uid), max((r.uid for r in self.rois), default=0) + 1, 1
        )
        seen: set[int] = set()
        for r in self.rois:
            if r.uid <= 0 or r.uid in seen:
                r.uid = self.next_uid
                self.next_uid += 1
            seen.add(r.uid)

    # ------------------------------------------------------------------
    # shape
    # ------------------------------------------------------------------

    @property
    def nz(self) -> int:
        return self.labels.shape[0]

    @property
    def ny(self) -> int:
        return self.labels.shape[1]

    @property
    def nx(self) -> int:
        return self.labels.shape[2]

    # ------------------------------------------------------------------
    # roi mutations
    # ------------------------------------------------------------------

    def add_roi(self, z: int, mask: np.ndarray, source: str = "") -> int | None:
        """Claim the free pixels of a boolean ``(Y, X)`` mask on plane ``z``.

        Pixels already owned by another ROI are dropped. Returns the new
        ROI's index, or None when fewer than ``min_pixels`` free pixels
        remain (the volume is untouched then). ``source`` names where the
        mask came from ("" = drawn by hand).
        """
        z = int(z)
        plane = self.labels[z]
        rows, cols = np.nonzero(np.asarray(mask, bool) & (plane == 0))
        if rows.size < self.min_pixels:
            return None
        self.rois.append(
            RoiRecord(z=z, area=int(rows.size), uid=self.next_uid, source=str(source))
        )
        self.next_uid += 1
        plane[rows, cols] = len(self.rois)
        self.dirty_planes.add(z)
        return len(self.rois) - 1

    def delete_roi(self, index: int) -> bool:
        """Drop one ROI and renumber the label values above it."""
        if not 0 <= index < len(self.rois):
            return False
        # the deleted ROI's plane plus every plane holding a renumbered one
        self.dirty_planes.add(self.rois[index].z)
        self.dirty_planes.update(r.z for r in self.rois[index + 1 :])
        self.labels[self.labels == index + 1] = 0
        self.labels[self.labels > index + 1] -= 1
        self.rois.pop(index)
        return True

    def clear(self) -> None:
        self.labels[:] = 0
        self.rois.clear()
        self.dirty_planes.update(range(self.nz))

    def snapshot(self) -> RoiLabelStore:
        """Deep copy (volume, records, names, ``next_uid``) that later
        mutations of either store cannot reach."""
        out = RoiLabelStore(
            self.nz,
            self.ny,
            self.nx,
            label_names=self.label_names,
            min_pixels=self.min_pixels,
            labels=self.labels.copy(),
            rois=[replace(r) for r in self.rois],
            next_uid=self.next_uid,
        )
        out.plane_axes = self.plane_axes
        return out

    # ------------------------------------------------------------------
    # labels / notes
    # ------------------------------------------------------------------

    def add_label_name(self, name: str) -> int:
        """Add a class name to the label set; returns its index (existing
        names return their current index instead of duplicating)."""
        name = str(name).strip()
        if not name:
            raise ValueError("label name must be non-empty")
        if name in self.label_names:
            return self.label_names.index(name)
        self.label_names = (*self.label_names, name)
        return len(self.label_names) - 1

    def set_class(self, index: int, class_index: int) -> None:
        """Assign a class label to ROI ``index``; UNLABELED (-1) clears."""
        if not UNLABELED <= class_index < len(self.label_names):
            raise IndexError(f"class index {class_index} out of range")
        self.rois[index].class_index = int(class_index)

    def set_note(self, index: int, note: str) -> None:
        self.rois[index].note = str(note)

    def set_color(self, index: int, rgb: tuple[int, int, int] | None) -> None:
        """Give ROI ``index`` an explicit display color; None reverts it to
        the class / hue color."""
        self.rois[index].color = None if rgb is None else tuple(int(v) for v in rgb)

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def uid_index(self, uid: int) -> int | None:
        """ROI index currently holding ``uid``, or None when it is gone."""
        for i, r in enumerate(self.rois):
            if r.uid == uid:
                return i
        return None

    def roi_at(self, z: int, row: int, col: int) -> int:
        """ROI index under a pixel, or -1 for background/out of range."""
        if not (0 <= z < self.nz and 0 <= row < self.ny and 0 <= col < self.nx):
            return -1
        return int(self.labels[z, row, col]) - 1

    def class_counts(self) -> list[int]:
        """number of ROIs per class name, in label-set order"""
        counts = [0] * len(self.label_names)
        for r in self.rois:
            if 0 <= r.class_index < len(counts):
                counts[r.class_index] += 1
        return counts

    def rois_on_plane(self, z: int) -> list[int]:
        return [i for i, r in enumerate(self.rois) if r.z == int(z)]

    @property
    def counts(self) -> list[int]:
        """per-ROI pixel counts, in ROI order"""
        return [r.area for r in self.rois]

    def roi_rgb(self, index: int) -> tuple[int, int, int]:
        """display color of one ROI: its explicit group color when set, else
        its class color when classified, else its own hue from
        ``ROI_COLORS`` (uint8 rgb)"""
        record = self.rois[index]
        if record.color is not None:
            return tuple(int(v) for v in record.color)
        if record.class_index >= 0:
            return tuple(int(round(c * 255)) for c in class_color(record.class_index))
        return tuple(int(v) for v in ROI_COLORS[index % len(ROI_COLORS)])

    def color_lut(self) -> np.ndarray:
        """(num_rois + 1, 3) uint8 lookup table indexed by label value;
        row 0 (background) is black"""
        lut = np.zeros((len(self.rois) + 1, 3), np.uint8)
        for i in range(len(self.rois)):
            lut[i + 1] = self.roi_rgb(i)
        return lut
