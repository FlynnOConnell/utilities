"""OME-NGFF-style labels zarr persistence for ``RoiLabelStore``.

Layout matches the repo's existing writer (``_add_suite2p_labels`` in
``arrays/suite2p.py``): a group whose attrs carry ``{"version": "0.5",
"labels": ["0"]}`` and whose ``0`` array is the ``(Z, Y, X)`` uint32 label
volume, chunked one plane per chunk, with ``image-label`` attrs. On top of
that:

- ``image-label.colors`` holds one rgba per label value (class color when
  the ROI is classified, its own hue otherwise), so napari and friends
  render the same colors as the GUI.
- ``image-label.properties`` holds one entry per label value with the
  mbo-specific per-ROI state: ``class-index`` / ``class`` (name), ``note``,
  ``z`` and ``area``. That is where the annotation round-trips from.
- the root ``mbo`` attr carries the label-name set and the source image
  path (a sidecar store has no ``../../0`` image group to point at, so
  provenance lives here instead of ``image-label.source``).

``load`` also accepts a labels zarr written by other tools (no
``properties``): records are then derived from the volume itself and come
back unclassified.

Chunking one plane per chunk is what makes the GUI's autosave cheap:
``save_dirty`` rewrites only the planes a mutation touched, plus attrs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mbo_utilities.annotation.store import RoiLabelStore, RoiRecord

__all__ = ["LabelsZarr"]

_COMPRESSION_LEVEL = 4


class LabelsZarr:
    """Incremental writer / reader for one labels zarr on disk."""

    def __init__(self, path):
        self.path = Path(path)
        self._root = None
        self._array = None

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------

    def _open_for_write(self, store: RoiLabelStore):
        import zarr
        from zarr.codecs import BytesCodec, GzipCodec

        shape = (store.nz, store.ny, store.nx)
        if self._array is not None and tuple(self._array.shape) == shape:
            return
        root = zarr.open_group(str(self.path), mode="a")
        arr = root.get("0")
        if arr is None or tuple(arr.shape) != shape or arr.dtype != np.uint32:
            arr = zarr.create(
                store=root.store,
                path="0",
                shape=shape,
                chunks=(1, store.ny, store.nx),
                dtype=np.uint32,
                codecs=[BytesCodec(), GzipCodec(level=_COMPRESSION_LEVEL)],
                overwrite=True,
            )
        self._root, self._array = root, arr

    def _write_attrs(self, store: RoiLabelStore, source_path=None):
        colors = [
            {"label-value": i + 1, "rgba": [*store.roi_rgb(i), 255]}
            for i in range(len(store.rois))
        ]
        properties = [
            {
                "label-value": i + 1,
                "class-index": int(r.class_index),
                "class": (
                    store.label_names[r.class_index] if r.class_index >= 0 else ""
                ),
                "note": r.note,
                "z": int(r.z),
                "area": int(r.area),
            }
            for i, r in enumerate(store.rois)
        ]
        self._root.attrs.update(
            {
                "version": "0.5",
                "labels": ["0"],
                "mbo": {
                    "label_names": list(store.label_names),
                    "source_image": str(source_path) if source_path else None,
                },
            }
        )
        self._array.attrs.update(
            {
                "version": "0.5",
                "image-label": {
                    "version": "0.5",
                    "colors": colors,
                    "properties": properties,
                },
            }
        )

    def save(self, store: RoiLabelStore, source_path=None) -> None:
        """Write the full volume and all metadata."""
        self._open_for_write(store)
        for z in range(store.nz):
            self._array[z] = store.labels[z].astype(np.uint32)
        self._write_attrs(store, source_path)
        store.dirty_planes.clear()

    def save_dirty(self, store: RoiLabelStore, source_path=None) -> None:
        """Write only the planes mutated since the last save, plus metadata.

        Metadata (classes, notes, label names) always rewrites — it is tiny
        and mutations to it never dirty a plane.
        """
        self._open_for_write(store)
        for z in sorted(store.dirty_planes):
            self._array[z] = store.labels[z].astype(np.uint32)
        self._write_attrs(store, source_path)
        store.dirty_planes.clear()

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------

    @staticmethod
    def load(path) -> RoiLabelStore:
        """Rebuild a ``RoiLabelStore`` from a labels zarr.

        Raises ``FileNotFoundError`` when there is no store at ``path`` and
        ``ValueError`` when the store is not a labels group this reader
        understands.
        """
        import zarr

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        root = zarr.open_group(str(path), mode="r")
        if "0" not in root:
            raise ValueError(f"{path} has no '0' labels array")
        arr = root["0"]
        if arr.ndim != 3:
            raise ValueError(f"labels array must be (Z, Y, X), got {arr.shape}")
        volume = np.asarray(arr[:])
        num = int(volume.max(initial=0))
        if num > np.iinfo(np.uint16).max:
            raise ValueError(f"{num} labels exceed the uint16 working range")
        volume = volume.astype(np.uint16)

        mbo = dict(root.attrs.get("mbo") or {})
        label_names = tuple(str(n) for n in (mbo.get("label_names") or ()))

        image_label = dict(arr.attrs.get("image-label") or {})
        by_value = {
            int(p["label-value"]): p
            for p in (image_label.get("properties") or ())
            if "label-value" in p
        }
        # per-label fallbacks (used when a foreign zarr has no properties),
        # computed per plane so cost is one volume pass, not one per label
        areas = np.bincount(volume.ravel(), minlength=num + 1)
        z_of = np.full(num + 1, -1)
        for z in range(volume.shape[0]):
            for value in np.unique(volume[z]):
                if value and z_of[value] < 0:
                    z_of[value] = z
        rois: list[RoiRecord] = []
        for value in range(1, num + 1):
            area = int(areas[value])
            z = int(z_of[value]) if z_of[value] >= 0 else 0
            props = by_value.get(value, {})
            class_index = int(props.get("class-index", -1))
            if not -1 <= class_index < max(len(label_names), 1):
                class_index = -1
            rois.append(
                RoiRecord(
                    z=int(props.get("z", z)),
                    area=int(props.get("area", area)),
                    class_index=class_index,
                    note=str(props.get("note", "")),
                )
            )
        nz, ny, nx = volume.shape
        return RoiLabelStore(
            nz, ny, nx, label_names=label_names, labels=volume, rois=rois
        )
