"""Tests for the GUI-free annotation model + NGFF labels zarr persistence.

``mbo_utilities.annotation`` is the half of the manual ROI tool that has no
GUI imports (see its README); these tests need no canvas.
"""

from __future__ import annotations

import numpy as np
import pytest

from mbo_utilities.annotation import (
    UNLABELED,
    LabelsZarr,
    RoiLabelStore,
    RoiRecord,
    class_color,
)


def disk(ny, nx, cy, cx, r):
    yy, xx = np.mgrid[:ny, :nx]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


@pytest.fixture
def store():
    return RoiLabelStore(3, 32, 32, min_pixels=4)


class TestStore:
    def test_add_roi_claims_pixels_on_its_plane(self, store):
        index = store.add_roi(1, disk(32, 32, 10, 10, 3))
        assert index == 0
        assert store.rois[0].z == 1
        assert store.labels[1, 10, 10] == 1
        assert store.labels[0].max() == 0 and store.labels[2].max() == 0
        assert store.counts == [store.rois[0].area]

    def test_overlap_keeps_only_free_pixels(self, store):
        store.add_roi(0, disk(32, 32, 10, 10, 4))
        first_area = store.rois[0].area
        store.add_roi(0, disk(32, 32, 13, 10, 4))
        assert store.rois[1].area < first_area
        # the first ROI kept every one of its pixels
        assert int((store.labels[0] == 1).sum()) == first_area

    def test_same_pixels_on_other_plane_are_free(self, store):
        store.add_roi(0, disk(32, 32, 10, 10, 4))
        index = store.add_roi(1, disk(32, 32, 10, 10, 4))
        assert store.rois[index].area == store.rois[0].area

    def test_too_small_returns_none_and_leaves_volume(self, store):
        assert store.add_roi(0, np.zeros((32, 32), bool)) is None
        one_px = np.zeros((32, 32), bool)
        one_px[5, 5] = True
        assert store.add_roi(0, one_px) is None
        assert store.labels.max() == 0
        assert store.rois == []

    def test_delete_renumbers(self, store):
        for i in range(3):
            store.add_roi(i, disk(32, 32, 8 + 5 * i, 8, 3))
        assert store.delete_roi(0)
        assert len(store.rois) == 2
        assert set(np.unique(store.labels)) == {0, 1, 2}
        # the old ROI 1 (on plane 1) is now label 1
        assert store.labels[1].max() == 1

    def test_delete_out_of_range(self, store):
        assert not store.delete_roi(0)
        store.add_roi(0, disk(32, 32, 10, 10, 3))
        assert not store.delete_roi(-1)
        assert not store.delete_roi(5)
        assert len(store.rois) == 1

    def test_clear(self, store):
        store.add_roi(0, disk(32, 32, 10, 10, 3))
        store.clear()
        assert store.rois == [] and store.labels.max() == 0

    def test_dirty_plane_tracking(self, store):
        assert store.dirty_planes == set()
        store.add_roi(2, disk(32, 32, 10, 10, 3))
        assert store.dirty_planes == {2}
        store.dirty_planes.clear()
        store.add_roi(0, disk(32, 32, 20, 20, 3))
        store.delete_roi(0)  # renumbers the plane-0 ROI too
        assert store.dirty_planes == {0, 2}
        store.dirty_planes.clear()
        store.clear()
        assert store.dirty_planes == {0, 1, 2}

    def test_roi_at(self, store):
        store.add_roi(1, disk(32, 32, 10, 10, 3))
        assert store.roi_at(1, 10, 10) == 0
        assert store.roi_at(0, 10, 10) == -1
        assert store.roi_at(1, 0, 0) == -1
        assert store.roi_at(9, 10, 10) == -1

    def test_label_names_and_classes(self, store):
        assert store.add_label_name("soma") == 0
        assert store.add_label_name("dendrite") == 1
        assert store.add_label_name("soma") == 0  # no duplicate
        store.add_roi(0, disk(32, 32, 10, 10, 3))
        store.add_roi(0, disk(32, 32, 20, 20, 3))
        store.set_class(0, 1)
        assert store.class_counts() == [0, 1]
        store.set_class(0, UNLABELED)
        assert store.class_counts() == [0, 0]
        with pytest.raises(IndexError):
            store.set_class(0, 2)
        with pytest.raises(ValueError):
            store.add_label_name("  ")

    def test_add_roi_assigns_uids_and_source(self, store):
        assert store.next_uid == 1
        store.add_roi(0, disk(32, 32, 10, 10, 3))
        store.add_roi(1, disk(32, 32, 20, 20, 3), source="rois_a:2")
        assert [r.uid for r in store.rois] == [1, 2]
        assert [r.source for r in store.rois] == ["", "rois_a:2"]
        assert store.next_uid == 3
        assert store.uid_index(2) == 1
        assert store.uid_index(99) is None

    def test_delete_never_reuses_uids(self, store):
        for i in range(3):
            store.add_roi(0, disk(32, 32, 8 + 5 * i, 8, 3))
        store.delete_roi(2)  # holds uid 3, the max so far
        assert store.next_uid == 4
        index = store.add_roi(0, disk(32, 32, 8, 24, 3))
        assert store.rois[index].uid == 4
        assert store.uid_index(3) is None

    def test_adopted_records_get_fresh_uids(self):
        labels = np.zeros((1, 8, 8), np.uint16)
        labels[0, 0, :3] = (1, 2, 3)
        rois = [
            RoiRecord(z=0, area=1, uid=5),
            RoiRecord(z=0, area=1),
            RoiRecord(z=0, area=1, uid=5),
        ]
        store = RoiLabelStore(1, 8, 8, labels=labels, rois=rois)
        assert [r.uid for r in store.rois] == [5, 6, 7]
        assert store.next_uid == 8

    def test_snapshot_isolation(self, store):
        store.add_label_name("soma")
        store.add_roi(0, disk(32, 32, 10, 10, 3))
        snap = store.snapshot()
        store.add_roi(1, disk(32, 32, 20, 20, 3))
        store.set_class(0, 0)
        store.set_note(0, "changed")
        assert len(snap.rois) == 1
        assert snap.rois[0].class_index == UNLABELED
        assert snap.rois[0].note == ""
        assert snap.labels[1].max() == 0
        assert snap.next_uid == 2 and store.next_uid == 3
        assert snap.label_names == ("soma",) and snap.min_pixels == 4
        snap.add_roi(2, disk(32, 32, 5, 5, 2))
        assert store.labels[2].max() == 0

    def test_colors_follow_class(self, store):
        store.add_label_name("soma")
        store.add_roi(0, disk(32, 32, 10, 10, 3))
        hue = store.roi_rgb(0)
        store.set_class(0, 0)
        expected = tuple(int(round(c * 255)) for c in class_color(0))
        assert store.roi_rgb(0) == expected
        assert store.roi_rgb(0) != hue
        lut = store.color_lut()
        assert tuple(lut[1]) == expected and tuple(lut[0]) == (0, 0, 0)


class TestLabelsZarr:
    def _populated(self):
        store = RoiLabelStore(3, 32, 32, label_names=("soma", "dendrite"))
        store.add_roi(0, disk(32, 32, 10, 10, 4))
        store.add_roi(2, disk(32, 32, 20, 20, 3))
        store.set_class(0, 1)
        store.set_note(1, "faint, recheck")
        return store

    def test_round_trip(self, tmp_path):
        store = self._populated()
        path = tmp_path / "manual_labels.zarr"
        LabelsZarr(path).save(store, source_path="/data/movie.tif")
        restored = LabelsZarr.load(path)
        assert np.array_equal(restored.labels, store.labels)
        assert restored.label_names == ("soma", "dendrite")
        assert [r.z for r in restored.rois] == [0, 2]
        assert [r.area for r in restored.rois] == [r.area for r in store.rois]
        assert restored.rois[0].class_index == 1
        assert restored.rois[1].class_index == UNLABELED
        assert restored.rois[1].note == "faint, recheck"

    def test_save_clears_dirty_and_save_dirty_is_incremental(self, tmp_path):
        store = self._populated()
        path = tmp_path / "manual_labels.zarr"
        writer = LabelsZarr(path)
        writer.save(store)
        assert store.dirty_planes == set()
        store.add_roi(1, disk(32, 32, 16, 16, 3))
        store.set_note(2, "new one")
        writer.save_dirty(store)
        assert store.dirty_planes == set()
        restored = LabelsZarr.load(path)
        assert np.array_equal(restored.labels, store.labels)
        assert restored.rois[2].note == "new one"

    def test_uid_source_round_trip(self, tmp_path):
        store = self._populated()
        store.add_roi(1, disk(32, 32, 16, 16, 3), source="rois_a:4")
        path = tmp_path / "manual_labels.zarr"
        LabelsZarr(path).save(store)
        restored = LabelsZarr.load(path)
        assert [r.uid for r in restored.rois] == [1, 2, 3]
        assert [r.source for r in restored.rois] == ["", "", "rois_a:4"]
        assert restored.next_uid == store.next_uid == 4

    def test_next_uid_survives_deleting_the_max_uid_roi(self, tmp_path):
        store = self._populated()  # uids 1, 2; next_uid 3
        store.delete_roi(1)
        path = tmp_path / "manual_labels.zarr"
        LabelsZarr(path).save(store)
        restored = LabelsZarr.load(path)
        assert restored.next_uid == 3
        index = restored.add_roi(1, disk(32, 32, 16, 16, 3))
        assert restored.rois[index].uid == 3  # uid 2 is never reused

    def test_legacy_zarr_loads_with_fresh_uids(self, tmp_path):
        import zarr

        store = self._populated()
        path = tmp_path / "manual_labels.zarr"
        LabelsZarr(path).save(store)
        # strip the uid-era attrs to mimic a zarr from before they existed
        root = zarr.open_group(str(path), mode="a")
        mbo = dict(root.attrs["mbo"])
        del mbo["next_uid"]
        root.attrs["mbo"] = mbo
        arr = root["0"]
        image_label = dict(arr.attrs["image-label"])
        image_label["properties"] = [
            {k: v for k, v in p.items() if k not in ("uid", "source")}
            for p in image_label["properties"]
        ]
        arr.attrs["image-label"] = image_label
        restored = LabelsZarr.load(path)
        assert [r.uid for r in restored.rois] == [1, 2]
        assert all(r.source == "" for r in restored.rois)
        assert restored.next_uid == 3

    def test_ngff_layout(self, tmp_path):
        import zarr

        store = self._populated()
        path = tmp_path / "manual_labels.zarr"
        LabelsZarr(path).save(store, source_path="/data/movie.tif")
        root = zarr.open_group(str(path), mode="r")
        assert root.attrs["version"] == "0.5"
        assert root.attrs["labels"] == ["0"]
        assert root.attrs["mbo"]["label_names"] == ["soma", "dendrite"]
        assert root.attrs["mbo"]["source_image"].endswith("movie.tif")
        arr = root["0"]
        assert arr.dtype == np.uint32 and arr.shape == (3, 32, 32)
        image_label = arr.attrs["image-label"]
        assert len(image_label["colors"]) == 2
        assert len(image_label["properties"]) == 2
        first = image_label["properties"][0]
        assert first["label-value"] == 1 and first["class"] == "dendrite"

    def test_foreign_zarr_without_properties_loads(self, tmp_path):
        import zarr

        path = tmp_path / "labels.zarr"
        volume = np.zeros((2, 16, 16), np.uint32)
        volume[0, 2:6, 2:6] = 1
        volume[1, 8:12, 8:12] = 2
        root = zarr.open_group(str(path), mode="w")
        arr = zarr.create(
            store=root.store,
            path="0",
            shape=volume.shape,
            dtype=np.uint32,
            overwrite=True,
        )
        arr[:] = volume
        root.attrs.update({"version": "0.5", "labels": ["0"]})
        restored = LabelsZarr.load(path)
        assert np.array_equal(restored.labels, volume)
        assert [r.z for r in restored.rois] == [0, 1]
        assert [r.area for r in restored.rois] == [16, 16]
        assert all(r.class_index == UNLABELED for r in restored.rois)

    def test_load_errors(self, tmp_path):
        import zarr

        with pytest.raises(FileNotFoundError):
            LabelsZarr.load(tmp_path / "missing.zarr")
        path = tmp_path / "flat.zarr"
        root = zarr.open_group(str(path), mode="w")
        arr = zarr.create(
            store=root.store,
            path="0",
            shape=(4, 4),
            dtype=np.uint32,
            overwrite=True,
        )
        arr[:] = 0
        with pytest.raises(ValueError):
            LabelsZarr.load(path)
