"""Staging transfers, against an in-memory stand-in for the storage client."""

from __future__ import annotations

import shutil
from pathlib import Path

from imgui_cloud import gcs


class FakeBlob:
    """A blob backed by a file on disk under the fake bucket's root."""

    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name

    @property
    def path_backing(self) -> Path:
        return self.bucket.root / self.name

    @property
    def size(self) -> int:
        return self.path_backing.stat().st_size

    def upload_from_filename(self, filepath):
        self.path_backing.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filepath, self.path_backing)

    def upload_from_string(self, text):
        self.path_backing.parent.mkdir(parents=True, exist_ok=True)
        self.path_backing.write_text(text)

    def download_to_filename(self, filepath):
        shutil.copyfile(self.path_backing, filepath)

    def download_as_text(self) -> str:
        return self.path_backing.read_text()

    def delete(self):
        self.path_backing.unlink()


class FakeBucket:
    def __init__(self, root):
        self.root = Path(root)

    def blob(self, name):
        return FakeBlob(self, name)

    def get_blob(self, name):
        blob = FakeBlob(self, name)
        return blob if blob.path_backing.exists() else None


class FakeClient:
    """Just enough google.cloud.storage.Client for :mod:`imgui_cloud.gcs`."""

    def __init__(self, root):
        self._bucket = FakeBucket(root)

    def bucket(self, name):
        return self._bucket

    def list_blobs(self, bucket, prefix=""):
        root = self._bucket.root
        return [
            FakeBlob(self._bucket, path.relative_to(root).as_posix())
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.relative_to(root).as_posix().startswith(prefix)
        ]


def make_tree(root: Path) -> Path:
    (root / "plane_1").mkdir(parents=True)
    (root / "plane_1" / "data.tif").write_bytes(b"0123456789")
    (root / "notes.txt").write_text("hello")
    return root


def test_upload_then_download_reproduces_the_tree(tmp_path):
    source = make_tree(tmp_path / "input")
    client = FakeClient(tmp_path / "bucket")

    progress = gcs.upload_tree(client, "bkt", "pre/run/input", str(source))
    assert progress.files_done == 2
    assert progress.bytes_done == 15
    assert progress.fraction == 1.0

    destination = tmp_path / "back"
    result = gcs.download_tree(client, "bkt", "pre/run/input", str(destination))
    assert result.files_done == 2
    assert (destination / "plane_1" / "data.tif").read_bytes() == b"0123456789"
    assert (destination / "notes.txt").read_text() == "hello"


def test_patterns_filter_by_relative_path_and_by_name(tmp_path):
    source = make_tree(tmp_path / "input")
    assert [p.name for p in gcs.matching_files(str(source), ["*.tif"])] == ["data.tif"]
    assert [p.name for p in gcs.matching_files(str(source), ["plane_1/*"])] == [
        "data.tif"
    ]
    assert len(gcs.matching_files(str(source))) == 2


def test_uploading_a_single_file_keeps_its_name(tmp_path):
    filepath = tmp_path / "one.tif"
    filepath.write_bytes(b"x")
    client = FakeClient(tmp_path / "bucket")
    gcs.upload_tree(client, "bkt", "pre/run/input", str(filepath))
    assert (tmp_path / "bucket" / "pre" / "run" / "input" / "one.tif").exists()


def test_progress_callback_sees_monotonic_growth(tmp_path):
    source = make_tree(tmp_path / "input")
    client = FakeClient(tmp_path / "bucket")
    seen = []
    gcs.upload_tree(
        client,
        "bkt",
        "pre",
        str(source),
        on_progress=lambda p: seen.append(p.bytes_done),
    )
    assert seen == sorted(seen)
    assert seen[-1] == 15


def test_text_helpers_and_existence(tmp_path):
    client = FakeClient(tmp_path / "bucket")
    assert gcs.read_text(client, "bkt", "pre/state.txt") is None
    assert gcs.exists(client, "bkt", "pre/state.txt") is False
    gcs.upload_text(client, "bkt", "pre/state.txt", gcs.STATE_RUNNING)
    assert gcs.read_text(client, "bkt", "pre/state.txt") == "RUNNING"
    assert gcs.exists(client, "bkt", "pre/state.txt") is True


def test_delete_prefix_removes_only_that_prefix(tmp_path):
    source = make_tree(tmp_path / "input")
    client = FakeClient(tmp_path / "bucket")
    gcs.upload_tree(client, "bkt", "runs/a", str(source))
    gcs.upload_tree(client, "bkt", "runs/b", str(source))
    assert gcs.delete_prefix(client, "bkt", "runs/a") == 2
    assert gcs.total_size(client, "bkt", "runs/b") == 15


def test_missing_input_directory_fails_loudly(tmp_path):
    try:
        gcs.matching_files(str(tmp_path / "nope"))
    except FileNotFoundError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")
