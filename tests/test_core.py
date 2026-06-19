import os
from collections.abc import Iterator
from typing import Any
from unittest import mock

import pytest

from gdrive_fsspec.core import (
    DIR_MIME_TYPE,
    GoogleDriveFile,
    GoogleDriveFileSystem,
    _finfo_from_response,
    _normalize_path,
)

TESTDIR = "gdrive_fsspec_testdir"


def _credentials_configured():
    token = os.getenv("GDRIVE_FSSPEC_CREDENTIALS_TYPE", "service_account")
    if token == "service_account":
        path = os.getenv("GDRIVE_FSSPEC_CREDENTIALS_PATH")
        return bool(path and path.strip())
    return True


@pytest.fixture()
def fs() -> Iterator[GoogleDriveFileSystem]:
    if not _credentials_configured():
        pytest.skip("GDRIVE_FSSPEC_CREDENTIALS_PATH not set")
    kwargs = {
        "creds": os.getenv("GDRIVE_FSSPEC_CREDENTIALS_PATH"),
        "token": os.getenv("GDRIVE_FSSPEC_CREDENTIALS_TYPE", "service_account"),
        "drive": os.getenv("GDRIVE_FSSPEC_DRIVE"),
    }
    fs = GoogleDriveFileSystem(skip_instance_cache=True, **kwargs)
    if fs.exists(TESTDIR):
        fs.rm(TESTDIR, recursive=True)
    fs.mkdir(TESTDIR, create_parents=True)
    try:
        yield fs
    finally:
        try:
            fs.rm(TESTDIR, recursive=True)
        except IOError:
            pass


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix, name, expected",
    [
        ("/a/b/", "c", "/a/b/c"),
        ("a/b", "c", "/a/b/c"),
    ],
)
def test_normalize_path(prefix: str, name: str, expected: str) -> None:
    assert _normalize_path(prefix, name) == expected


@pytest.mark.parametrize(
    "mime_type, expected_type",
    [
        ("text/plain", "file"),
        (DIR_MIME_TYPE, "directory"),
    ],
)
def test_finfo_from_response_type(mime_type: str, expected_type: str) -> None:
    info = _finfo_from_response(
        {"name": "child", "mimeType": mime_type}, path_prefix="parent"
    )
    assert info["type"] == expected_type
    assert info["name"] == "parent/child"


def test_finfo_from_response_casts_size():
    assert _finfo_from_response({"name": "x", "size": "12"})["size"] == 12


def test_finfo_from_response_defaults_missing_size():
    assert _finfo_from_response({"name": "x"})["size"] == 0


def test_finfo_from_response_strips_leading_slash():
    info = _finfo_from_response({"name": "f"}, path_prefix="/top")
    assert info["name"] == "top/f"


def test_finfo_from_response_does_not_mutate_input() -> None:
    original = {"name": "x", "mimeType": "text/plain", "size": "5"}
    snapshot = dict(original)
    _finfo_from_response(original)
    assert original == snapshot


# ---------------------------------------------------------------------------
# Construction and connection (no network)
# ---------------------------------------------------------------------------


def test_create_anon(anon_fs: GoogleDriveFileSystem) -> None:
    assert anon_fs.srv is not None


def test_strip_path(anon_fs: GoogleDriveFileSystem) -> None:
    assert anon_fs._strip_path("gdrive://foo/bar") == "foo/bar"
    assert anon_fs._strip_path("foo/bar") == "foo/bar"


def test_auth_kwargs():
    fs = GoogleDriveFileSystem(
        token="anon",
        auth_kwargs={"user_email": "test@example.com"},
        skip_instance_cache=True,
    )
    assert fs.srv is not None
    assert fs.auth_kwargs == {"user_email": "test@example.com"}


def test_connect_invalid_method():
    with pytest.raises(ValueError):
        GoogleDriveFileSystem(token="bogus", skip_instance_cache=True)


def test_invalid_access_raises():
    with pytest.raises(KeyError):
        GoogleDriveFileSystem(token="anon", access="nope", skip_instance_cache=True)


@pytest.mark.parametrize(
    "access, expected_scopes",
    [
        ("full_control", ["https://www.googleapis.com/auth/drive"]),
        ("read_only", ["https://www.googleapis.com/auth/drive.readonly"]),
    ],
)
def test_access_scopes_mapping(access: str, expected_scopes: list[str]) -> None:
    fs = GoogleDriveFileSystem(token="anon", access=access, skip_instance_cache=True)
    assert fs.scopes == expected_scopes


def test_drive_kw_without_drive(anon_fs: GoogleDriveFileSystem) -> None:
    assert anon_fs._drive_kw() == {}


def test_drive_kw_with_drive(anon_fs: GoogleDriveFileSystem) -> None:
    anon_fs.drive = "drive-123"
    kw = anon_fs._drive_kw()
    assert kw["driveId"] == "drive-123"
    assert kw["supportsAllDrives"] is True


def test_root_info(anon_fs: GoogleDriveFileSystem) -> None:
    info = anon_fs.info("")
    assert info["type"] == "directory"
    assert info["id"] == anon_fs.root_file_id


def test_drive_id_from_name_single_match(anon_fs: GoogleDriveFileSystem) -> None:
    anon_fs.drives = [{"id": "1", "name": "foo"}, {"id": "2", "name": "bar"}]
    assert anon_fs._drive_id_from_name("foo") == "1"


def test_drive_id_from_name_missing(anon_fs: GoogleDriveFileSystem) -> None:
    anon_fs.drives = [{"id": "1", "name": "foo"}]
    with pytest.raises(ValueError):
        anon_fs._drive_id_from_name("missing")


def test_drive_id_from_name_duplicate(anon_fs: GoogleDriveFileSystem) -> None:
    anon_fs.drives = [{"id": "1", "name": "dup"}, {"id": "2", "name": "dup"}]
    with pytest.raises(ValueError):
        anon_fs._drive_id_from_name("dup")


@pytest.mark.parametrize(
    "creds",
    [
        {"type": "service_account"},
        '{"type": "service_account"}',
    ],
)
def test_service_account_creds_parsing(creds: str | dict[str, Any]) -> None:
    target = "gdrive_fsspec.core.service_account.Credentials.from_service_account_info"
    with mock.patch(target) as from_info:
        GoogleDriveFileSystem(
            token="service_account", creds=creds, skip_instance_cache=True
        )
    from_info.assert_called_once()
    assert from_info.call_args.kwargs["info"] == {"type": "service_account"}


@pytest.mark.parametrize("creds", ["", "   ", "\t\n"])
def test_service_account_empty_creds_raises(creds: str) -> None:
    with pytest.raises(ValueError, match="Empty credentials"):
        GoogleDriveFileSystem(
            token="service_account", creds=creds, skip_instance_cache=True
        )


# ---------------------------------------------------------------------------
# Integration (require live Google Drive credentials)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_simple(fs: GoogleDriveFileSystem) -> None:
    assert fs.ls("")
    data = b"hello"
    fn = TESTDIR + "/testfile"
    with fs._open(fn, "wb") as f:
        assert isinstance(f, GoogleDriveFile)
        f.write(data)
    assert fs.cat(fn) == data


@pytest.mark.integration
def test_create_directory(fs: GoogleDriveFileSystem) -> None:
    fs.makedirs(TESTDIR + "/data")
    fs.makedirs(TESTDIR + "/data/bar/baz")

    assert fs.exists(TESTDIR + "/data")
    assert fs.exists(TESTDIR + "/data/bar")
    assert fs.exists(TESTDIR + "/data/bar/baz")

    data = b"intermediate path"
    with fs._open(TESTDIR + "/data/bar/test", "wb") as f:
        assert isinstance(f, GoogleDriveFile)
        f.write(data)
    assert fs.cat(TESTDIR + "/data/bar/test") == data
