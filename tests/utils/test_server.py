import os
import socket
import urllib.error
import urllib.request

import qim3d


def _free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _get(port: int, url_path: str) -> tuple[int, bytes]:
    """GET a url path from the local server, returning (status, body)."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}{url_path}") as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _make_tree(tmp_path):
    """Create a tree that mimics the viz use cases:

    a single tif file, a sibling file that should never be visible, and an
    OME-Zarr store (directory with subfolders and a hidden file).
    """
    (tmp_path / "data.tif").write_bytes(b"tiff-data")
    (tmp_path / "secret.txt").write_text("top secret")
    store = tmp_path / "store.zarr"
    (store / "0" / "0").mkdir(parents=True)
    (store / "zarr.json").write_text("{}")
    (store / ".zattrs").write_text("{}")
    (store / "0" / "0" / "chunk.npy").write_bytes(b"chunk")
    return tmp_path


def test_single_file_only_is_served(tmp_path):
    """Only the requested file may be served; siblings and listings may not."""
    root = _make_tree(tmp_path)
    port = _free_port()
    server = qim3d.utils.start_http_server(
        str(root), port=port, allowed_paths=["data.tif"]
    )
    try:
        # The requested file is served
        status, body = _get(port, "/data.tif")
        assert status == 200
        assert body == b"tiff-data"

        # A sibling file is not
        status, _ = _get(port, "/secret.txt")
        assert status == 404

        # The directory listing is not
        status, _ = _get(port, "/")
        assert status == 404

        # A file whose name starts with the allowed one is not
        (root / "data.tiff").write_bytes(b"other")
        status, _ = _get(port, "/data.tiff")
        assert status == 404

        # Path traversal is not
        status, _ = _get(port, "/../secret.txt")
        assert status == 404
        status, _ = _get(port, "/%2e%2e/secret.txt")
        assert status == 404
        status, _ = _get(port, "/data.tif/../../secret.txt")
        assert status == 404
    finally:
        server.shutdown()


def test_ome_zarr_directory_is_served(tmp_path):
    """A directory (OME-Zarr store) is served together with its whole
    subtree, including hidden files, but not its siblings."""
    root = _make_tree(tmp_path)
    port = _free_port()
    server = qim3d.utils.start_http_server(
        str(root), port=port, allowed_paths=["store.zarr"]
    )
    try:
        # Files inside the store (including in subfolders) are served
        status, body = _get(port, "/store.zarr/zarr.json")
        assert status == 200
        assert body == b"{}"
        status, _ = _get(port, "/store.zarr/0/0/chunk.npy")
        assert status == 200

        # Hidden files inside the store are served (OME-Zarr needs .zattrs)
        status, _ = _get(port, "/store.zarr/.zattrs")
        assert status == 200

        # The store directory listing is served
        status, _ = _get(port, "/store.zarr/")
        assert status == 200

        # Siblings are not
        status, _ = _get(port, "/data.tif")
        assert status == 404
        status, _ = _get(port, "/secret.txt")
        assert status == 404

        # The root listing is not
        status, _ = _get(port, "/")
        assert status == 404

        # A sibling directory with a similar name is not
        (root / "store.zarr.bak").mkdir()
        (root / "store.zarr.bak" / "leak.txt").write_text("nope")
        status, _ = _get(port, "/store.zarr.bak/leak.txt")
        assert status == 404
    finally:
        server.shutdown()


def test_no_restriction_serves_whole_directory(tmp_path):
    """Backwards compatible behavior: with allowed_paths=None the whole
    directory is served."""
    root = _make_tree(tmp_path)
    port = _free_port()
    server = qim3d.utils.start_http_server(str(root), port=port)
    try:
        status, _ = _get(port, "/")
        assert status == 200
        status, _ = _get(port, "/secret.txt")
        assert status == 200
        status, _ = _get(port, "/data.tif")
        assert status == 200
    finally:
        server.shutdown()


def test_server_does_not_change_cwd(tmp_path):
    """The server binds to the directory without changing the process CWD."""
    root = _make_tree(tmp_path)
    port = _free_port()
    cwd_before = os.getcwd()
    server = qim3d.utils.start_http_server(str(root), port=port)
    try:
        assert os.getcwd() == cwd_before
        status, _ = _get(port, "/data.tif")
        assert status == 200
    finally:
        server.shutdown()
