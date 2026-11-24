import logging
import os
import threading
from collections.abc import Sequence
from http.server import HTTPServer, SimpleHTTPRequestHandler

_logger = logging.getLogger(__name__)


class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    """A `SimpleHTTPRequestHandler` with CORS headers and an optional
    allow-list of served paths.

    When the ``allowed_paths`` argument (a set of relative paths, with
    ``/`` as separator) is provided, a request is only served if its
    resolved path equals one of the allowed paths, or lies underneath one
    of them (this is what makes it possible to serve an entire directory
    tree, such as an OME-Zarr store, while still hiding everything else).
    Any other request is answered with a 404. When ``allowed_paths`` is
    None, everything under the server root is served, as in the standard
    behavior.
    """

    def __init__(self, *args, allowed_paths: set[str] | None = None, **kwargs):
        # Note: must be set before super().__init__, because it calls
        # handle(), which in turn reads allowed_paths in send_head
        self.allowed_paths = allowed_paths
        super().__init__(*args, **kwargs)

    def end_headers(self):
        """Add CORS headers to each response."""
        # Allow requests from any origin, or restrict to specific domains by specifying the origin
        self.send_header("Access-Control-Allow-Origin", "*")
        # Allow specific methods
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        # Allow specific headers (if needed)
        self.send_header(
            "Access-Control-Allow-Headers", "X-Requested-With, Content-Type"
        )
        super().end_headers()

    def list_directory(self, path: str | os.PathLike):
        """Helper to produce a directory listing, includes hidden files."""
        try:
            file_list = os.listdir(path)
        except OSError:
            self.send_error(404, "No permission to list directory")
            return None

        # Sort the file list
        file_list.sort(key=lambda a: a.lower())

        # Format the list with hidden files included
        displaypath = os.path.basename(path)
        r = ['<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">']
        r.append(f"<html>\n<title>Directory listing for {displaypath}</title>\n")
        r.append(f"<body>\n<h2>Directory listing for {displaypath}</h2>\n")
        r.append("<hr>\n<ul>")
        for name in file_list:
            fullname = os.path.join(path, name)
            displayname = linkname = name

            # Append the files and directories to the HTML list
            if os.path.isdir(fullname):
                displayname = name + "/"
                linkname = name + "/"
            r.append(f'<li><a href="{linkname}">{displayname}</a></li>')
        r.append("</ul>\n<hr>\n</body>\n</html>\n")
        encoded = "\n".join(r).encode("utf-8", "surrogateescape")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        # Write the encoded HTML directly to the response
        self.wfile.write(encoded)

    def send_head(self):
        """Enforce the allowed-paths restriction before serving anything."""
        if self.allowed_paths is not None:
            try:
                path = self.translate_path(self.path)
                rel = os.path.relpath(path, self.directory)
            except Exception:
                # Fail closed: anything we cannot resolve is denied
                rel = None
            if rel is not None and any(
                rel == allowed or rel.startswith(allowed + os.sep)
                for allowed in self.allowed_paths
            ):
                return super().send_head()
            self.send_error(404, "Not found")
            return None
        return super().send_head()


def start_http_server(
    directory: str,
    port: int = 8000,
    allowed_paths: str | Sequence[str] | None = None,
) -> HTTPServer:
    """
    Starts an HTTP server serving the specified directory on the given port with CORS enabled.

    Parameters
    directory (str): The directory to serve as the server root.
    port (int): The port number to use (default is 8000).
    allowed_paths (str | Sequence[str] | None): Optionally restricts which
        resources under the root directory can be requested. Only these
        paths, relative to ``directory`` (single file paths, or a directory
        together with its entire subtree), are served; any other request is
        answered with a 404. If None (default), the whole directory is
        served.

    """
    directory = os.path.abspath(directory)

    if allowed_paths is None:
        allowed: set[str] | None = None
    else:
        if isinstance(allowed_paths, str):
            allowed_paths = [allowed_paths]
        allowed = {p.replace(os.sep, "/") for p in allowed_paths}

    # Bind the directory and the allow-list to the handler instances of this
    # specific server, without changing the working directory of the process
    class _Handler(CustomHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args, directory=directory, allowed_paths=allowed, **kwargs
            )

    # Create the server
    server = HTTPServer(("", port), _Handler)

    # Run the server in a separate thread so it doesn't block execution
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    if allowed is not None:
        what = "'{}'".format(", ".join(sorted(allowed)))
    else:
        what = "the whole directory"
    _logger.info(f"Serving {what} from '{directory}'\nhttp://localhost:{port}/")

    return server
