import os
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

from qim3d.utils._logger import log

from .helpers import (
    SOURCE_FNM,
    NotInstalledError,
    get_volume_explorer_dir,
    get_node_binaries_dir,
    get_nvm_dir,
    get_viewer_binaries,
    get_viewer_dir,
    run_for_platform,
)
from .installation import Installer

# CLI entry point provided by @qim3d/volume-explorer
START_COMMAND = 'volume-explorer --no-open'
DEFAULT_VIEWER_PORT = 4173
DEFAULT_FILE_SERVER_PORT = 8042

# Lock, so two threads can safely read and write to is_installed
c = threading.Condition()
is_installed = True


def run_global(port: int = DEFAULT_VIEWER_PORT):
    linux_func = lambda: subprocess.run(
        f"{START_COMMAND} -p {port}", shell=True, stderr=subprocess.DEVNULL
    )

    # First sourcing the node.js, if sourcing via fnm doesnt help and user would have to do it any other way, it would throw an error and suggest to install viewer to qim library
    windows_func = lambda: subprocess.run(
        ['powershell.exe', SOURCE_FNM, f"{START_COMMAND} -p {port}"],
        shell=True,
        stderr=subprocess.DEVNULL,
    )

    run_for_platform(
        linux_func=linux_func, windows_func=windows_func, macos_func=linux_func
    )


def run_within_qim_dir(port: int = DEFAULT_VIEWER_PORT):
    base_dir = get_volume_explorer_dir()
    viewer_dir = get_viewer_dir(base_dir)
    viewer_bin = get_viewer_binaries(viewer_dir)

    def linux_func():
        # Looks for node binaries installed in qim3d/viz/volume_explorer/.nvm
        node_bin = get_node_binaries_dir(get_nvm_dir(base_dir))
        if node_bin is None:
            # Didn't find node binaries there so it looks for environment variable to tell it where is nvm folder
            node_bin = get_node_binaries_dir(Path(str(os.getenv('NVM_DIR'))))

        if node_bin is not None:
            subprocess.run(
                f'export PATH="$PATH:{viewer_bin}:{node_bin}" && {START_COMMAND} -p {port}',
                shell=True,
                stderr=subprocess.DEVNULL,
            )

    def windows_func():
        node_bin = get_node_binaries_dir(base_dir)
        if node_bin is not None:
            subprocess.run(
                [
                    'powershell.exe',
                    f"$env:PATH = $env:PATH + ';{viewer_bin};{node_bin}';",
                    f"{START_COMMAND} -p {port}",
                ],
                stderr=subprocess.DEVNULL,
            )

    run_for_platform(
        linux_func=linux_func, windows_func=windows_func, macos_func=linux_func
    )


def try_opening_volume_explorer(
    filename: str | None = None,
    open_browser: bool = True,
    file_server_port: int = DEFAULT_FILE_SERVER_PORT,
    viewer_port: int = DEFAULT_VIEWER_PORT,
):
    """
    Opens a visualization window using the Volume Explorer web app. Works both for common file types (Tiff, Nifti, etc.) and for OME-Zarr stores.

    The function starts the `volume-explorer` CLI (preferring a global install, falling back to the bundled copy inside qim3d) and launches a local HTTP server to serve the selected dataset. Optionally, it opens the default browser automatically. If the viewer binary cannot be found, a NotInstalledError is raised.
    """

    global is_installed
    c.acquire()
    is_installed = True
    c.release()

    # We do a delay open for the browser, just so that the volume-explorer has time to start.
    def delayed_open():
        time.sleep(3)
        global is_installed
        c.acquire()
        if is_installed:
            filename_norm = os.path.normpath(os.path.abspath(filename))

            # Start the http server
            qim3d.utils.start_http_server(
                os.path.dirname(filename_norm), port=file_server_port
            )

            viz_url = f'http://localhost:{viewer_port}/?src=http://localhost:{file_server_port}/{os.path.basename(filename_norm)}'

            if open_browser:
                webbrowser.open_new_tab(viz_url)

            log.info(f'\nVisualization url:\n{viz_url}\n')
        c.release()

    delayed_window = threading.Thread(target=delayed_open)
    delayed_window.start()

    # First try if the user has it globally
    run_global(port=viewer_port)

    # Then try to also find node.js installed in qim package
    run_within_qim_dir(port=viewer_port)

    # If we got to this part, it means that the viewer is not installed and we don't want to
    # open browser with non-working window
    c.acquire()
    is_installed = False
    c.release()

    delayed_window.join()

    # If we still get an error, it is not installed in location we expect it to be installed and have to raise an error
    # which will be caught in the command line and it will ask for installation
    raise NotInstalledError


def volume_explorer(
    filename: str | None = None,
    open_browser: bool = True,
    file_server_port: int = DEFAULT_FILE_SERVER_PORT,
    viewer_port: int = DEFAULT_VIEWER_PORT,
):
    """
    Launch the Volume Explorer web viewer for a given dataset path.

    Starts a local file server and opens a dedicated visualization window in your default web browser. This function is particularly effective for viewing OME-Zarr stores and other large datasets that benefit from on-demand loading. If the viewer is not found, it prompts to handle the installation automatically.
    """

    try:
        try_opening_volume_explorer(
            filename,
            open_browser=open_browser,
            file_server_port=file_server_port,
            viewer_port=viewer_port,
        )

    except NotInstalledError:
        message = (
            "Volume Explorer is not installed or qim3d cannot find it.\n"
            "You can either:\n\to  Use 'qim3d viz SOURCE -m k3d' to display data using a different method\n"
            "\to  Install volume-explorer yourself (e.g. 'npm install -g @qim3d/volume-explorer' or 'npx volume-explorer')\n"
            "\to  Let qim3d install volume-explorer now (it will also install node.js in qim3d library)\n"
            "Do you want qim3d to install volume-explorer now?"
        )
        print(message)
        answer = input('[Y/n]:')
        if answer in 'Yy':
            Installer().install()
            try_opening_volume_explorer(
                filename,
                open_browser=open_browser,
                file_server_port=file_server_port,
                viewer_port=viewer_port,
            )


# Backwards compatibility for older API callers
itk_vtk = volume_explorer
