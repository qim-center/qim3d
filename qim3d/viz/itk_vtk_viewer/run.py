import subprocess
import platform
from pathlib import Path
import os
import qim3d.utils
from qim3d.utils.logger import log

# from .helpers import get_qim_dir, get_nvm_dir, get_viewer_binaries, get_viewer_dir, get_node_binaries_dir, NotInstalledError, SOURCE_FNM
from .helpers import *
import webbrowser
import threading
import time


# Start viewer
START_COMMAND = "itk-vtk-viewer -s"

# Lock, so two threads can safely read and write to is_installed
c = threading.Condition()
is_installed = True


def run_global(port=3000):
    linux_func = lambda: subprocess.run(
        START_COMMAND+f" -p {port}", shell=True, stderr=subprocess.DEVNULL
    )

    # First sourcing the node.js, if sourcing via fnm doesnt help and user would have to do it any other way, it would throw an error and suggest to install viewer to qim library
    windows_func = lambda: subprocess.run(
        ["powershell.exe", SOURCE_FNM, START_COMMAND+f" -p {port}"],
        shell=True,
        stderr=subprocess.DEVNULL,
    )

    run_for_platform(
        linux_func=linux_func, windows_func=windows_func, macos_func=linux_func
    )


def run_within_qim_dir(port=3000):  
    dir = get_itk_dir()
    viewer_dir = get_viewer_dir(dir)
    viewer_bin = get_viewer_binaries(viewer_dir)

    def linux_func():
        # Looks for node binaries installed in qim3d/viz/itk_vtk_viewer/.nvm
        node_bin = get_node_binaries_dir(get_nvm_dir(dir))
        if node_bin is None:
            # Didn't find node binaries there so it looks for enviroment variable to tell it where is nvm folder
            node_bin = get_node_binaries_dir(Path(str(os.getenv("NVM_DIR"))))

        if node_bin is not None:
            subprocess.run(
                f'export PATH="$PATH:{viewer_bin}:{node_bin}" && {START_COMMAND+f" -p {port}"}',
                shell=True,
                stderr=subprocess.DEVNULL,
            )

    def windows_func():
        node_bin = get_node_binaries_dir(dir)
        if node_bin is not None:
            subprocess.run(
                [
                    "powershell.exe",
                    f"$env:PATH = $env:PATH + ';{viewer_bin};{node_bin}';",
                    START_COMMAND+f" -p {port}",
                ],
                stderr=subprocess.DEVNULL,
            )

    run_for_platform(
        linux_func=linux_func, windows_func=windows_func, macos_func=linux_func
    )


def itk_vtk(
    filename: str = None,
    open_browser: bool = True,
    file_server_port: int = 8042,
    viewer_port: int = 3000,
):
    """
    Opens a visualization window using the itk-vtk-viewer. Works both for common file types (Tiff, Nifti, etc.) and for **OME-Zarr stores**.

    This function starts the itk-vtk-viewer, either using a global
    installation or a local installation within the QIM package. It also starts
    an HTTP server to serve the file to the viewer. Optionally, it can
    automatically open a browser window to display the viewer. If the viewer
    is not installed, it raises a NotInstalledError.

    Args:
        filename (str, optional): Path to the file or OME-Zarr store to be visualized. Trailing slashes in
            the path are normalized. Defaults to None.
        open_browser (bool, optional): If True, opens the visualization in a new browser tab.
            Defaults to True.
        file_server_port (int, optional): The port number for the local file server that hosts
            the store. Defaults to 8042.
        viewer_port (int, optional): The port number for the itk-vtk-viewer server. Defaults to 3000.

    Raises:
        NotInstalledError: Raised if the itk-vtk-viewer is not installed in the expected location.

    Example:
        ```python
        import qim3d

        # Download data
        downloader = qim3d.io.Downloader()
        data = downloader.Okinawa_Forams.Okinawa_Foram_1(load_file=True, virtual_stack=True)

        # Export to OME-Zarr
        qim3d.io.export_ome_zarr("Okinawa_Foram_1.zarr", data)

        # Start visualization
        qim3d.viz.itk_vtk("Okinawa_Foram_1.zarr")
        ```
        <pre style="margin-left: 12px; margin-right: 12px; color:#454545">
        Downloading Okinawa_Foram_1.tif
        https://archive.compute.dtu.dk/download/public/projects/viscomp_data_repository/Okinawa_Forams/Okinawa_Foram_1.tif
        1.85GB [00:17, 111MB/s]                                                         

        Loading Okinawa_Foram_1.tif
        Loading: 100%
         1.85GB/1.85GB  [00:02<00:00, 762MB/s]
        Loaded shape: (995, 1014, 984)
        Using virtual stack
        Exporting data to OME-Zarr format at Okinawa_Foram_1.zarr
        Number of scales: 5
        Creating a multi-scale pyramid
        - Scale 0: (995, 1014, 984)
        - Scale 1: (498, 507, 492)
        - Scale 2: (249, 254, 246)
        - Scale 3: (124, 127, 123)
        - Scale 4: (62, 63, 62)
        Writing data to disk
        All done!

        itk-vtk-viewer
        => Serving /home/fima/Notebooks/Qim3d on port 3000

            enp0s31f6 => http://10.52.0.158:3000/
            wlp0s20f3 => http://10.197.104.229:3000/

        Serving directory '/home/fima/Notebooks/Qim3d'
        http://localhost:8042/

        Visualization url:
        http://localhost:3000/?rotate=false&fileToLoad=http://localhost:8042/Okinawa_Foram_1.zarr

        </pre>
                
        ![itk-vtk-viewer](assets/screenshots/itk-vtk-viewer.gif)

    """

    global is_installed
    # This might seem redundant but is here in case we have to go through the installation first
    # If we have to install first this variable is set to False and doesn't disappear
    # So when we want to run the newly installed viewer it would still be false and webbrowser wouldnt open
    c.acquire()
    is_installed = True
    c.release()

    # We do a delay open for the browser, just so that the itk-vtk-viewer has time to start.
    # Timing is not critical, this is just so that the user does not see the "server cannot be reached" page
    def delayed_open():
        time.sleep(3)
        global is_installed
        c.acquire()
        if is_installed:

            # Normalize the filename. This is necessary for trailing slashes by the end of the path
            filename_norm = os.path.normpath(os.path.abspath(filename))

            # Start the http server
            qim3d.utils.start_http_server(
                os.path.dirname(filename_norm), port=file_server_port
            )

            viz_url = f"http://localhost:{viewer_port}/?rotate=false&fileToLoad=http://localhost:{file_server_port}/{os.path.basename(filename_norm)}"

            if open_browser:
                webbrowser.open_new_tab(viz_url)

            log.info(f"\nVisualization url:\n{viz_url}\n")
        c.release()

    # Start the delayed open in a separate thread
    delayed_window = threading.Thread(target=delayed_open)
    delayed_window.start()

    # First try if the user doesn't have it globally
    run_global(port=viewer_port)

    # Then try to also find node.js installed in qim package
    run_within_qim_dir(port=viewer_port)

    # If we got to this part, it means that the viewer is not installed and we don't want to
    # open browser with non-working window
    # We sat the flag is_installed to False which will be read in the other thread to let it know not to open the browser
    c.acquire()
    is_installed = False
    c.release()

    delayed_window.join()

    # If we still get an error, it is not installed in location we expect it to be installed and have to raise an error
    # which will be caught in the command line and it will ask for installation
    raise NotInstalledError
