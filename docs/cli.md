# Command line interface
The library also includes a command line interface for easy access to some functionalities, and the convenience of using qim3d directly from your terminal. 

This offers quick interactions, making it ideal for tasks that require efficiency or when certain functionalities need to run on a server. 

!!! Example
    ``` title="Command"
    qim3d gui --data-explorer
    ```
    ![CLI Data Explorer](assets/screenshots/CLI-data_explorer.gif)


## Graphical User Interfaces

### `qim3d gui`
!!! quote "Reference"
    The GUIs available in `qim3d` are built using Gradio:
    <https://github.com/gradio-app/gradio>

    ```bibtex
    @article{abid2019gradio,
    title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
    author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
    journal = {arXiv preprint arXiv:1906.02569},
    year = {2019},
    }
    ```
| Arguments | Description |
| --------- | ----------- |
| `--data-explorer` | Starts the Data Explorer |
| `--iso3d` | Starts the 3D Isosurfaces visualization |
| `--local-thickness` | Starts the Local thickness tool |
| `--anotation-tool` | Starts the annotation tool |
| `--layers` | Starts the tool for segmenting layers |
| `--host` | Desired host for the server. By default runs on `0.0.0.0`  |
| `--platform` | Uses the Qim platform API for a unique path and port depending on the username |


!!! Example

    Here's an example of how to open the [Data Explorer](gui.md#data_explorer)

    ``` title="Command"
    qim3d gui --data-explorer
    ```
    ``` title="Output"
    Running on local URL:  http://127.0.0.1:7860
    ```

    In this case, the GUI will be available at `http://127.0.0.1:7860`

    ![Data explorer GUI](assets/screenshots/GUI-data_explorer.png)


!!! Example

    Or for the local thickness GUI:

    ``` title="Command"
    qim3d gui --local-thickness --host 127.0.0.1 --platform
    ```

    ``` title="Output"
    {'username': 'fima', 'jupyter_port': '57326', 'gradio_port': '47326'}


    ╭────────────────────────╮
    │ Starting gradio server │
    ├────────────────────────╯
    ├ Gradio
    ├ Using port 47326
    ╰ Running at 10.52.0.158

    http://127.0.0.1:47326/gui/fima/47326/
    INFO:     Started server process [1534019]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://127.0.0.1:47326 (Press CTRL+C to quit)

    ```

    In this case, the GUI will be available at `http://127.0.0.1:47326/gui/fima/47326/`

    ![Local thickness GUI](assets/screenshots/GUI-local_thickness.png)













## Data visualization
The command line interface also allows you to easily visualize data.

### `qim3d viz`
!!! quote "Reference"
    Volumetric visualization uses [itk-vtk-viewer](https://kitware.github.io/itk-vtk-viewer/docs/index.html) and [K3D](https://github.com/K3D-tools/K3D-jupyter).

You can launch volumetric visualizations directly from the command line. By default, it will launch the `itk-vtk-viewer`, but `k3d` can be selected by passing the argument `--method k3d`.

!!! info
    If `itk-vtk-viewer` is not installed, you will be prompted for automatic installation via the `qim3d` library.

| Argument         | Description     |
| ---------------- | --------------- |
| `source`         | Path to the volume file or OME-Zarr store (any image format supported by `qim3d.io.load()`). |
| `--method`       | Visualization method: `itk-vtk-viewer` (default) or `k3d`. |
| `--destination`  | Custom `.html` file to be saved when using `k3d`. By default, `k3d.html` is saved. An file is aways saved when calling `k3d`. |
| `--no-browser`   | Prevent the file from opening automatically when finished.|


!!! Example "Visualization of a OME-Zarr store"
    ``` title="Command"
    qim3d viz Okinawa_Foram_1.zarr/
    ```

    ``` title="Output"
    itk-vtk-viewer
    => Serving /home/fima/Notebooks/Qim3d on port 3000

        enp0s31f6 => http://10.52.0.158:3000/
        wlp0s20f3 => http://10.197.104.229:3000/

    Serving directory '/home/fima/Notebooks/Qim3d'
    http://localhost:8042/

    Visualization url:
    http://localhost:3000/?rotate=false&fileToLoad=http://localhost:8042/Okinawa_Foram_1.zarr
    ```

    A new tab in the default browser will be open with the visualization:

    ![itk-vtk-viewer](assets/screenshots/itk-vtk-viewer.gif)

!!! Example "Example using k3d"
    ``` title="Command"
    qim3d viz blobs_256x256x256.tif --method k3d
    ```

    ``` title="Output"
    Loading data from cement_128x128x128.tif
    Loading: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.02MB/2.02MB  [00:00<00:00, 936MB/s]
    Volume using 2.0 MB of memory

    System memory:
    • Total.: 31.0 GB
    • Used..: 18.8 GB (60.8%)
    • Free..: 12.1 GB (39.2%)
    Done, volume shape: (128, 128, 128)

    Generating k3d plot...
    Done, plot available at <k3d.html>
    Opening in default browser...

    ```
    And a new tab will be opened in the default browser with the interactive k3d plot:

    ![CLI k3d](assets/screenshots/CLI-k3d.png){ width="512" }

Or an specific path for destination can be used. We can also choose to not open the browser:

!!! Example
    ```
    qim3d viz cement_128x128x128.tif --destination my_plot.html --no-browser

    ```
    
    ``` title="Output"
    Loading data from cement_128x128x128.tif
    Loading: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.02MB/2.02MB  [00:00<00:00, 909MB/s]
    Volume using 2.0 MB of memory

    System memory:
    • Total.: 31.0 GB
    • Used..: 18.9 GB (61.1%)
    • Free..: 12.0 GB (38.9%)
    Done, volume shape: (128, 128, 128)

    Generating k3d plot...
    Done, plot available at <my_plot.html>

    ```

    This writes to disk the `my_plot.html` file.

## File preview
Command line interface, which allows users to preview 3D structers or 2D images directly in command line.
###  `qim3d preview`
| Arguments | Description |
| --------- | ----------- |
| `--axis` | Specifies from which axis the slice will be taken. If the object is 2D image, then this is ignored. Defaults to 0.|
| `--slice` | Specifies which slice will be displayed. If the number exceeds number of slices, the last one is taken. Defaults to the middle slice.|
| `--resolution` | How many characters will be used to display the image in command line. Defaults to 80.|
| `--absolute_values` |If values are low the image might be just black square. By default maximum value is set to 255. This flag turns this behaviour off.|

!!! Example
    ``` title="Command"
    qim3d preview blobs_256x256x256.tif 
    ```

    ![CLI k3d](assets/preview/default.png){ width="512" }

!!! Example
    ``` title="Command"
    qim3d preview blobs_256x256x256.tif --resolution 30
    ```

    ![CLI k3d](assets/preview/res30.png){ width="512" }

!!! Example
    ``` title="Command"
    qim3d preview blobs_256x256x256.tif --resolution 50 --axis 1
    ```

    ![CLI k3d](assets/preview/axis1.png){ width="512" }

!!! Example
    ``` title="Command"
    qim3d preview blobs_256x256x256.tif --resolution 50 --axis 2 --slice 0
    ```

    ![CLI k3d](assets/preview/relativeIntensity.png){ width="512" }

!!! Example
    ``` title="Command"
    qim3d preview qim_logo.png --resolution 40
    ```

    ![CLI k3d](assets/preview/qimLogo.png){ width="512" }
