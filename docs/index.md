<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">

<audio id="audio" src="assets/qim3d.mp3"></audio>

<script>
document.addEventListener("DOMContentLoaded", function() {
  const audio = document.getElementById("audio");
  const playButton = document.getElementById("playButton");

  playButton.addEventListener("click", function() {
    const icon = playButton.querySelector("i");
    if (audio.paused) {
      audio.play();
      icon.classList.remove("fa-circle-play");
      icon.classList.add("fa-circle-pause");
    } else {
      audio.pause();
      icon.classList.remove("fa-circle-pause");
      icon.classList.add("fa-circle-play");
    }
  });

  audio.addEventListener("ended", function() {
    const icon = playButton.querySelector("i");
    icon.classList.remove("fa-circle-pause");
    icon.classList.add("fa-circle-play");
  });
});
</script>

# ![qim3d logo](assets/qim3d-logo.svg){ width="25%" }

[![PyPI version](https://badge.fury.io/py/qim3d.svg)](https://badge.fury.io/py/qim3d)
[![Downloads](https://static.pepy.tech/badge/qim3d)](https://pepy.tech/project/qim3d)

The **`qim3d`** (kɪm θriː diː <button id="playButton"><i class="fa-regular fa-circle-play"></i></button>)  library is designed for **Quantitative Imaging in 3D** using Python. It offers a range of features, including data loading and manipulation, image processing and filtering, data visualization, and analysis of imaging results.

You can easily load and process 3D image data from various file formats, apply filters and transformations to the data, visualize the results using interactive plots and 3D volumetric rendering.

Whether you are working with medical imaging data, materials science data, or any other type of 3D imaging data, `qim3d` provides a convenient and powerful set of tools to help you analyze and understand your data.

!!! Example "Interactive volume slicer"
    ```python
    import qim3d

    vol = qim3d.examples.bone_128x128x128
    qim3d.viz.slicer(vol)
    ```
    ![viz slicer](assets/screenshots/viz-slicer.gif)

!!! Example "Synthetic data generation"
    ```python
    import qim3d

    # Generate synthetic collection of blobs
    num_objects = 15
    synthetic_collection, labels = qim3d.generate.collection(num_objects = num_objects)

    # Visualize synthetic collection
    qim3d.viz.vol(synthetic_collection)
    ```
    <iframe src="https://platform.qim.dk/k3d/synthetic_collection_default.html" width="100%" height="500" frameborder="0"></iframe>

!!! Example "Structure tensor"
    ```python
    import qim3d

    vol = qim3d.examples.NT_128x128x128
    val, vec = qim3d.processing.structure_tensor(vol, visualize = True, axis = 2)
    ```

    ![structure tensor](assets/screenshots/structure_tensor_visualization.gif)

## Installation

### Create environment

Creating a `conda` environment is not required but recommended.

??? info "Miniconda installation and setup"

    [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) is a free minimal installer for conda. 

    Here are some quick command line instructions to help you set up the latest Miniconda installer promptly. For graphical installers (.exe and .pkg) and instructions on hash checking, please refer to [Installing Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/).

    === "Windows"
        These three commands quickly and quietly install the latest 64-bit version of the installer and then clean up after themselves. To install a different version or architecture of Miniconda for Windows, change the name of the `.exe` installer in the `curl` command.

        ```bash
        curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
        start /wait "" miniconda.exe /S
        del miniconda.exe
        ```
        After installing, open the “Anaconda Prompt (miniconda3)” program to use Miniconda3. For the Powershell version, use “Anaconda Powershell Prompt (miniconda3)”.
    
    === "macOS"
        These four commands quickly and quietly install the latest M1 macOS version of the installer and then clean up after themselves. To install a different version or architecture of Miniconda for macOS, change the name of the `.sh` installer in the `curl` command.

        ```bash
        mkdir -p ~/miniconda3
        curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
        rm -rf ~/miniconda3/miniconda.sh
        ```

        After installing, initialize your newly-installed Miniconda. The following commands initialize for bash and zsh shells:

        ```bash
        ~/miniconda3/bin/conda init bash
        ~/miniconda3/bin/conda init zsh
        ```

    === "Linux"
        These four commands quickly and quietly install the latest 64-bit version of the installer and then clean up after themselves. To install a different version or architecture of Miniconda for Linux, change the name of the `.sh` installer in the `wget` command.

        ```bash
        mkdir -p ~/miniconda3
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
        rm -rf ~/miniconda3/miniconda.sh
        ```

        After installing, initialize your newly-installed Miniconda. The following commands initialize for bash and zsh shells:

        ```bash
        ~/miniconda3/bin/conda init bash
        ~/miniconda3/bin/conda init zsh
        ```
Once you have `conda` installed, create a new enviroment:

    conda create -n qim3d python=3.11

After the environment is created, activate it by running:

    conda activate qim3d

### Install using `pip`

The latest stable version can be simply installed using `pip`:

    pip install qim3d

!!! note
    The base installation of `qim3d` does not include deep-learning dependencies by design, keeping the library lightweight for scenarios where these dependencies are unnecessary.

    If you need to use deep-learning features, you can install the additional dependencies by running: `pip install qim3d['deep-learning']`

## Troubleshooting

### Get the latest version

The library is under constant development, so make sure to keep your installation updated:

    pip install --upgrade qim3d


## Collaboration

Contributions to `qim3d` are welcome!

If you find a bug, have a feature request, or would like to contribute code, please open an issue or submit a pull request.

You can find us at Gitlab:
[https://lab.compute.dtu.dk/QIM/tools/qim3d](https://lab.compute.dtu.dk/QIM/tools/qim3d
)

This project is licensed under the [MIT License](https://lab.compute.dtu.dk/QIM/tools/qim3d/-/blob/main/LICENSE).

### Contributors

Below is a list of contributors to the project, arranged in chronological order of their first commit to the repository:

| Author                    |   Commits | First commit |
|:--------------------------|----------:|-------------:|
| Felipe Delestro           |       170 | 2023-05-12   |
| Stefan Engelmann Jensen   |        29 | 2023-06-29   |
| Oskar Kristoffersen       |        15 | 2023-07-05   |
| Christian Kento Rasmussen |        19 | 2024-02-01   |
| Alessia Saccardo          |         7 | 2024-02-19   |
| David Grundfest           |         4 | 2024-04-12   |
| Anna Bøgevang Ekner       |         3 | 2024-04-18   |

## Support

The development of `qim3d` is supported by:

![Novo Nordisk Foundation](https://novonordiskfonden.dk//app/uploads/NNF-INT_logo_tagline_blue_RGB_solid.png){ width="256" }
