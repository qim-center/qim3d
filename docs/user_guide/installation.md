# Installation

## Create environment

Creating a `conda` environment is not required but recommended.

??? info "Miniconda installation and setup"

    [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) is a free minimal installer for conda. 

    Here are some quick instructions to help you set up the latest Miniconda installer for your system: 

    === "Windows"

        The easiest way to install Miniconda on Windows is through the graphical interface installer. Follow these steps:

        1. Download the installer [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe).
        2. Run the installer and follow the on-screen instructions.
        3. When the installation finishes, open `Anaconda Prompt (miniconda3)` from the Start menu.
        
    
    === "macOS"
 
        The easiest way to install Miniconda on macOS is through the graphical interface installer. Follow these steps:

        1. Download the correct installer for your processor version. If you are unsure about your version, check [here](https://support.apple.com/en-us/116943).
            - For Intel processors, download [x86](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.pkg)
            - For Apple Silicon (M1/M2/M3 etc) processors, download [arm64](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.pkg)

        2. Run the installer and follow the on-screen instructions.
        
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
Once you have `conda` installed, open your terminal and create a new enviroment:

    conda create -n qim3d python=3.11

After the environment is created, activate it by running:

    conda activate qim3d

Remember, if you chose to create an environment to install `qim3d`, it needs to be activated each time before using the library.

## Install using `pip`

The latest stable version can be simply installed using `pip`. Open your terminal and run:

    pip install qim3d

After completing the installation, you can verify its success by running one or both of the following commands:

    qim3d

or:

    pip show qim3d

If either command displays information about the qim3d library, the installation was successful.

## Optional dependencies

`qim3d` includes some features that require additional Python packages. These are not installed by default, keeping the base library lightweight. 

You can install optional dependencies for specific features:

| Feature | Optional dependency | Install command |
|---------|-------------------|----------------|
| Deep-learning / model training | `torch`, `torchvision`, `torchinfo`, `monai` | `pip install qim3d[deep-learning]` |
| Synthetic data generation | `noise` | `pip install qim3d[synthetic-data]` |
| GUI / interactive tools | `gradio` | `pip install qim3d[gui]` |
| All optional features | All of the above | `pip install qim3d[all]` |

!!! note
    If you try to run a script that requires an optional dependency and it is not installed, `qim3d` will show a ImportError with instructions on how to install the missing dependency.

!!! tip "Installing Multiple Features"
    You can install multiple optional dependency groups simultaneously by separating the names with a comma, without spaces, inside the brackets.     
    For example, to install both the `synthetic-data` features and the `deep-learning` features, use the following command:

    ```bash
    pip install qim3d[synthetic-data,deep-learning]
    ```

!!! note "Installing qim3d And CIL In The Same Environment"

    `CIL` is an open-source Python framework for tomographic imaging, and is often used together with `qim3d`. To install both packages in the same environment, create and activate a new environment, then run:

    ```bash
    conda install -c conda-forge -c ccpi cil=26.0.0 && pip install qim3d
    ```

    This installs CIL first, and then qim3d. The installation order is important: `CIL` must be installed before `qim3d` to avoid package conflicts.


## Troubleshooting

Here are some solutions for commonly found issues during installation and usage of `qim3d`.

### Failed building

Some Windows users could face an build error during installation.

??? Bug "ERROR: Failed building wheel for noise"
    ```
    Building wheels for collected packages: noise, outputformat, asciitree, ffmpy
    Building wheel for noise (setup.py) ... error
    error: subprocess-exited-with-error

    × python setup.py bdist_wheel did not run successfully.
    │ exit code: 1
    ╰─> [14 lines of output]
        running bdist_wheel
        running build
        running build_py
        creating build
        creating build\lib.win-amd64-cpython-311
        creating build\lib.win-amd64-cpython-311\noise
        copying perlin.py -> build\lib.win-amd64-cpython-311\noise
        copying shader.py -> build\lib.win-amd64-cpython-311\noise
        copying shader_noise.py -> build\lib.win-amd64-cpython-311\noise
        copying test.py -> build\lib.win-amd64-cpython-311\noise
        copying __init__.py -> build\lib.win-amd64-cpython-311\noise
        running build_ext
        building 'noise._simplex' extension
        error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
        [end of output]

    note: This error originates from a subprocess, and is likely not a problem with pip.
    ERROR: Failed building wheel for noise
    ```

This issue occurs because the system lacks the necessary tools to compile the library requirements. To resolve this, follow these steps:

- Go to the [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) page and click on "Download build tools."
- Run the installer and ensure that `Desktop development with C++` is checked. ![Windows build tools](assets/screenshots/Troubleshooting-Windows_build_tools.png)
- Restart your computer
- Activate your conda enviroment and run `pip install qim3d` again

### Get the latest version

The library is under constant development, so make sure to keep your installation updated:

    pip install --upgrade qim3d

