from setuptools import setup, find_packages
import os


# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="qim3d",
    version="0.2.0",
    author="Felipe Delestro",
    author_email="fima@dtu.dk",
    description="QIM tools and user interfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://lab.compute.dtu.dk/QIM/tools/qim3d",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: User Interfaces",
    ],
    python_requires=">=3.6",
    install_requires=[
        "albumentations>=1.3.1",
        "gradio>=3.44.3",
        "h5py>=3.9.0",
        "localthickness>=0.1.2",
        "matplotlib>=3.7.1",
        "monai>=1.2.0",
        "numpy>=1.25.2",
        "outputformat>=0.1.3",
        "Pillow>=10.0.1",
        "plotly>=5.14.1",
        "scipy>=1.11.2",
        "seaborn>=0.12.2",
        "setuptools>=68.0.0",
        "tifffile>=2023.4.12",
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "torchinfo>=1.8.0",
        "tqdm>=4.65.0",
    ],
)
