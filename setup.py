from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="qim",
    version="0.1.0",
    author="Felipe Delestro",
    author_email="fima@dtu.dk",
    description="QIM tools and user interfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://lab.compute.dtu.dk/QIM/qim",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: User Interfaces",
    ],
    python_requires=">=3.6",
    install_requires=open("requirements.txt").readlines(),
)
