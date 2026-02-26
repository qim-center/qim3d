<audio id="audio" src="assets/qim3d.mp3"></audio>

<script>
document.addEventListener("DOMContentLoaded", function() {
  const audio = document.getElementById("audio");
  const playButton = document.getElementById("playButton");

  if (!playButton || !audio) return;

  playButton.addEventListener("click", function() {
    if (audio.paused) {
      audio.play();
      playButton.innerHTML = "⏸"; // Swaps to pause symbol
    } else {
      audio.pause();
      playButton.innerHTML = "▶️"; // Swaps back to play symbol
    }
  });

  audio.addEventListener("ended", function() {
    playButton.innerHTML = "▶️";
  });
});
</script>


# ![qim3d logo](assets/Qim-logo_standard-full-title-transparent-background.png){ width="65%" }

[![PyPI version](https://badge.fury.io/py/qim3d.svg)](https://badge.fury.io/py/qim3d)
[![Downloads](https://static.pepy.tech/badge/qim3d)](https://pepy.tech/project/qim3d)

The **`qim3d`** (kɪm θriː di: <button id="playButton" style="background: none; border: none; cursor: pointer; padding: 0; margin-left: 2px; font-size: 0.75em;" title="Play pronunciation">▶️</button> )  library is designed for **Quantitative Imaging in 3D** using Python. It offers a range of features, including data loading and manipulation, image processing and filtering, data visualization, and analysis of imaging results.

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
    num_volumes = 15
    volume_collection, labels = qim3d.generate.volume_collection(num_volumes = num_volumes)

    # Visualize the collection
    qim3d.viz.vol(volume_collection)
    ```
    <iframe src="https://platform.qim.dk/k3d/synthetic_collection_default.html" width="100%" height="500" frameborder="0"></iframe>

!!! Example "Structure tensor"
    ```python
    import qim3d

    vol = qim3d.examples.fibers_150x150x150
    val, vec = qim3d.processing.structure_tensor(vol, visualize = True, axis = 1)
    ```
    ![structure tensor](assets/screenshots/structure_tensor_visualization_fibers.gif)
