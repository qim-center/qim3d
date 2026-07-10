# Visualization

The qim3d library allows for quick and easy visualization of volumetric data.
First, qim3d is imported, and a mussel volume is loaded:
``` py
import qim3d
downloader = qim3d.io.Downloader()
volume = downloader.Mussel.ClosedMussel1_DOWNSAMPLED(load_file=True)
```

## Slices
Equidistant slices of the mussel µCT-scan can be viewed. Here 15 slices are chosen:
``` py
qim3d.viz.slices_grid(volume, num_slices=15, color_map='Blues')
```
<div class="notebook-output-figure">
  <img src="../../assets/screenshots/mussel_slices_grid.png" alt="mussel_slices">
</div>

One can interactively scroll through the different axes of the volume:
``` py
qim3d.viz.slicer_orthogonal(volume, colormap='Blues')
```
<div class="notebook-output-figure">
    <img src="../../assets/screenshots/mussel_slicer_orthogonal.png" alt="mussel_slicer_orthogonal">
</div>

## Histogram
Histograms of the voxel intensities from a volume can help detect possible segmentation thresholds. The `coarseness` parameter is used to subsample the original volume to get a quick estimate:
``` py
qim3d.viz.histogram(volume, bins=100, coarseness=2)
```

<div class="notebook-output-figure">
    <pre>Subsampled volume has size 12.5% of the original volume.</pre>
    <img src="../../assets/screenshots/mussel_histogram.png" alt="mussel_histogram">
</div>
