# Synthetic Data

Synthetic volumetric data can be generated using `qim3d.generate`.

## Blobs
Synthetic 3D blobs are generated using Perlin noise. The `noise_scale` parameter controls the texture and roughness of the generated structure:

``` py
import qim3d

# Generate a synthetic blob with Perlin noise
vol = qim3d.generate.volume(noise_scale=0.03, noise_type='perlin')

# Visualize slices of the volume
qim3d.viz.slices_grid(vol, n_slices=15)
```
<div class="notebook-output-figure">
    <img src="../../assets/screenshots/synthetic_data_blob.gif" alt="synthetic_data_blob">
</div>

The noise scale can be adjusted to change the roughness of the blob. A higher `noise_scale` creates finer, more detailed Perlin noise features:

``` py
# Generate a smoother blob (low frequency noise)
smooth_blob = qim3d.generate.volume(noise_scale=0.01)

# Generate a rougher blob (high frequency noise)
rough_blob = qim3d.generate.volume(noise_scale=0.05)
```

<div class="notebook-output-figure">
    <img src="../../assets/screenshots/synthetic_data_blobs_noise.png" alt="synthetic_data_blobs_noise_scale">
</div>

## Clusters
Synthetic cluster volumes can be generated using `qim3d.generate.berry`. This function creates a cluster of individual units (drupelets) arranged in a natural spherical pattern:

``` py
# Generate a synthetic cluster volume
cluster = qim3d.generate.berry(num_drupelets=60, core_radius=20)

# Visualize slices of the cluster
qim3d.viz.slices_grid(cluster, n_slices=15)
```
<div class="notebook-output-figure">
    <img src="../../assets/screenshots/synthetic_data_berry.gif" alt="synthetic_data_berry">
</div>

Parameters such as `num_drupelets`, `drupelet_radius`, and `return_labels` can be explored to adjust cluster density or obtain instance labels for segmentation:

``` py
# Generate a cluster with labeled individual units
cluster, labels = qim3d.generate.berry(num_drupelets=80, return_labels=True)

# Visualize labeled units with a segmentation colormap
cmap = qim3d.viz.colormaps.segmentation(n_labels=len(labels))
qim3d.viz.slices_grid(labels, colormap=cmap, n_slices=15)
```
<div class="notebook-output-figure">
    <img src="../../assets/screenshots/synthetic_cluster_labels.png" alt="synthetic_cluster_labels">
</div>

## Fibers
Synthetic fiber bundles and twisted ropes can be generated using `qim3d.generate.rope`:

``` py
# Generate a synthetic fiber bundle
fiber_bundle = qim3d.generate.rope(num_threads=18, twist_rate=2.0)

# Visualize slices along the length
qim3d.viz.slices_grid(fiber_bundle, n_slices=15)
```
<div class="notebook-output-figure">
    <img src="../../assets/screenshots/synthetic_data_rope.gif" alt="synthetic_data_rope">
</div>

Parameters such as `num_threads`, `twist_rate`, and `thread_thickness` allow exploring different fiber bundle densities and twist frequencies:

``` py
# Generate a fiber bundle with a higher twist rate and instance labels
rope_vol, thread_labels = qim3d.generate.rope(
    num_threads=12,
    twist_rate=4.0,
    thread_thickness=12,
    return_labels=True
)

# Visualize labeled fibers with a segmentation colormap
cmap = qim3d.viz.colormaps.segmentation(n_labels=len(thread_labels))
qim3d.viz.slices_grid(thread_labels, colormap=cmap, n_slices=15)
```
<div class="notebook-output-figure">
    <img src="../../assets/screenshots/synthetic_rope_labels.png" alt="synthetic_rope_labels">
</div>
