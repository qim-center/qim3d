# Data Handling

The qim3d library allows for easy data handling, downloading, loading and saving in multiple ways.
First, qim3d is imported:
``` py
import qim3d
```

## Downloading volumes
The [QIM data repository](https://data.qim.dk/) contains different volumes readily available for download using qim3d.Here, a mussel volume is downloaded:
``` py
downloader = qim3d.io.Downloader()
data = downloader.Mussel.ClosedMussel1_DOWNSAMPLED(load_file=True)
```
<div class="notebook-output"><pre>
Downloading ClosedMussel1_DOWNSAMPLED.tif
https://archive.compute.dtu.dk/download/public/projects/viscomp_data_repository/Mussel/ClosedMussel1_DOWNSAMPLED.tif
143MB [00:05, 26.5MB/s]  

Loading ClosedMussel1_DOWNSAMPLED.tif
Using virtual stack
</pre></div>


A full list of all avaialable volumes for download can be printed:
``` py
downloader = qim3d.io.Downloader()
downloader.list_files()
```
<div class="notebook-output"><pre>
╭──────╮
│ Coal │
╰──────╯
Coal.CoalBrikett                                  (2.24GB)
Coal.CoalBrikettZoom_DOWNSAMPLED                  (238.50MB)
Coal.CoalBrikett_Zoom                             (3.73GB)

╭────────╮
│ Corals │
╰────────╯
Corals.Coral_1                                    (2.27GB)
Corals.Coral_2                                    (2.38GB)
Corals.Coral2_DOWNSAMPLED                         (152.66MB)
Corals.Coral_1_1                                  (1.83GB)
Corals.Coral_1_2                                  (1.83GB)
Corals.Coral_1_3                                  (1.83GB)
Corals.MexCoral                                   (2.24GB)

╭─────────────╮
│ Cowry_Shell │
╰─────────────╯
Cowry_Shell.Cowry_DOWNSAMPLED                     (116.91MB)
Cowry_Shell.Cowry_Shell                           (1.83GB)

╭──────╮
│ Crab │
╰──────╯
Crab.HerrmitCrab                                  (2.38GB)
Crab.OkinawaCrab                                  (1.86GB)

╭───────────────╮
│ Deer_Mandible │
╰───────────────╯
Deer_Mandible.Animal_Mandible                     (2.79GB)
Deer_Mandible.DeerMandible_DOWNSAMPLED            (638.71MB)

╭──────╮
│ Foam │
╰──────╯
Foam.Foam                                         (3.73GB)
Foam.Foam_2                                       (3.73GB)
Foam.Foam_2_zoom                                  (3.73GB)
Foam.Foam_DOWNSAMPLED                             (238.49MB)

╭───────────╮
│ Hourglass │
╰───────────╯
Hourglass.Hourglass                               (3.73GB)
Hourglass.Hourglass_4X-80kV-Air-9s-1_97um         (1.83GB)
Hourglass.Hourglass_longexp_rerun                 (3.73GB)

╭──────╮
│ Kiwi │
╰──────╯
Kiwi.Kiwi                                         (2.87GB)

╭────────╮
│ Loofah │
╰────────╯
Loofah.Loofah                                     (2.24GB)
Loofah.Loofah_DOWNSAMPLED                         (143.10MB)

╭───────────────────╮
│ Marine_Gastropods │
╰───────────────────╯
Marine_Gastropods.MarineGastropod1_DOWNSAMPLED    (143.10MB)
Marine_Gastropods.MarineGastropod2_DOWNSAMPLED    (166.95MB)
Marine_Gastropods.MarineGatropod_1                (2.24GB)
Marine_Gastropods.MarineGatropod_2                (2.61GB)

╭────────╮
│ Mussel │
╰────────╯
Mussel.ClosedMussel1                              (2.24GB)
Mussel.ClosedMussel1_DOWNSAMPLED                  (143.11MB)

╭────────────╮
│ Oak_Branch │
╰────────────╯
Oak_Branch.Oak_branch                             (2.38GB)
Oak_Branch.OakBranch_DOWNSAMPLED                  (152.67MB)

╭────────────────╮
│ Okinawa_Forams │
╰────────────────╯
Okinawa_Forams.Okinawa_Foram_1                    (1.85GB)
Okinawa_Forams.Okinawa_Foram_2                    (1.84GB)

╭──────────╮
│ Physalis │
╰──────────╯
Physalis.Physalis                                 (3.73GB)
Physalis.Physalis_DOWNSAMPLED                     (238.50MB)

╭───────────╮
│ Raspberry │
╰───────────╯
Raspberry.Raspberry2                              (2.98GB)
Raspberry.Raspberry2_DOWNSAMPLED                  (190.81MB)

╭──────╮
│ Rope │
╰──────╯
Rope.FibreRope1                                   (1.83GB)
Rope.FibreRope1_DOWNSAMPLED                       (686.81MB)

╭────────────╮
│ Sea_Urchin │
╰────────────╯
Sea_Urchin.Cordatum_Shell                         (1.85GB)
Sea_Urchin.Cordatum_Spine                         (183.40MB)
Sea_Urchin.SeaUrchin                              (2.61GB)

╭───────╮
│ Sepia │
╰───────╯
Sepia.Sepia_Tiffs                                 (2.70GB)

╭───────╮
│ Snail │
╰───────╯
Snail.Escargot                                    (2.61GB)

╭────────╮
│ Sponge │
╰────────╯
Sponge.Sponge                                     (1.12GB)
</pre></div>


## Loading and saving files
The qim3d library handles loading in volumetric data of many different file formats, like Tiff, HDF5, TXRM/TXM/XRM, NifTI, PIL, VOL/VGI, DICOM. Simply use the load function:
``` py
vol = qim3d.io.load('./Mussel/ClosedMussel1_DOWNSAMPLED.tif')
```
Volumes can also be saved to specific file paths and in different file formats, like saving the mussel volume in the NIfTI format:
```py
qim3d.io.save('./processed/ClosedMussel.nii')
```



## OME-Zarr files
The qim3d library can also be used for converting volumes to the OME-Zarr format to enable faster, more memory-efficient analysis by working with a chunked, multiscale version of the volume that is optimized for on-demand access:
``` py
qim3d.io.export_ome_zarr("Mussel.zarr", data, chunk_size=100, downsample_rate=2, replace=True)
```
OME-Zarr volumes can easily be loaded in:
``` py
volume = qim3d.io.import_ome_zarr("Mussel.zarr", scale=1, load=True)
```
<div class="notebook-output"><pre>
Data contains 4 scales:
- Scale 0: (300, 500, 500)
- Scale 1: (150, 250, 250)
- Scale 2: (75, 125, 125)
- Scale 3: (38, 62, 62)

Loading scale 1 with shape (150, 250, 250)
</pre></div>
