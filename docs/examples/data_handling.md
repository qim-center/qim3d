# Data Handling

The qim3d library allows for easy data handling, downloading, loading and saving in multiple ways.
First, qim3d is imported:
``` py
import qim3d
```

## Downloading volumes
The [QIM data repository](https://data-repository.qim.dk/) contains different volumes readily available for download using qim3d. Here, a mussel volume is downloaded and loaded:
``` py
downloader = qim3d.io.Downloader()
data = downloader.load_dataset("mussel", format="tiff")
```

A full list of all available datasets for download can be returned:
``` py
downloader.get_datasets()
```
<div class="notebook-output"><pre>
[{'categories': ['material'],
  'contributors': [],
  'description': '...',
  'id': 'coal-briquette',
  'license': None,
  'page_url': '/datasets/coal-briquette/',
  'summary': 'Coal briquette from a bag of BBQ coal',
  'title': 'Coal Briquette',
  'volumes': [{'format': 'zarr',
               'size_bytes': 2400000000,
               'url': 'https://public.qim.dk/coal_briquette/coal_briquette.zarr'},
              {'format': 'tiff',
               'size_bytes': 2400082900,
               'url': 'https://public.qim.dk/coal_briquette/coal_briquette.tif'}]},
 ...]
</pre></div>

## Loading and saving files
The qim3d library handles loading in volumetric data of many different file formats, like Tiff, HDF5, TXRM/TXM/XRM, NifTI, PIL, VOL/VGI, DICOM. Simply use the load function:
``` py
vol = qim3d.io.load("./mussel/mussel.tif")
```
Volumes can also be saved to specific file paths and in different file formats, like saving the mussel volume in the NIfTI format:
```py
qim3d.io.save("./processed/ClosedMussel.nii")
```



## OME-Zarr files
The qim3d library can also be used for converting volumes to the OME-Zarr format to enable faster, more memory-efficient analysis by working with a chunked, multiscale version of the volume that is optimized for on-demand access:
``` py
qim3d.io.export_ome_zarr(
    "Mussel.zarr", data, chunk_size=100, downsample_rate=2, replace=True
)
```
OME-Zarr volumes can easily be loaded in:
``` py
volume = qim3d.io.import_ome_zarr("Mussel.zarr", scale=1, load=True)
```
<div class="notebook-output"><pre>
Data contains 5 scales:
- Scale 0: (600, 1000, 1000)
- Scale 1: (300, 500, 500)
- Scale 2: (150, 250, 250)
- Scale 3: (75, 125, 125)
- Scale 4: (37, 62, 62)

Loading scale 1 with shape (300, 500, 500)
</pre></div>
