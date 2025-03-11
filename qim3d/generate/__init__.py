"""Generation for synthetic datasets."""

from ._aggregators import volume_collection
from ._generators import noise_volume, volume

__all__ = ['volume', 'volume_collection', 'noise_volume']
