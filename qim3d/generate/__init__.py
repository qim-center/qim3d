"""Generation for synthetic datasets."""

from ._aggregators import volume_collection
from ._generators import background, volume

__all__ = ['volume', 'volume_collection', 'background']
