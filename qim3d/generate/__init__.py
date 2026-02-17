"""Generation for synthetic datasets."""

from ._aggregators import volume_collection
from ._generators import (
    ParameterVisualizer,
    background,
    volume,
)
from ._shapes import berry, rope

__all__ = [
    'volume',
    'volume_collection',
    'background',
    'ParameterVisualizer',
    'berry',
    'rope',
]
