# Create a class to handle to fibers in volumetric data
import os
import numpy as np
from typing import List, Optional, Tuple

class FiberBundle:
    VALID_EXTENSIONS = {'.nii', '.tiff', '.tif', '.dcm', '.vgi', '.txrm'}

    def __init__(
        self,
        centrelines: List[np.ndarray],
        voxel_size: Tuple[float, float, float],
        volume_dimensions: Tuple[int, int, int],
        origin: Tuple[float, float, float],
        volume_path: str,
        bundle_ids: Optional[List[Optional[int]]] = None
    ):
        self.centrelines = centrelines
        self.voxel_size = voxel_size
        self.volume_dimensions = volume_dimensions
        self.origin = origin
        self.volume_path = volume_path
        self.bundle_ids = bundle_ids if bundle_ids is not None else [None] * len(centrelines)

        self._validate()

    def _validate(self):
        if len(self.bundle_ids) != len(self.centrelines):
            raise ValueError("Length of bundle_ids must match number of centrelines")

        if not os.path.isfile(self.volume_path):
            raise FileNotFoundError(f"Volume file not found: {self.volume_path}")

        _, ext = os.path.splitext(self.volume_path)
        if ext.lower() not in self.VALID_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}. Must be one of {self.VALID_EXTENSIONS}")

    def assign_bundle_ids(self, ids: List[Optional[int]]):
        """Assign bundle IDs to the centrelines."""
        if len(ids) != len(self.centrelines):
            raise ValueError("Length of new bundle_ids must match number of centrelines")
        self.bundle_ids = ids

    def get_bundle_list(self, bundle_id: int) -> List[np.ndarray]:
        """Return a list of centrelines belonging to a specific bundle ID."""
        return [cline for cline, bid in zip(self.centrelines, self.bundle_ids) if bid == bundle_id]

    def get_bundle(self, bundle_id: int) -> 'FiberBundle':
        """Return a new FiberBundle instance containing only the centrelines of the specified bundle ID."""
        selected_centrelines = []
        selected_ids = []
        for cline, bid in zip(self.centrelines, self.bundle_ids):
            if bid == bundle_id:
                selected_centrelines.append(cline)
                selected_ids.append(bid)

        return FiberBundle(
            centrelines=selected_centrelines,
            voxel_size=self.voxel_size,
            volume_dimensions=self.volume_dimensions,
            origin=self.origin,
            volume_path=self.volume_path,
            bundle_ids=selected_ids
        )

    def summary(self):
        print(f"Number of centrelines: {len(self.centrelines)}")
        print(f"Voxel size: {self.voxel_size}")
        print(f"Volume dimensions: {self.volume_dimensions}")
        print(f"Origin: {self.origin}")
        print(f"Volume path: {self.volume_path}")
        unique_ids = set(filter(lambda x: x is not None, self.bundle_ids))
        print(f"Unique bundle IDs: {sorted(unique_ids)}")

