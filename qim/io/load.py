import tifffile
import h5py
import os
import difflib


class DataLoader:
    def __init__(self, **kwargs):
        self.verbose = False

        # Virtual stack is False by default
        self.virtual_stack = kwargs.get("virtual_stack", False)

    def load_tiff(self, path):
        if self.virtual_stack:
            vol = tifffile.memmap(path)
        else:
            vol = tifffile.imread(path)

        return vol

    def load_h5(self, path):
        with h5py.File(path, "r") as f:
            vol = f["data"][:]
        return vol

    def load(self, path):
        # Load a single file
        if os.path.isfile(path):
            if path.endswith(".tif") or path.endswith(".tiff"):
                return self.load_tiff(path)
            elif path.endswith(".h5"):
                return self.load_h5(path)
            else:
                raise ValueError("Unsupported file format")

        # Load a directory
        elif os.path.isdir(path):
            raise NotImplementedError("Loading from directory is not implemented yet")

        # Fails
        else:
            # Find the closest matching path to warn the user
            parent_dir = os.path.dirname(path)
            parent_files = os.listdir(parent_dir)
            valid_paths = [os.path.join(parent_dir, file) for file in parent_files]
            similar_paths = difflib.get_close_matches(path, valid_paths)
            if similar_paths:
                suggestion = similar_paths[0]  # Get the closest match
                message = f"Invalid path.\nDid you mean '{suggestion}'?"
                raise ValueError(message)
            else:
                raise ValueError("Invalid path")


def load(path, **kwargs):
    return DataLoader(**kwargs).load(path)
