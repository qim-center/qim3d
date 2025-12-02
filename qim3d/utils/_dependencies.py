import importlib
from typing import Any


def optional_import(
    module_name: str,
    extra: str | None = None,
    raise_on_missing: bool = True,  # This is always True in your use case
) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:  # Catch the ModuleNotFoundError
        msg = f"Missing dependency '{module_name}'."
        if extra:
            msg += f' Install via: pip install qim3d[{extra}]'

        # --- CRITICAL CHANGE HERE ---
        # Raise your custom error, suppressing the original exception traceback.
        if raise_on_missing:
            # raise ImportError(msg) from e # This links the two tracebacks
            raise ImportError(msg) from None  # This suppresses the original traceback

        # ... (rest of the function for log.warning when raise_on_missing=False) ...
