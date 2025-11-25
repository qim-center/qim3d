import importlib
import warnings


class MissingOptionalDependencyWarning(UserWarning):
    """Warning for missing optional dependencies."""



def optional_import(module_name: str, extra: str | None = None):
    """
    Attempt to import an optional dependency.
    Emits a warning instead of raising ImportError.

    Parameters
    ----------
    module_name : str
        The module to import.
    extra : str or None
        The optional-dependency group to suggest, e.g. "deep-learning".

    Returns
    -------
    module or None

    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        msg = f"Optional dependency '{module_name}' is not installed."
        if extra:
            msg += f' Install it with: pip install qim3d[{extra}]'
        warnings.warn(msg, MissingOptionalDependencyWarning, stacklevel=2)
        return None
