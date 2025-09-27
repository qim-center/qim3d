import functools
import inspect
from collections.abc import Callable
from typing import Any

import qim3d
import qim3d.operations


def coarseness(*volumes: str) -> Callable:
    """
    Decorator for subsampling volumes before passing them into a function.

    Args:
        *volumes (str): The parameter names of the volumes which are subsampled when the coarseness parameter is passed a value in the decorated function.

    """

    def find_kwargs(sig: inspect.Signature) -> str | None:
        """Find the **kwargs parameter name, return None if it does not exist."""
        for pname, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return pname
        return None

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        kwargs_pname = find_kwargs(sig)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Handle if the original function does not have coarseness nor kwargs
            if (
                'coarseness' in kwargs
                and 'coarseness' not in sig.parameters
                and not kwargs_pname
            ):
                coarseness = kwargs.pop(
                    'coarseness'
                )  # remove it from kwargs, otherwise bind wont work
            else:
                coarseness = None

            boundargs = sig.bind(*args, **kwargs)
            boundargs.apply_defaults()
            mapping = boundargs.arguments

            if 'coarseness' in mapping:
                coarseness = mapping.pop('coarseness')
            elif kwargs_pname and 'coarseness' in mapping[kwargs_pname]:
                # Handle if the original function has a **kwargs parameter.
                # Have to modify mapping since boundargs.kwargs is dynamically computed from it.
                coarseness = mapping[kwargs_pname].pop('coarseness')

            if coarseness:
                for pname in volumes:
                    vol = mapping[pname]
                    vol = qim3d.operations.subsample(vol, coarseness)
                    mapping[pname] = vol
            return func(*boundargs.args, **boundargs.kwargs)

        return wrapper

    return decorator
