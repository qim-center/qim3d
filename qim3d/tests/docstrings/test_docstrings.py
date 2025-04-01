import pytest
import matplotlib
from mktestdocs import check_docstring, get_codeblock_members, grab_code_blocks

from qim3d.tests import get_all_functions_by_module

matplotlib.use('Agg')

# Get dictionary of functions by module
functions_by_module = get_all_functions_by_module()

@pytest.mark.parametrize('func', functions_by_module["mesh"], ids=lambda d: d.__name__)
def test_docstrings_mesh(func):
    check_docstring(obj=func)

# ------------------------- NOTE: This does not work yet -------------------- #
# @pytest.mark.parametrize('func', functions_detection, ids=lambda d: d.__name__)
# def test_docstrings_detection(func):

#     # Combine all code blocks into one 
#     for b in grab_code_blocks(func.__doc__, lang = "python"):
#         all_code = f"{all_code}\n{b}"

#     exec(all_code)
# --------------------------------------------------------------------------- #