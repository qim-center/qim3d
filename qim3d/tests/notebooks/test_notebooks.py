from testbook import testbook


def test_blob_detection_notebook():
    with testbook('./docs/notebooks/blob_detection.ipynb', execute=True) as tb:
        pass


def test_filters_notebook():
    with testbook('./docs/notebooks/filters.ipynb', execute=True) as tb:
        pass


def test_local_thickness_notebook():
    with testbook('./docs/notebooks/local_thickness.ipynb', execute=True) as tb:
        pass


def test_logging_notebook():
    with testbook('./docs/notebooks/Logging.ipynb', execute=True) as tb:
        pass


def test_references_from_doi_notebook():
    with testbook('./docs/notebooks/References from DOI.ipynb', execute=True) as tb:
        pass


def test_structure_tensor_notebook():
    with testbook('./docs/notebooks/structure_tensor.ipynb', execute=True) as tb:
        pass
