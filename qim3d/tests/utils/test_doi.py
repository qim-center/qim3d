import qim3d

doi = "https://doi.org/10.1007/s10851-021-01041-3"


def test_get_bibtex():
    bibtext = qim3d.utils.doi.get_bibtex(doi)

    assert "Measuring Shape Relations Using r-Parallel Sets" in bibtext


def test_get_reference():
    reference = qim3d.utils.doi.get_reference(doi)

    assert "Stephensen" in reference
