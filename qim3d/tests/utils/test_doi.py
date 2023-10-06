import qim3d

doi = "https://doi.org/10.1007/s10851-021-01041-3"


def test_get_bibtex():
    bibtext = qim3d.utils.doi.get_bibtex(doi)

    assert (
        bibtext
        == "@article{Stephensen_2021,\n\tdoi = {10.1007/s10851-021-01041-3},\n\turl = {https://doi.org/10.1007%2Fs10851-021-01041-3},\n\tyear = 2021,\n\tmonth = {jun},\n\tpublisher = {Springer Science and Business Media {LLC}},\n\tvolume = {63},\n\tnumber = {8},\n\tpages = {1069--1083},\n\tauthor = {Hans J. T. Stephensen and Anne Marie Svane and Carlos B. Villanueva and Steven A. Goldman and Jon Sporring},\n\ttitle = {Measuring Shape Relations Using r-Parallel Sets},\n\tjournal = {Journal of Mathematical Imaging and Vision}\n}"
    )


def test_get_reference():
    reference = qim3d.utils.doi.get_reference(doi)

    assert (
        reference
        == "Stephensen, H. J. T., Svane, A. M., Villanueva, C. B., Goldman, S. A., & Sporring, J. (2021). Measuring Shape Relations Using r-Parallel Sets. Journal of Mathematical Imaging and Vision, 63(8), 1069â\x80\x931083. https://doi.org/10.1007/s10851-021-01041-3\n"
    )
