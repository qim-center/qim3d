""" Deals with DOI for references """
import json
import requests
from qim3d.utils._logger import log


def _validate_response(response: requests.Response) -> bool:
    # Check if we got a good response
    if not response.ok:
        log.error(f"Could not read the provided DOI ({response.reason})")
        return False

    return True


def _doi_to_url(doi: str) -> str:
    if doi[:3] != "http":
        url = "https://doi.org/" + doi
    else:
        url = doi

    return url


def _make_request(doi: str, header: str) -> requests.Response:
    # Get url from doi
    url = _doi_to_url(doi)

    # run the request
    response = requests.get(url, headers=header, timeout=10)

    if not _validate_response(response):
        return None

    return response


def _log_and_get_text(doi, header) -> str:
    response = _make_request(doi, header)

    if response and response.encoding:
        # Explicitly decode the response content using the specified encoding
        text = response.content.decode(response.encoding)
        log.info(text)
        return text
    elif response:
        # If encoding is not specified, default to UTF-8
        text = response.text
        log.info(text)
        return text


def get_bibtex(doi: str):
    """Generates bibtex from doi"""
    header = {"Accept": "application/x-bibtex"}

    return _log_and_get_text(doi, header)

def custom_header(doi: str, header: str) -> str:
    """Allows a custom header to be passed

    Example:
        import qim3d
        doi = "https://doi.org/10.1101/2022.11.08.515664"
        header = {"Accept": "text/bibliography"}
        custom_header = qim3d.utils.custom_header(doi, header)

    """
    return _log_and_get_text(doi, header)

def get_metadata(doi: str) -> dict:
    """Generates a metadata dictionary from doi"""
    header = {"Accept": "application/vnd.citationstyles.csl+json"}
    response = _make_request(doi, header)

    metadata = json.loads(response.text)

    return metadata

def get_reference(doi: str) -> str:
    """Generates a metadata dictionary from doi and use it to build a reference string"""

    metadata = get_metadata(doi)
    reference_string = build_reference_string(metadata)

    return reference_string

def build_reference_string(metadata: dict) -> str:
    """Generates a reference string from metadata"""
    authors = ", ".join([f"{author['family']} {author['given']}" for author in metadata['author']])
    year = metadata['issued']['date-parts'][0][0]
    title = metadata['title']
    publisher = metadata['publisher']
    url = metadata['URL']
    doi = metadata['DOI']

    reference_string = f"{authors} ({year}). {title}. {publisher} ({url}). DOI: {doi}"

    return reference_string
