""" Deals with DOI for references """
import json
import requests
from qim3d.io.logger import log


def _validate_response(response):
    # Check if we got a good response
    if not response.ok:
        log.error(f"Could not read the provided DOI ({response.reason})")
        return False

    return True


def _doi_to_url(doi):
    if doi[:3] != "http":
        url = "https://doi.org/" + doi
    else:
        url = doi

    return url


def _make_request(doi, header):
    # Get url from doi
    url = _doi_to_url(doi)

    # run the request
    response = requests.get(url, headers=header, timeout=10)

    if not _validate_response(response):
        return None

    return response


def _log_and_get_text(doi, header):
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


def get_bibtex(doi):
    """Generates bibtex from doi"""
    header = {"Accept": "application/x-bibtex"}

    return _log_and_get_text(doi, header)


def get_reference(doi):
    """Generates basic reference from doi"""
    header = {"Accept": "text/bibliography"}
    return _log_and_get_text(doi, header)


def cusom_header(doi, header):
    """Allows a custom header to be passed

    For example:
        doi = "https://doi.org/10.1101/2022.11.08.515664"
        header = {"Accept": "text/bibliography"}
        response = qim3d.utils.doi.cusom_header(doi, header)

    """
    return _log_and_get_text(doi, header)

def get_metadata(doi):
    """Generates a metadata dictionary from doi"""
    header = {"Accept": "application/vnd.citationstyles.csl+json"}
    response = _make_request(doi, header)

    metadata = json.loads(response.text)
    return metadata
