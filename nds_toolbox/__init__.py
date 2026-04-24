"""Neural Data Science toolbox.

Install with ``pip install nds-toolbox`` and import with ``import nds_toolbox``.
"""

from importlib import metadata as _metadata
from pathlib import Path as _Path
import json as _json
from urllib.parse import unquote as _unquote
from urllib.parse import urlparse as _urlparse


_DISTRIBUTION_NAME = "nds-toolbox"
_FALLBACK_VERSION = "0.1.5"


def _is_relative_to(path, parent):
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _distribution_matches_import_path(distribution):
    package_path = _Path(__file__).resolve()

    try:
        dist_root = _Path(distribution.locate_file("")).resolve()
    except Exception:
        dist_root = None

    if dist_root is not None and _is_relative_to(package_path, dist_root):
        return True

    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        return False

    try:
        url = _json.loads(direct_url).get("url")
    except (_json.JSONDecodeError, TypeError):
        return False

    parsed = _urlparse(url)
    if parsed.scheme != "file":
        return False

    source_root = _Path(_unquote(parsed.path)).resolve()
    return _is_relative_to(package_path, source_root)


def _get_version():
    try:
        distribution = _metadata.distribution(_DISTRIBUTION_NAME)
    except _metadata.PackageNotFoundError:
        return _FALLBACK_VERSION

    if not _distribution_matches_import_path(distribution):
        return _FALLBACK_VERSION

    return distribution.version


__version__ = _get_version()
