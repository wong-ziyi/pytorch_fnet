"""
Lightweight subset of the :mod:`aicsimageio` API needed for the test suite.

The original project pulls in dependencies that currently do not provide
pre-built wheels for Python 3.13 (for example ``lxml<5``).  To keep the test
environment self-contained we implement the small surface area exercised by
``pytorch_fnet`` instead of depending on the upstream package.
"""

from .simple_image import AICSImage  # noqa: F401
from .writers import OmeTiffWriter  # noqa: F401

__all__ = ["AICSImage", "OmeTiffWriter"]
