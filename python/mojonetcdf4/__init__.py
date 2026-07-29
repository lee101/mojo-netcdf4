"""Mojo-accelerated netCDF variable read/write."""

from netCDF4 import __version__ as upstream_version

from ._api import (
    Dataset,
    Group,
    Variable,
    chartostring,
    date2index,
    date2num,
    default_fillvals,
    getlibversion,
    num2date,
    stringtochar,
)

__version__ = "0.1.0"
__all__ = [
    "Dataset",
    "Group",
    "Variable",
    "chartostring",
    "date2index",
    "date2num",
    "default_fillvals",
    "getlibversion",
    "num2date",
    "stringtochar",
]
