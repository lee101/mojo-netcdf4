"""File-backed netCDF4 API with Mojo numeric transforms."""

from __future__ import annotations

from collections.abc import Mapping
import warnings

import netCDF4 as _nc
import numpy as np

from . import _kernels


def _attribute(var, name, dtype, unsigned=False):
    if name not in var.ncattrs():
        return None
    try:
        value = np.asarray(var.getncattr(name), dtype=dtype)
    except (TypeError, ValueError, OverflowError):
        warnings.warn(
            f"{name} cannot be safely interpreted as {dtype}; ignoring it",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    original = np.asarray(var.getncattr(name))
    try:
        if not np.all(
            (value == original)
            | (
                np.issubdtype(value.dtype, np.floating)
                & np.isnan(value)
                & np.isnan(original)
            )
        ):
            warnings.warn(
                f"{name} cannot be safely cast to {dtype}; ignoring it",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
    except TypeError:
        return None
    if unsigned:
        value = value.view(np.dtype(dtype).newbyteorder("=").str.replace("i", "u"))
    return value


def _mask_config(var, raw, unsigned=False):
    dtype = var.dtype
    missing = _attribute(var, "missing_value", dtype, unsigned)
    fill = _attribute(var, "_FillValue", dtype, unsigned)
    if fill is None and dtype.kind in "iuf":
        default_fill = var.get_fill_value()
        if default_fill is None:
            default_fill = _nc.default_fillvals[dtype.str[1:]]
        fill = np.asarray(default_fill, dtype=dtype)
        if unsigned:
            fill = fill.view(np.dtype(dtype).str.replace("i", "u"))

    valid_range = _attribute(var, "valid_range", dtype, unsigned)
    if valid_range is not None and valid_range.size == 2:
        valid_min, valid_max = valid_range.reshape(-1)
    else:
        valid_min = _attribute(var, "valid_min", dtype, unsigned)
        valid_max = _attribute(var, "valid_max", dtype, unsigned)
    scalar = lambda value: None if value is None else np.asarray(value).reshape(-1)[0]
    return {
        "fill": scalar(fill),
        "missing": missing,
        "valid_min": scalar(valid_min),
        "valid_max": scalar(valid_max),
    }


def _fill_for_write(var):
    missing = _attribute(var, "missing_value", var.dtype)
    if missing is not None:
        values = np.asarray(missing).reshape(-1)
        if values.size != 1:
            raise RuntimeError(
                "cannot assign a masked array when missing_value is not scalar"
            )
        return values[0]
    fill = _attribute(var, "_FillValue", var.dtype)
    if fill is not None:
        return np.asarray(fill).reshape(-1)[0]
    default_fill = var.get_fill_value()
    if default_fill is not None:
        return default_fill
    return np.asarray(_nc.default_fillvals[var.dtype.str[1:]], dtype=var.dtype)


def _read_scale_attributes(scale, offset, enabled):
    if not enabled:
        return None, None
    try:
        if scale is not None:
            float(scale)
        if offset is not None:
            float(offset)
    except (TypeError, ValueError):
        warnings.warn(
            "invalid scale_factor or add_offset attribute; no unpacking done",
            RuntimeWarning,
            stacklevel=3,
        )
        return None, None
    return scale, offset


def _full_write_args(var, key, data):
    if key is Ellipsis:
        full = True
    elif isinstance(key, slice):
        full = key == slice(None)
    elif isinstance(key, tuple):
        full = len(key) == var.ndim and all(
            item == slice(None) for item in key
        )
    else:
        full = False
    if not full or np.shape(np.ma.getdata(data)) != var.shape:
        return None
    return (
        np.zeros(var.ndim, dtype=np.intp),
        np.asarray(var.shape, dtype=np.intp),
        np.ones(var.ndim, dtype=np.intp),
    )


class Variable:
    """Covered subset of :class:`netCDF4.Variable`."""

    def __init__(
        self,
        variable,
        *,
        mask=True,
        scale=True,
        always_mask=True,
    ):
        object.__setattr__(self, "_mojo_variable", variable)
        object.__setattr__(self, "_mojo_mask", bool(mask))
        object.__setattr__(self, "_mojo_scale", bool(scale))
        object.__setattr__(self, "_mojo_always_mask", bool(always_mask))
        object.__setattr__(
            self, "_mojo_backend_lsd", bool(getattr(variable, "_has_lsd", False))
        )
        object.__setattr__(self, "_mojo_backend_mask", False)
        object.__setattr__(self, "_mojo_backend_always_mask", True)
        object.__setattr__(
            self, "_mojo_scale_attr", getattr(variable, "scale_factor", None)
        )
        object.__setattr__(
            self, "_mojo_offset_attr", getattr(variable, "add_offset", None)
        )
        object.__setattr__(
            self, "_mojo_unsigned_attr", getattr(variable, "_Unsigned", False)
        )
        variable.set_auto_maskandscale(False)

    def __getattr__(self, name):
        return getattr(self._mojo_variable, name)

    def __setattr__(self, name, value):
        if name.startswith("_mojo_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._mojo_variable, name, value)
            if name == "scale_factor":
                object.__setattr__(self, "_mojo_scale_attr", value)
            elif name == "add_offset":
                object.__setattr__(self, "_mojo_offset_attr", value)
            elif name == "_Unsigned":
                object.__setattr__(self, "_mojo_unsigned_attr", value)

    def setncattr(self, name, value):
        self._mojo_variable.setncattr(name, value)
        if name == "scale_factor":
            object.__setattr__(self, "_mojo_scale_attr", value)
        elif name == "add_offset":
            object.__setattr__(self, "_mojo_offset_attr", value)
        elif name == "_Unsigned":
            object.__setattr__(self, "_mojo_unsigned_attr", value)

    def setncatts(self, attributes):
        self._mojo_variable.setncatts(attributes)
        if "scale_factor" in attributes:
            object.__setattr__(
                self, "_mojo_scale_attr", attributes["scale_factor"]
            )
        if "add_offset" in attributes:
            object.__setattr__(
                self, "_mojo_offset_attr", attributes["add_offset"]
            )
        if "_Unsigned" in attributes:
            object.__setattr__(
                self, "_mojo_unsigned_attr", attributes["_Unsigned"]
            )

    def delncattr(self, name):
        self._mojo_variable.delncattr(name)
        if name == "scale_factor":
            object.__setattr__(self, "_mojo_scale_attr", None)
        elif name == "add_offset":
            object.__setattr__(self, "_mojo_offset_attr", None)
        elif name == "_Unsigned":
            object.__setattr__(self, "_mojo_unsigned_attr", False)

    def __repr__(self):
        return repr(self._mojo_variable)

    def __len__(self):
        return len(self._mojo_variable)

    def __getitem__(self, key):
        if self.dtype.kind not in "iuf":
            return self._mojo_variable[key]
        scale_attr, offset_attr = _read_scale_attributes(
            self._mojo_scale_attr, self._mojo_offset_attr, self._mojo_scale
        )
        unsigned = (
            self._mojo_scale
            and self.dtype.kind == "i"
            and self._mojo_unsigned_attr in ("true", "True")
        )
        if (
            self._mojo_mask
            and scale_attr is None
            and offset_attr is None
            and not unsigned
        ):
            if self._mojo_backend_always_mask != self._mojo_always_mask:
                self._mojo_variable.set_always_mask(self._mojo_always_mask)
                object.__setattr__(
                    self, "_mojo_backend_always_mask", self._mojo_always_mask
                )
            if not self._mojo_backend_mask:
                self._mojo_variable.set_auto_mask(True)
                object.__setattr__(self, "_mojo_backend_mask", True)
            return self._mojo_variable[key]

        if self._mojo_backend_mask:
            self._mojo_variable.set_auto_mask(False)
            object.__setattr__(self, "_mojo_backend_mask", False)
        raw = self._mojo_variable[key]
        scalar_result = isinstance(raw, np.generic)
        if not isinstance(raw, (np.ndarray, np.generic)):
            return raw
        if scalar_result:
            raw = np.asarray(raw)

        if unsigned:
            raw = raw.view(raw.dtype.str.replace("i", "u"))

        config = (
            _mask_config(self._mojo_variable, raw, unsigned)
            if self._mojo_mask
            else {"fill": None, "missing": None, "valid_min": None, "valid_max": None}
        )
        scale = 1.0 if scale_attr is None else float(scale_attr)
        offset = 0.0 if offset_attr is None else float(offset_attr)
        needs_scale = (
            (scale_attr is not None and offset_attr is not None)
            or (scale_attr is not None and scale != 1.0)
            or (offset_attr is not None and offset != 0.0)
        )

        if needs_scale:
            data, mask = _kernels.unpack(raw, scale, offset, **config)
            operand_dtypes = [raw.dtype]
            if scale_attr is not None:
                operand_dtypes.append(np.asarray(scale_attr).dtype)
            if offset_attr is not None:
                operand_dtypes.append(np.asarray(offset_attr).dtype)
            result_dtype = np.result_type(*operand_dtypes)
            if (
                scale_attr is not None
                and offset_attr is not None
                and scale == 1.0
                and offset == 0.0
            ):
                result_dtype = np.asarray(scale_attr).dtype
            data = data.astype(result_dtype, copy=False)
            if mask is not None and np.any(mask):
                np.copyto(data, raw, where=mask, casting="unsafe")
        else:
            data = raw
            mask = _kernels.make_mask(raw, **config) if self._mojo_mask else None

        if self._mojo_mask:
            fill = config["fill"]
            if fill is None:
                fill = self._mojo_variable.get_fill_value()
            if fill is None:
                fill = _nc.default_fillvals[self.dtype.str[1:]]
            masked = np.ma.array(data, mask=mask, fill_value=fill, copy=False)
            if masked.shape == () and bool(np.ma.getmaskarray(masked)):
                return masked[()]
            if scalar_result and needs_scale:
                return np.asarray(data)[()]
            if not self._mojo_always_mask and not np.any(mask):
                return np.asarray(masked)
            return masked
        if scalar_result:
            return np.asarray(data)[()]
        return data

    def __setitem__(self, key, data):
        if self.dtype.kind not in "iuf":
            self._mojo_variable[key] = data
            return

        scale_attr = (
            self._mojo_scale_attr if self._mojo_scale else None
        )
        offset_attr = (
            self._mojo_offset_attr if self._mojo_scale else None
        )
        lsd = getattr(self._mojo_variable, "least_significant_digit", None)
        direct_args = (
            _full_write_args(self._mojo_variable, key, data)
            if lsd is not None
            else None
        )
        if lsd is not None and (
            direct_args is not None
            or not self._mojo_backend_lsd
            or scale_attr is not None
            or offset_attr is not None
        ):
            data = _kernels.quantize(data, int(lsd))

        if scale_attr is not None or offset_attr is not None:
            scale = 1.0 if scale_attr is None else float(scale_attr)
            offset = 0.0 if offset_attr is None else float(offset_attr)
            data = _kernels.pack(
                data, self.dtype, scale, offset, _fill_for_write(self._mojo_variable)
            )
        elif np.ma.isMaskedArray(data):
            data = np.ma.filled(data, _fill_for_write(self._mojo_variable))
        if direct_args is not None:
            self._mojo_variable._put(np.asarray(data), *direct_args)
        else:
            self._mojo_variable[key] = data

    def assignValue(self, value):
        if self.dimensions:
            raise IndexError(
                "to assign values to a non-scalar variable, use a slice"
            )
        self[...] = value

    def getValue(self):
        if self.dimensions:
            raise IndexError(
                "to retrieve values from a non-scalar variable, use slicing"
            )
        return self[slice(None)]

    def set_auto_mask(self, value):
        object.__setattr__(self, "_mojo_mask", bool(value))

    def set_auto_scale(self, value):
        object.__setattr__(self, "_mojo_scale", bool(value))

    def set_auto_maskandscale(self, value):
        self.set_auto_mask(value)
        self.set_auto_scale(value)

    def set_always_mask(self, value):
        object.__setattr__(self, "_mojo_always_mask", bool(value))


class _Variables(Mapping):
    def __init__(self, owner):
        self.owner = owner
        self.cache = {}

    def __iter__(self):
        return iter(self.owner._mojo_container.variables)

    def __len__(self):
        return len(self.owner._mojo_container.variables)

    def __getitem__(self, name):
        wrapped = self.cache.get(name)
        if wrapped is None:
            wrapped = self.owner._wrap_variable(
                self.owner._mojo_container.variables[name]
            )
            self.cache[name] = wrapped
        return wrapped


class _Container:
    def _initialize(self, container, mask=True, scale=True, always_mask=True):
        object.__setattr__(self, "_mojo_container", container)
        object.__setattr__(self, "_mojo_variables", {})
        object.__setattr__(self, "_mojo_variables_view", _Variables(self))
        object.__setattr__(self, "_mojo_groups", {})
        object.__setattr__(self, "_mojo_mask", bool(mask))
        object.__setattr__(self, "_mojo_scale", bool(scale))
        object.__setattr__(self, "_mojo_always_mask", bool(always_mask))

    def __getattr__(self, name):
        return getattr(self._mojo_container, name)

    def __setattr__(self, name, value):
        if name.startswith("_mojo_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._mojo_container, name, value)

    def __repr__(self):
        return repr(self._mojo_container)

    @property
    def variables(self):
        return self._mojo_variables_view

    @property
    def groups(self):
        return {
            name: self._wrap_group(group)
            for name, group in self._mojo_container.groups.items()
        }

    def _wrap_group(self, group):
        key = id(group)
        wrapped = self._mojo_groups.get(key)
        if wrapped is None:
            wrapped = Group(
                group,
                mask=self._mojo_mask,
                scale=self._mojo_scale,
                always_mask=self._mojo_always_mask,
            )
            self._mojo_groups[key] = wrapped
        return wrapped

    def _wrap_variable(self, variable):
        key = id(variable)
        wrapped = self._mojo_variables.get(key)
        if wrapped is None:
            wrapped = Variable(
                variable,
                mask=self._mojo_mask,
                scale=self._mojo_scale,
                always_mask=self._mojo_always_mask,
            )
            self._mojo_variables[key] = wrapped
        return wrapped

    def createDimension(self, dimname, size=None):
        return self._mojo_container.createDimension(dimname, size)

    def createVariable(
        self,
        varname,
        datatype,
        dimensions=(),
        compression=None,
        zlib=False,
        complevel=4,
        shuffle=True,
        szip_coding="nn",
        szip_pixels_per_block=8,
        blosc_shuffle=1,
        fletcher32=False,
        contiguous=False,
        chunksizes=None,
        endian="native",
        least_significant_digit=None,
        significant_digits=None,
        quantize_mode="BitGroom",
        fill_value=None,
        chunk_cache=None,
    ):
        variable = self._mojo_container.createVariable(
            varname,
            datatype,
            dimensions=dimensions,
            compression=compression,
            zlib=zlib,
            complevel=complevel,
            shuffle=shuffle,
            fletcher32=fletcher32,
            contiguous=contiguous,
            chunksizes=chunksizes,
            szip_coding=szip_coding,
            szip_pixels_per_block=szip_pixels_per_block,
            blosc_shuffle=blosc_shuffle,
            endian=endian,
            least_significant_digit=None,
            significant_digits=significant_digits,
            quantize_mode=quantize_mode,
            fill_value=fill_value,
            chunk_cache=chunk_cache,
        )
        if least_significant_digit is not None:
            variable.setncattr(
                "least_significant_digit", least_significant_digit
            )
        return self._wrap_variable(variable)

    def createGroup(self, groupname):
        return self._wrap_group(self._mojo_container.createGroup(groupname))

    def set_auto_mask(self, value):
        object.__setattr__(self, "_mojo_mask", bool(value))
        for variable in self.variables.values():
            variable.set_auto_mask(value)
        for group in self.groups.values():
            group.set_auto_mask(value)

    def set_auto_scale(self, value):
        object.__setattr__(self, "_mojo_scale", bool(value))
        for variable in self.variables.values():
            variable.set_auto_scale(value)
        for group in self.groups.values():
            group.set_auto_scale(value)

    def set_auto_maskandscale(self, value):
        self.set_auto_mask(value)
        self.set_auto_scale(value)

    def set_always_mask(self, value):
        object.__setattr__(self, "_mojo_always_mask", bool(value))
        for variable in self.variables.values():
            variable.set_always_mask(value)
        for group in self.groups.values():
            group.set_always_mask(value)


class Group(_Container):
    def __init__(self, group, *, mask=True, scale=True, always_mask=True):
        self._initialize(group, mask, scale, always_mask)


class Dataset(_Container):
    """Open a netCDF file with a netCDF4-compatible covered API."""

    def __init__(
        self,
        filename,
        mode="r",
        clobber=True,
        format="NETCDF4",
        diskless=False,
        persist=False,
        keepweakref=False,
        memory=None,
        encoding=None,
        parallel=False,
        comm=None,
        info=None,
        auto_complex=False,
        **kwargs,
    ):
        options = dict(
            clobber=clobber,
            format=format,
            diskless=diskless,
            persist=persist,
            keepweakref=keepweakref,
            memory=memory,
            parallel=parallel,
            comm=comm,
            info=info,
            auto_complex=auto_complex,
            **kwargs,
        )
        if encoding is not None:
            options["encoding"] = encoding
        container = _nc.Dataset(filename, mode=mode, **options)
        self._initialize(container)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self):
        return self._mojo_container.close()


default_fillvals = _nc.default_fillvals
date2num = _nc.date2num
num2date = _nc.num2date
date2index = _nc.date2index
chartostring = _nc.chartostring
stringtochar = _nc.stringtochar
getlibversion = _nc.getlibversion
