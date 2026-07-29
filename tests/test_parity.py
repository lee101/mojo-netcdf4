"""Behavioral and numerical parity with netCDF4 1.7."""

from __future__ import annotations

import datetime

import netCDF4 as nc
import numpy as np
import pytest

import mojonetcdf4 as mnc


def assert_array_equal(ours, theirs):
    assert np.ma.isMaskedArray(ours) == np.ma.isMaskedArray(theirs)
    if np.ma.isMaskedArray(ours):
        assert np.array_equal(
            np.ma.getmaskarray(ours), np.ma.getmaskarray(theirs)
        )
        assert np.allclose(
            np.ma.getdata(ours),
            np.ma.getdata(theirs),
            equal_nan=True,
        )
    else:
        assert np.array_equal(ours, theirs)
    assert np.asarray(ours).dtype == np.asarray(theirs).dtype


def create_plain(api, path):
    with api.Dataset(path, "w") as dataset:
        dataset.createDimension("time", None)
        dataset.createDimension("station", 4)
        variable = dataset.createVariable(
            "observation",
            "f8",
            ("time", "station"),
            compression="zlib",
            complevel=1,
        )
        variable.units = "m s-1"
        variable[0:3, :] = np.arange(12, dtype=np.float64).reshape(3, 4)
        variable[3, :] = [12.0, 13.0, 14.0, 15.0]


def test_plain_file_slicing_and_metadata(tmp_path):
    ours_path = tmp_path / "ours.nc"
    ref_path = tmp_path / "reference.nc"
    create_plain(mnc, ours_path)
    create_plain(nc, ref_path)
    with mnc.Dataset(ours_path) as ours, nc.Dataset(ref_path) as theirs:
        ov = ours.variables["observation"]
        tv = theirs.variables["observation"]
        assert ov.shape == tv.shape == (4, 4)
        assert ov.dimensions == tv.dimensions
        assert ov.units == tv.units
        assert ov.filters() == tv.filters()
        assert_array_equal(ov[:], tv[:])
        assert_array_equal(ov[1:4:2, 1:], tv[1:4:2, 1:])
        assert_array_equal(ov[[3, 1], [2, 0]], tv[[3, 1], [2, 0]])
    with mnc.Dataset(ours_path, "r+") as ours, nc.Dataset(
        ref_path, "r+"
    ) as theirs:
        selection = ([3, 1], [2, 0])
        values = np.array([[101.0, 102.0], [103.0, 104.0]])
        ours.variables["observation"][selection] = values
        theirs.variables["observation"][selection] = values
        assert_array_equal(
            ours.variables["observation"][:],
            theirs.variables["observation"][:],
        )


@pytest.mark.parametrize(
    "dtype",
    ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f4", "f8"],
)
def test_primitive_dtype_parity(tmp_path, dtype):
    ours_path = tmp_path / f"ours-{dtype}.nc"
    ref_path = tmp_path / f"ref-{dtype}.nc"
    values = np.arange(8, dtype=dtype)
    for api, path in ((mnc, ours_path), (nc, ref_path)):
        with api.Dataset(path, "w") as dataset:
            dataset.createDimension("x", 8)
            dataset.createVariable("v", dtype, ("x",))[:] = values
    with mnc.Dataset(ours_path) as ours, nc.Dataset(ref_path) as theirs:
        assert_array_equal(ours.variables["v"][:], theirs.variables["v"][:])


def create_scaled(api, path):
    with api.Dataset(path, "w") as dataset:
        dataset.createDimension("x", 8)
        variable = dataset.createVariable(
            "temperature", "i2", ("x",), fill_value=-9999
        )
        variable.scale_factor = np.float32(0.125)
        variable.add_offset = np.float32(250.0)
        variable.valid_range = np.array([-800, 800], dtype=np.int16)
        values = np.ma.array(
            [249.0, 250.0, 251.0, 252.5, 300.0, 400.0, 248.0, 260.0],
            mask=[0, 0, 1, 0, 0, 0, 0, 0],
        )
        variable[:] = values


def test_scaled_masked_read_write_parity(tmp_path):
    ours_path = tmp_path / "ours-scaled.nc"
    ref_path = tmp_path / "ref-scaled.nc"
    create_scaled(mnc, ours_path)
    create_scaled(nc, ref_path)
    with mnc.Dataset(ours_path) as ours, nc.Dataset(ref_path) as theirs:
        ov = ours.variables["temperature"]
        tv = theirs.variables["temperature"]
        assert_array_equal(ov[:], tv[:])
        assert type(ov[0]) is type(tv[0])
        assert ov[0] == tv[0]
        assert ov[2] is np.ma.masked and tv[2] is np.ma.masked
        assert np.asarray(ov[:]).dtype == np.dtype("f4")
        ov.set_auto_maskandscale(False)
        tv.set_auto_maskandscale(False)
        assert np.array_equal(ov[:], tv[:])


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("scale_factor", np.float32(0.5)),
        ("scale_factor", np.float64(0.5)),
        ("add_offset", np.float32(10.0)),
        ("add_offset", np.float64(10.0)),
    ],
)
def test_single_scale_offset_attribute(tmp_path, attribute, value):
    paths = [tmp_path / "ours.nc", tmp_path / "ref.nc"]
    for api, path in ((mnc, paths[0]), (nc, paths[1])):
        with api.Dataset(path, "w") as dataset:
            dataset.createDimension("x", 4)
            variable = dataset.createVariable("v", "i2", ("x",))
            setattr(variable, attribute, value)
            variable[:] = np.array([10.0, 11.0, 12.0, 13.0])
    with mnc.Dataset(paths[0]) as ours, nc.Dataset(paths[1]) as theirs:
        assert_array_equal(ours.variables["v"][:], theirs.variables["v"][:])


def test_invalid_scale_attribute_is_ignored_on_read(tmp_path):
    paths = [tmp_path / "ours.nc", tmp_path / "ref.nc"]
    for api, path in ((mnc, paths[0]), (nc, paths[1])):
        with api.Dataset(path, "w") as dataset:
            dataset.createDimension("x", 2)
            variable = dataset.createVariable("v", "i2", ("x",))
            variable.set_auto_maskandscale(False)
            variable[:] = [1, 2]
            variable.scale_factor = "invalid"
    with pytest.warns((RuntimeWarning, UserWarning)):
        with mnc.Dataset(paths[0]) as ours:
            ours_data = ours.variables["v"][:]
    with pytest.warns(UserWarning):
        with nc.Dataset(paths[1]) as theirs:
            their_data = theirs.variables["v"][:]
    assert_array_equal(ours_data, their_data)


def test_missing_values_and_valid_limits(tmp_path):
    paths = [tmp_path / "ours.nc", tmp_path / "ref.nc"]
    raw = np.array([-99, -88, -2, 0, 5, 11], dtype=np.int16)
    for api, path in ((mnc, paths[0]), (nc, paths[1])):
        with api.Dataset(path, "w") as dataset:
            dataset.createDimension("x", raw.size)
            variable = dataset.createVariable("v", "i2", ("x",), fill_value=-99)
            variable.missing_value = np.array([-88, -77], dtype=np.int16)
            variable.valid_min = np.int16(-1)
            variable.valid_max = np.int16(10)
            variable.set_auto_maskandscale(False)
            variable[:] = raw
    with mnc.Dataset(paths[0]) as ours, nc.Dataset(paths[1]) as theirs:
        assert_array_equal(ours.variables["v"][:], theirs.variables["v"][:])


def test_float_nan_fill_value(tmp_path):
    paths = [tmp_path / "ours.nc", tmp_path / "ref.nc"]
    values = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    for api, path in ((mnc, paths[0]), (nc, paths[1])):
        with api.Dataset(path, "w") as dataset:
            dataset.createDimension("x", 3)
            variable = dataset.createVariable(
                "v", "f4", ("x",), fill_value=np.nan
            )
            variable[:] = values
    with mnc.Dataset(paths[0]) as ours, nc.Dataset(paths[1]) as theirs:
        assert_array_equal(ours.variables["v"][:], theirs.variables["v"][:])


def test_fill_disabled_matches_upstream_default_masking(tmp_path):
    paths = [tmp_path / "ours.nc", tmp_path / "ref.nc"]
    values = np.array([-32767, 1], dtype=np.int16)
    for api, path in ((mnc, paths[0]), (nc, paths[1])):
        with api.Dataset(path, "w") as dataset:
            dataset.createDimension("x", 2)
            variable = dataset.createVariable(
                "v", "i2", ("x",), fill_value=False
            )
            variable[:] = values
    with mnc.Dataset(paths[0]) as ours, nc.Dataset(paths[1]) as theirs:
        assert_array_equal(ours.variables["v"][:], theirs.variables["v"][:])


@pytest.mark.parametrize("dtype", ["i1", "u1"])
def test_default_byte_fill_is_masked_during_scaled_read(tmp_path, dtype):
    paths = [tmp_path / "ours.nc", tmp_path / "ref.nc"]
    raw = np.array([nc.default_fillvals[dtype], 2], dtype=dtype)
    for api, path in ((mnc, paths[0]), (nc, paths[1])):
        with api.Dataset(path, "w") as dataset:
            dataset.createDimension("x", 2)
            variable = dataset.createVariable("v", dtype, ("x",))
            variable.set_auto_maskandscale(False)
            variable[:] = raw
            variable.scale_factor = np.float32(0.5)
    with mnc.Dataset(paths[0]) as ours, nc.Dataset(paths[1]) as theirs:
        assert_array_equal(ours.variables["v"][:], theirs.variables["v"][:])


def test_auto_mask_scale_and_always_mask(tmp_path):
    path = tmp_path / "modes.nc"
    with mnc.Dataset(path, "w") as dataset:
        dataset.createDimension("x", 3)
        variable = dataset.createVariable("v", "i2", ("x",))
        variable.scale_factor = 0.25
        variable[:] = [1.0, 2.0, 3.0]
    with mnc.Dataset(path) as dataset:
        variable = dataset.variables["v"]
        assert np.ma.isMaskedArray(variable[:])
        variable.set_always_mask(False)
        assert not np.ma.isMaskedArray(variable[:])
        variable.set_auto_mask(False)
        assert np.allclose(variable[:], [1.0, 2.0, 3.0])
        variable.set_auto_scale(False)
        assert np.array_equal(variable[:], [4, 8, 12])


def test_least_significant_digit_quantization(tmp_path):
    paths = [tmp_path / "ours.nc", tmp_path / "ref.nc"]
    values = np.array(
        [np.pi, np.e, -1.234567, 100.009, 0.0001], dtype=np.float64
    )
    for api, path in ((mnc, paths[0]), (nc, paths[1])):
        with api.Dataset(path, "w") as dataset:
            dataset.createDimension("x", values.size)
            variable = dataset.createVariable(
                "v", "f8", ("x",), least_significant_digit=3
            )
            variable[:] = values
    updated = values[::-1] * 1.125
    with mnc.Dataset(paths[0], "r+") as ours, nc.Dataset(paths[1], "r+") as theirs:
        assert ours.variables["v"]._mojo_backend_lsd
        ours.variables["v"][:] = updated
        theirs.variables["v"][:] = updated
    with mnc.Dataset(paths[0]) as ours, nc.Dataset(paths[1]) as theirs:
        assert_array_equal(ours.variables["v"][:], theirs.variables["v"][:])


def test_unsigned_convention(tmp_path):
    paths = [tmp_path / "ours.nc", tmp_path / "ref.nc"]
    raw = np.array([-1, 0, 1, -32768], dtype=np.int16)
    for api, path in ((mnc, paths[0]), (nc, paths[1])):
        with api.Dataset(path, "w") as dataset:
            dataset.createDimension("x", 4)
            variable = dataset.createVariable("v", "i2", ("x",))
            variable.setncattr("_Unsigned", "true")
            variable.set_auto_maskandscale(False)
            variable[:] = raw
    with mnc.Dataset(paths[0]) as ours, nc.Dataset(paths[1]) as theirs:
        assert_array_equal(ours.variables["v"][:], theirs.variables["v"][:])


def test_scalar_variable_assign_and_get(tmp_path):
    paths = [tmp_path / "ours.nc", tmp_path / "ref.nc"]
    for api, path in ((mnc, paths[0]), (nc, paths[1])):
        with api.Dataset(path, "w") as dataset:
            variable = dataset.createVariable("answer", "i4")
            variable.assignValue(42)
    with mnc.Dataset(paths[0]) as ours, nc.Dataset(paths[1]) as theirs:
        assert ours.variables["answer"].getValue() == theirs.variables[
            "answer"
        ].getValue()


def test_groups_attributes_and_context_manager(tmp_path):
    path = tmp_path / "groups.nc"
    with mnc.Dataset(path, "w") as dataset:
        dataset.title = "stations"
        group = dataset.createGroup("forecast")
        group.createDimension("x", 2)
        variable = group.createVariable("rain", "f4", ("x",))
        variable.long_name = "rainfall"
        variable[:] = [0.5, 1.25]
    with mnc.Dataset(path) as dataset:
        assert dataset.title == "stations"
        variable = dataset.groups["forecast"].variables["rain"]
        assert variable.long_name == "rainfall"
        assert np.allclose(variable[:], [0.5, 1.25])
        dataset.set_auto_maskandscale(False)
        assert not np.ma.isMaskedArray(
            dataset.groups["forecast"].variables["rain"][:]
        )


def test_character_variable_passthrough(tmp_path):
    paths = [tmp_path / "ours.nc", tmp_path / "ref.nc"]
    names = np.array(["alpha", "beta"], dtype="S5")
    for api, path in ((mnc, paths[0]), (nc, paths[1])):
        with api.Dataset(path, "w") as dataset:
            dataset.createDimension("row", 2)
            dataset.createDimension("name_strlen", 5)
            variable = dataset.createVariable("name", "S1", ("row", "name_strlen"))
            variable._Encoding = "ascii"
            variable[:] = names
    with mnc.Dataset(paths[0]) as ours, nc.Dataset(paths[1]) as theirs:
        assert_array_equal(ours.variables["name"][:], theirs.variables["name"][:])


def test_date_helpers_match_upstream():
    dates = [
        datetime.datetime(2000, 1, 1),
        datetime.datetime(2000, 1, 2, 12),
    ]
    units = "hours since 2000-01-01 00:00:00"
    assert np.array_equal(mnc.date2num(dates, units), nc.date2num(dates, units))
    assert np.array_equal(
        mnc.num2date([0, 36], units), nc.num2date([0, 36], units)
    )
    strings = np.array(["alpha", "beta"], dtype="U5")
    characters = nc.stringtochar(strings)
    assert np.array_equal(mnc.stringtochar(strings), characters)
    assert np.array_equal(mnc.chartostring(characters), strings)


def test_date2index_matches_upstream(tmp_path):
    path = tmp_path / "dates.nc"
    units = "hours since 2000-01-01 00:00:00"
    with nc.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 3)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = units
        time[:] = [0, 24, 48]
    target = datetime.datetime(2000, 1, 2)
    with nc.Dataset(path) as dataset:
        expected = nc.date2index(target, dataset.variables["time"])
    with mnc.Dataset(path) as dataset:
        actual = mnc.date2index(
            target, dataset.variables["time"]._mojo_variable
        )
    assert actual == expected
