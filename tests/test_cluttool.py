import gzip
from array import array
from pathlib import Path

import png
import pytest

import cluttool.cluttool as cluttool_module
from cluttool.cluttool import ColorLUT

TESTS_DIR = Path(__file__).parent
GOLDENGATE_PNG = TESTS_DIR / 'goldengate.png'
BOURBON_CUBE_GZ = TESTS_DIR / 'Bourbon 64.cube.gz'


@pytest.fixture
def bourbon_cube_path(tmp_path):
    dest = tmp_path / 'bourbon.cube'
    dest.write_bytes(gzip.decompress(BOURBON_CUBE_GZ.read_bytes()))
    return str(dest)


def write_identity_cube(path: str, sample_count: int) -> None:
    with open(path, 'w') as f:
        f.write(f'LUT_3D_SIZE {sample_count}\n')
        for b in range(sample_count):
            for g in range(sample_count):
                for r in range(sample_count):
                    f.write(
                        f'{r / (sample_count - 1):.6f} {g / (sample_count - 1):.6f} {b / (sample_count - 1):.6f}\n'
                    )


def read_3dl(path):
    """Read a 3DL file into (sample_intervals, [(r, g, b), ...])."""
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    intervals = [int(v) for v in lines[0].split()]
    rows = [tuple(int(v) for v in line.split()) for line in lines[1:]]
    return intervals, rows


def read_cube_data(path):
    """Read the flat RGB data rows from a cube file."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.split()[0].upper() in (
                'TITLE',
                'LUT_3D_SIZE',
                'DOMAIN_MIN',
                'DOMAIN_MAX',
            ):
                continue
            data.append(tuple(float(v) for v in line.split()))
    return data


def test_from_haldclut() -> None:
    clut = ColorLUT.from_haldclut(str(GOLDENGATE_PNG))
    # 512x512 Hald CLUT is level 8 -> 8**2 == 64 samples per axis.
    assert clut.sample_count == 64
    assert clut.input_domain == 255
    assert len(clut.data) == 3 * 64**3


def test_interpolation_at_grid_points_is_exact() -> None:
    # Interpolating at an exact sample position must return that sample's stored
    # value. Regression test: the lower index and the fractional weight must be
    # derived consistently, or rounding shifts the result by a whole sample.
    clut = ColorLUT.from_haldclut(str(GOLDENGATE_PNG))
    sd = clut.sample_distance
    n = clut.sample_count
    for r, g, b in ((0, 0, 0), (n - 1, n - 1, n - 1), (59, 54, 35), (23, 62, 62)):
        stored = clut.get_color_value_from_index(r, g, b)
        interpolated = clut.get_interpolated_color_value(r * sd, g * sd, b * sd)
        assert tuple(stored) == pytest.approx(tuple(interpolated), abs=1e-9)


def test_from_cube(bourbon_cube_path) -> None:
    clut = ColorLUT.from_cube(bourbon_cube_path)
    assert clut.sample_count == 32
    assert clut.input_domain == 1.0
    assert len(clut.data) == 3 * 32**3
    # First data row of the fixture.
    assert clut.data[0:3] == array('f', (0.038818, 0.0, 0.065033))


def test_from_cube_missing_size(tmp_path) -> None:
    path = tmp_path / 'no_size.cube'
    path.write_text('0.0 0.0 0.0\n1.0 1.0 1.0\n')
    with pytest.raises(ValueError, match='LUT_3D_SIZE'):
        ColorLUT.from_cube(str(path))


def test_from_cube_rejects_1d(tmp_path) -> None:
    path = tmp_path / 'one_d.cube'
    path.write_text('LUT_1D_SIZE 16\n')
    with pytest.raises(ValueError, match='1D'):
        ColorLUT.from_cube(str(path))


def test_write_haldclut_requires_level_for_non_square(
    bourbon_cube_path, tmp_path
) -> None:
    clut = ColorLUT.from_cube(bourbon_cube_path)
    # 32 is not a perfect square, so a level cannot be inferred.
    with pytest.raises(ValueError, match='perfect square'):
        clut.write_haldclut(str(tmp_path / 'out.png'))


def test_write_haldclut_dimensions(bourbon_cube_path, tmp_path) -> None:
    clut = ColorLUT.from_cube(bourbon_cube_path)
    dest = tmp_path / 'bourbon.png'
    clut.write_haldclut(str(dest), level=8)
    width, height, _, meta = png.Reader(filename=str(dest)).read()
    assert (width, height) == (512, 512)
    assert meta['bitdepth'] == 8
    assert not meta['alpha']
    assert not meta['greyscale']


def test_cube_haldclut_identity_roundtrip(tmp_path) -> None:
    # A level-4 Hald CLUT has 16 samples per axis (a perfect square), so the
    # cube -> Hald CLUT -> cube round trip is lossless apart from 8-bit quant.
    src_cube = tmp_path / 'identity.cube'
    write_identity_cube(str(src_cube), 16)

    png_path = tmp_path / 'identity.png'
    ColorLUT.from_cube(str(src_cube)).write_haldclut(str(png_path))

    out_cube = tmp_path / 'roundtrip.cube'
    ColorLUT.from_haldclut(str(png_path)).write_cube(str(out_cube))

    expected = read_cube_data(str(src_cube))
    actual = read_cube_data(str(out_cube))
    assert len(actual) == len(expected) == 16**3
    max_error = max(
        abs(e - a) for erow, arow in zip(expected, actual) for e, a in zip(erow, arow)
    )
    assert max_error <= 1 / 255 + 1e-9


def test_measure_error_identity_is_zero() -> None:
    # A LUT measured against itself has no resampling or quantization error.
    clut = ColorLUT.from_haldclut(str(GOLDENGATE_PNG))
    error = clut.measure_error_against(clut)
    assert error.max == pytest.approx(0.0, abs=1e-9)
    assert error.rms == pytest.approx(0.0, abs=1e-9)


def test_measure_error_decreases_with_level(bourbon_cube_path, tmp_path) -> None:
    clut = ColorLUT.from_cube(bourbon_cube_path)

    def error_at(level):
        dest = tmp_path / f'lvl{level}.png'
        clut.write_haldclut(str(dest), level=level)
        written = ColorLUT.from_haldclut(str(dest))
        return written.measure_error_against(clut).rms

    # More samples -> a closer reproduction of the source LUT.
    assert error_at(6) < error_at(3) < error_at(2)


def test_find_haldclut_level_picks_smallest_within_target(bourbon_cube_path) -> None:
    clut = ColorLUT.from_cube(bourbon_cube_path)
    level, error, satisfied, _ = clut.find_haldclut_level(0.01)
    assert satisfied
    assert error.max <= 0.01
    # A tighter target requires at least as many samples.
    tighter_level, _, tighter_ok, _ = clut.find_haldclut_level(0.0025)
    assert tighter_ok
    assert tighter_level >= level


def test_find_haldclut_level_unreachable_returns_best(bourbon_cube_path) -> None:
    clut = ColorLUT.from_cube(bourbon_cube_path)
    # Below the 8-bit quantization floor, so unreachable.
    level, error, satisfied, _ = clut.find_haldclut_level(0.0001)
    assert not satisfied
    assert level >= 2
    assert error.max > 0.0001


def test_write_haldclut_max_error_output_is_within_target(
    bourbon_cube_path, tmp_path
) -> None:
    clut = ColorLUT.from_cube(bourbon_cube_path)
    dest = tmp_path / 'auto.png'
    clut.write_haldclut(str(dest), max_error=0.01)
    written = ColorLUT.from_haldclut(str(dest))
    assert written.measure_error_against(clut).max <= 0.01


numpy_required = pytest.mark.skipif(
    cluttool_module.np is None, reason='numpy is not installed'
)


@pytest.fixture
def no_numpy(monkeypatch):
    """Force the pure-Python fallback so it is exercised even when numpy is installed."""
    monkeypatch.setattr(cluttool_module, 'np', None)


@numpy_required
def test_numpy_and_scalar_render_agree(bourbon_cube_path, monkeypatch) -> None:
    clut = ColorLUT.from_cube(bourbon_cube_path)
    numpy_data = clut.render_haldclut(4, 8)[0]
    monkeypatch.setattr(cluttool_module, 'np', None)
    scalar_data = clut.render_haldclut(4, 8)[0]
    assert list(numpy_data) == list(scalar_data)


@numpy_required
def test_numpy_and_scalar_error_agree(bourbon_cube_path, monkeypatch) -> None:
    clut = ColorLUT.from_cube(bourbon_cube_path)
    small = ColorLUT(clut.render_haldclut(4, 8)[0], sample_count=16, input_domain=255)
    numpy_error = small.measure_error_against(clut)
    monkeypatch.setattr(cluttool_module, 'np', None)
    scalar_error = small.measure_error_against(clut)
    assert numpy_error.max == pytest.approx(scalar_error.max, abs=1e-9)
    assert numpy_error.rms == pytest.approx(scalar_error.rms, abs=1e-9)


def test_write_haldclut_16_bit(bourbon_cube_path, tmp_path) -> None:
    clut = ColorLUT.from_cube(bourbon_cube_path)
    dest = tmp_path / 'bourbon16.png'
    clut.write_haldclut(str(dest), level=8, bit_depth=16)
    _, _, _, meta = png.Reader(filename=str(dest)).read()
    assert meta['bitdepth'] == 16


def test_write_3dl_structure(bourbon_cube_path, tmp_path) -> None:
    clut = ColorLUT.from_cube(bourbon_cube_path)
    dest = tmp_path / 'bourbon.3dl'
    clut.write_3dl(str(dest))
    intervals, rows = read_3dl(str(dest))
    # The header has one input sample interval per axis position, spanning the
    # full 10-bit output domain.
    assert len(intervals) == clut.sample_count
    assert intervals[0] == 0
    assert intervals[-1] == 1023
    assert intervals == sorted(intervals)
    # One output triple per grid point, each within the output domain.
    assert len(rows) == clut.sample_count**3
    assert all(0 <= c <= 1023 for row in rows for c in row)


def test_write_3dl_identity_values_blue_fastest(tmp_path) -> None:
    # 3DL stores the output values with the blue axis varying fastest, then
    # green, then red.  For an identity LUT each row should reproduce its own
    # grid coordinate, scaled to the 10-bit output domain.
    src_cube = tmp_path / 'identity.cube'
    write_identity_cube(str(src_cube), 16)
    dest = tmp_path / 'identity.3dl'
    ColorLUT.from_cube(str(src_cube)).write_3dl(str(dest))

    _, rows = read_3dl(str(dest))
    n = 16
    assert len(rows) == n**3
    for r, g, b in ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (15, 15, 15)):
        row = rows[r * n * n + g * n + b]
        expected = tuple(round(c / (n - 1) * 1023) for c in (r, g, b))
        assert max(abs(a - e) for a, e in zip(row, expected)) <= 1


def test_from_cube_rejects_short_data_line(tmp_path) -> None:
    # A data line with fewer than 3 tokens would otherwise shift every
    # subsequent channel by one without an error.
    path = tmp_path / 'short.cube'
    path.write_text('LUT_3D_SIZE 2\n0.5 0.5\n')
    with pytest.raises(ValueError, match='3 values'):
        ColorLUT.from_cube(str(path))


def test_from_cube_rejects_truncated_size(tmp_path) -> None:
    path = tmp_path / 'bad_size.cube'
    path.write_text('LUT_3D_SIZE\n')
    with pytest.raises(ValueError, match='LUT_3D_SIZE'):
        ColorLUT.from_cube(str(path))


def test_from_cube_rejects_truncated_domain(tmp_path) -> None:
    path = tmp_path / 'bad_domain.cube'
    path.write_text('LUT_3D_SIZE 2\nDOMAIN_MAX 1.0\n')
    with pytest.raises(ValueError, match='DOMAIN_MAX'):
        ColorLUT.from_cube(str(path))


def test_validate_rejects_empty_data() -> None:
    # Empty data must give a clean ValueError, not an IndexError from data[0].
    with pytest.raises(ValueError, match='flat list of numbers'):
        ColorLUT(array('f'), sample_count=2, input_domain=1.0)


def test_validate_rejects_zero_input_domain() -> None:
    # A zero input domain would make sample_distance 0 and divide by zero later.
    data = array('f', [0.0] * (3 * 2**3))
    with pytest.raises(ValueError, match='positive'):
        ColorLUT(data, sample_count=2, input_domain=0.0)


def test_scalar_render_clamps_out_of_gamut(no_numpy) -> None:
    # Out-of-gamut stored values must be clamped (like the numpy path) instead
    # of overflowing array('B'); runs on the default pure-Python path.
    sc = 2
    data = array('f', [1.05, -0.02, 0.5] * sc**3)
    clut = ColorLUT(data, sample_count=sc, input_domain=1.0)
    rendered, width, height, output_domain = clut.render_haldclut(2, 8)
    assert output_domain == 255
    assert all(0 <= v <= 255 for v in rendered)
    assert max(rendered) == 255  # 1.05 clamped down to the ceiling
    assert min(rendered) == 0  # -0.02 clamped up to the floor


def test_scalar_measure_error_identity_is_zero(no_numpy) -> None:
    # Exercises the pure-Python measure path; a LUT vs itself has zero error.
    sc = 4
    data = array('f', [((i * 37) % 101) / 100.0 for i in range(3 * sc**3)])
    clut = ColorLUT(data, sample_count=sc, input_domain=1.0)
    error = clut.measure_error_against(clut)
    assert error.max == pytest.approx(0.0, abs=1e-9)
    assert error.rms == pytest.approx(0.0, abs=1e-9)


@numpy_required
def test_numpy_and_scalar_render_agree_blue_fastest(monkeypatch) -> None:
    # A blue-fastest data layout must be honored identically by both paths; a
    # hardcoded red-fastest reshape would transpose the R/B channels.
    sc = 4
    data = array('f', [((i * 37) % 101) / 100.0 for i in range(3 * sc**3)])
    clut = ColorLUT(
        data, sample_count=sc, input_domain=1.0, red_increments_fastest=False
    )
    numpy_data = clut.render_haldclut(2, 8)[0]
    monkeypatch.setattr(cluttool_module, 'np', None)
    scalar_data = clut.render_haldclut(2, 8)[0]
    assert list(numpy_data) == list(scalar_data)


@numpy_required
def test_numpy_and_scalar_measure_agree_blue_fastest(monkeypatch) -> None:
    # The numpy measure path must honor a blue-fastest reference's layout just
    # like the scalar path's r/b swap does.
    sc = 4
    data = array('f', [((i * 37) % 101) / 100.0 for i in range(3 * sc**3)])
    reference = ColorLUT(
        data, sample_count=sc, input_domain=1.0, red_increments_fastest=False
    )
    self_lut = ColorLUT(data, sample_count=sc, input_domain=1.0)
    numpy_error = self_lut.measure_error_against(reference)
    monkeypatch.setattr(cluttool_module, 'np', None)
    scalar_error = self_lut.measure_error_against(reference)
    assert numpy_error.max == pytest.approx(scalar_error.max, abs=1e-9)
    assert numpy_error.rms == pytest.approx(scalar_error.rms, abs=1e-9)
