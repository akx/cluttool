#!/usr/bin/env python3
#
# Copyright (C) 2018 Troy Sankey
# Copyright (C) 2026 Aarni Koskela
# This file is released under the GNU GPL, version 3 or a later revision.
# For further details see the COPYING file
#
# References:
# - Hald CLUT reference: http://www.quelsolaar.com/technology/clut.html
# - 3D LUT (3DL) reference: http://download.autodesk.com/us/systemdocs/pdf/lustre_color_management_user_guide.pdf#page=14

from __future__ import annotations

import numbers
from array import array
from collections.abc import Iterator
from dataclasses import dataclass
from os import PathLike
from typing import TYPE_CHECKING, SupportsFloat

if TYPE_CHECKING:
    from typing import Self  # Python 3.11+

import png

try:
    import numpy as np
except ImportError:  # numpy is an optional acceleration dependency
    np = None  # type: ignore


class ValidationError(ValueError):
    pass


def is_perfect_six_root(n) -> bool:
    c = int(n ** (1 / 6.0))
    return (c**6 == n) or ((c + 1) ** 6 == n)


def write_png(
    path: str | PathLike, data, width: int, height: int, bit_depth: int
) -> None:
    """
    Write a flat array of RGB samples to a PNG file.
    """
    writer = png.Writer(
        width=width,
        height=height,
        bitdepth=bit_depth,
        greyscale=False,
        alpha=False,
    )
    with open(path, 'wb') as destfile:
        writer.write_array(destfile, data)


def uniform_intervals(end: float, samples, floating_point: bool = False):
    """
    Make `samples` uniformly distributed numbers from 0 to `end`.
    """
    dist = end / float(samples - 1)
    values = [dist * i for i in range(samples)]
    if not floating_point:
        values = [int(round(v)) for v in values]
        for idx in range(1, samples):
            actual_dist = values[idx] - values[idx - 1]
            error_frac = abs(float(actual_dist) / dist - 1.0)
            if error_frac > 0.07:
                raise ValueError(
                    'input parameters to uniform_intervals would yield a non-uniform distribution.'
                )
    return values


@dataclass
class ErrorResult:
    rms: float
    max: float


class Value3D:
    def __init__(self, components) -> None:
        self.components = tuple(components)

    def __iter__(self) -> Iterator:
        return iter(self.components)

    def __add__(self, y):
        return Value3D(
            (
                self.components[0] + y.components[0],
                self.components[1] + y.components[1],
                self.components[2] + y.components[2],
            )
        )

    def __mul__(self, y):
        return Value3D(
            (
                self.components[0] * y,
                self.components[1] * y,
                self.components[2] * y,
            )
        )

    def __rmul__(self, x):
        return self.__mul__(x)


class ColorLUT:
    def __init__(
        self,
        data,
        *,
        sample_count: int,
        input_domain: SupportsFloat,
        red_increments_fastest: bool = True,
    ) -> None:
        self._validate(data, input_domain=input_domain, sample_count=sample_count)
        self.data = data
        self.sample_count = sample_count
        self.input_domain = input_domain
        self.red_increments_fastest = red_increments_fastest
        if data.typecode in 'fd':
            self.datatype = numbers.Real
        elif data.typecode in 'bBhHiIlL':
            self.datatype = numbers.Integral
        self.sample_distance = float(self.input_domain) / float(self.sample_count - 1)

    def _validate(
        self,
        data,
        *,
        input_domain: SupportsFloat,
        sample_count: int,
    ) -> None:
        if (
            not isinstance(data, array)
            or len(data) == 0
            or not isinstance(data[0], numbers.Number)
        ):
            raise ValidationError('data parameter should be a flat list of numbers.')
        if not isinstance(sample_count, int):
            raise ValidationError('sample_count parameter should be of type int.')
        if not isinstance(input_domain, numbers.Number):
            raise ValidationError('input_domain parameter should be a number.')
        if float(input_domain) <= 0:
            raise ValidationError('input_domain parameter must be positive.')
        if isinstance(input_domain, numbers.Integral):
            if data.typecode not in 'bBhHiIlL':
                raise ValidationError(
                    'input_domain parameter should have the same type as the data.'
                )
        else:
            if data.typecode not in 'fd':
                raise ValidationError(
                    'input_domain parameter should have the same type as the data.'
                )
        if not len(data) == 3 * (sample_count**3):
            raise ValidationError(
                'The sample intervals do not appear to match the matrix dimensions.'
            )

    def get_color_value_from_index(self, r_idx, g_idx, b_idx):
        """
        Determine the output color value given 3D matrix indices.
        """
        if not self.red_increments_fastest:
            r_idx, b_idx = b_idx, r_idx
        sc = self.sample_count
        idx = ((r_idx) + (sc * g_idx) + (sc**2 * b_idx)) * 3
        return Value3D(self.data[idx : idx + 3])

    def get_interpolated_color_value(self, r_input, g_input, b_input):
        """
        Determine the output color value using trilinear interpolation.

        Algorithm adapted from https://en.wikipedia.org/wiki/Trilinear_interpolation
        """

        # For each axis, locate the lower sample index v_0 and the fractional
        # distance v_d in [0, 1] to the next sample.  The index and the fraction
        # must be derived from the same quantity, otherwise rounding can pair an
        # index with a fraction belonging to its neighbor and shift the result
        # by a whole sample.  The lower index is clamped to sample_count - 2 so
        # that v_1 = v_0 + 1 stays in range, which also covers the border case
        # where v_input equals the maximum input value.
        def locate(v_input):
            pos = v_input / self.sample_distance
            v_0_idx = max(0, min(int(pos), self.sample_count - 2))
            return v_0_idx, pos - v_0_idx

        r_0_idx, r_d = locate(r_input)
        g_0_idx, g_d = locate(g_input)
        b_0_idx, b_d = locate(b_input)

        r_1_idx = r_0_idx + 1
        g_1_idx = g_0_idx + 1
        b_1_idx = b_0_idx + 1

        c_000 = self.get_color_value_from_index(r_0_idx, g_0_idx, b_0_idx)
        c_001 = self.get_color_value_from_index(r_0_idx, g_0_idx, b_1_idx)
        c_010 = self.get_color_value_from_index(r_0_idx, g_1_idx, b_0_idx)
        c_011 = self.get_color_value_from_index(r_0_idx, g_1_idx, b_1_idx)
        c_100 = self.get_color_value_from_index(r_1_idx, g_0_idx, b_0_idx)
        c_101 = self.get_color_value_from_index(r_1_idx, g_0_idx, b_1_idx)
        c_110 = self.get_color_value_from_index(r_1_idx, g_1_idx, b_0_idx)
        c_111 = self.get_color_value_from_index(r_1_idx, g_1_idx, b_1_idx)

        c_00 = c_000 * (1.0 - r_d) + c_100 * r_d
        c_01 = c_001 * (1.0 - r_d) + c_101 * r_d
        c_10 = c_010 * (1.0 - r_d) + c_110 * r_d
        c_11 = c_011 * (1.0 - r_d) + c_111 * r_d

        c_0 = c_00 * (1.0 - g_d) + c_10 * g_d
        c_1 = c_01 * (1.0 - g_d) + c_11 * g_d

        c = c_0 * (1.0 - b_d) + c_1 * b_d

        return c

    def get_values_translated(
        self,
        *,
        increment_red_fastest: bool = True,
        output_sample_count: int,
        output_domain: SupportsFloat,
    ):
        """
        Make an iterable of output color values in sequence.

        If necessary, reorder the output data values in order to make them
        correspond to red/blue input channels incrementing most/least rapidly
        by default.  Switch increment_red_fastest=False for the opposite
        behavior
        """
        interpolate_output = output_sample_count != self.sample_count
        scale_output = output_domain != self.input_domain

        input_domain_f = float(self.input_domain)
        scaling_factor = float(output_domain) / input_domain_f

        if increment_red_fastest:
            indexes = (
                (r, g, b)
                for b in range(output_sample_count)
                for g in range(output_sample_count)
                for r in range(output_sample_count)
            )
        else:
            indexes = (
                (r, g, b)
                for r in range(output_sample_count)
                for g in range(output_sample_count)
                for b in range(output_sample_count)
            )

        if interpolate_output:
            input_values = (
                Value3D(idx) * (input_domain_f / float(output_sample_count - 1))
                for idx in indexes
            )
            output_values = (
                self.get_interpolated_color_value(*input_value)
                for input_value in input_values
            )
        else:
            output_values = (self.get_color_value_from_index(*idx) for idx in indexes)

        if scale_output:
            output_values = (
                output_value * scaling_factor for output_value in output_values
            )

        return output_values

    @classmethod
    def from_haldclut(cls, src) -> Self:
        src_png = png.Reader(filename=src)
        width, height, data, meta = src_png.read_flat()
        if 'palette' in meta:
            raise ValidationError('The given PNG file uses a color palette. Refusing.')
        if 'gamma' in meta:
            raise ValidationError(
                'The given PNG file contains a gamma value. Refusing.'
            )
        if 'transparent' in meta:
            raise ValidationError(
                'The given PNG file specifies a transparent color. Refusing.'
            )
        if meta['alpha']:  # pragma: no cover
            raise ValidationError(
                'The given PNG file contains an alpha channel. Refusing.'
            )
        if meta['greyscale']:

            def triple_generator(d):
                for val in d:
                    yield val
                    yield val
                    yield val

            data = array(data.typecode, triple_generator(data))
        bitdepth = meta['bitdepth']
        if bitdepth not in (8, 16):
            raise ValidationError(
                f'The given PNG file specifies an unsupported bit depth {bitdepth}. Refusing.'
            )
        if width != height or not is_perfect_six_root(width**2):
            raise ValidationError(
                'The given PNG file does not have appropriate Hald CLUT dimensions. Refusing.'
            )
        sample_count = int(round((width**2) ** (1.0 / 3)))
        input_domain = 2**bitdepth - 1
        print(f'from_haldclut(): PNG dimensions = {width}×{height}')
        print(f"from_haldclut(): PNG bit depth = {bitdepth}")
        print(f'from_haldclut(): PNG array typecode = {data.typecode}')
        print(f'from_haldclut(): Inferred input domain = {input_domain}')
        print(
            f'from_haldclut(): Inferred 3D matrix dimensions = {sample_count}×{sample_count}×{sample_count}'
        )
        return cls(data, sample_count=sample_count, input_domain=input_domain)

    @classmethod
    def from_3dl(cls, src: str | PathLike) -> Self:
        raise NotImplementedError()

    @classmethod
    def from_cube(cls, src: str | PathLike) -> Self:
        sample_count = None
        domain_min = (0.0, 0.0, 0.0)
        domain_max = (1.0, 1.0, 1.0)
        data = array('f')
        with open(src) as srcfile:
            for line in srcfile:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                keyword = parts[0].upper()
                if keyword == 'TITLE':
                    continue
                elif keyword == 'LUT_3D_SIZE':
                    if len(parts) < 2:
                        raise ValidationError(
                            'LUT_3D_SIZE line is missing its size value. Refusing.'
                        )
                    sample_count = int(parts[1])
                elif keyword == 'LUT_1D_SIZE':
                    raise ValidationError('1D Cube LUTs are not supported. Refusing.')
                elif keyword == 'DOMAIN_MIN':
                    if len(parts) < 4:
                        raise ValidationError(
                            'DOMAIN_MIN line must have three values. Refusing.'
                        )
                    domain_min = tuple(float(v) for v in parts[1:4])
                elif keyword == 'DOMAIN_MAX':
                    if len(parts) < 4:
                        raise ValidationError(
                            'DOMAIN_MAX line must have three values. Refusing.'
                        )
                    domain_max = tuple(float(v) for v in parts[1:4])
                else:
                    try:
                        values = [float(v) for v in parts]
                    except ValueError:
                        raise ValidationError(f'Unexpected line in Cube file: {line}')
                    if len(values) != 3:
                        raise ValidationError(
                            f'Cube data line does not have exactly 3 values: {line}'
                        )
                    data.extend(values)
        if sample_count is None:
            raise ValidationError(
                'The given Cube file is missing LUT_3D_SIZE. Refusing.'
            )
        if domain_min != (0.0, 0.0, 0.0) or domain_max != (1.0, 1.0, 1.0):
            raise ValidationError(
                'The given Cube file uses a non-default input domain, which is not supported. Refusing.'
            )
        if len(data) != 3 * sample_count**3:
            raise ValidationError(
                'The Cube file data does not match the declared LUT_3D_SIZE. Refusing.'
            )
        print(f'from_cube(): LUT_3D_SIZE = {sample_count}')
        return cls(data, sample_count=sample_count, input_domain=1.0)

    def _grid_coordinates(self, sample_count):
        """
        Build the (sample_count**3, 3) array of input coordinates, in this LUT's
        input domain, ordered with the red axis incrementing fastest (matching
        the flat data layout: index = r + n*g + n**2*b).
        """
        axis = np.arange(sample_count, dtype=np.float64)
        b_idx, g_idx, r_idx = np.meshgrid(axis, axis, axis, indexing='ij')
        step = self.input_domain / (sample_count - 1)
        return np.stack(
            [r_idx.ravel() * step, g_idx.ravel() * step, b_idx.ravel() * step],
            axis=1,
        )

    def _interpolate_coords_numpy(self, coords):
        """
        Trilinearly sample this LUT at an (M, 3) array of r/g/b input values
        (in this LUT's input domain), returning an (M, 3) output array.

        This is a vectorized equivalent of get_interpolated_color_value().
        """
        n = self.sample_count
        # The flat data is reshaped [slowest, g, fastest, channel].  With red
        # incrementing fastest the layout is [b, g, r]; otherwise it is [r, g, b],
        # matching the r/b swap that get_color_value_from_index performs.
        grid = np.asarray(self.data, dtype=np.float64).reshape(n, n, n, 3)
        pos = coords / self.sample_distance
        i0 = np.clip(pos.astype(np.int64), 0, n - 2)
        frac = pos - i0
        r0, g0, b0 = i0[:, 0], i0[:, 1], i0[:, 2]
        r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1
        r_d, g_d, b_d = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]

        if self.red_increments_fastest:

            def at(ri, gi, bi):
                return grid[bi, gi, ri]
        else:

            def at(ri, gi, bi):
                return grid[ri, gi, bi]

        c_00 = at(r0, g0, b0) * (1.0 - r_d) + at(r1, g0, b0) * r_d
        c_01 = at(r0, g0, b1) * (1.0 - r_d) + at(r1, g0, b1) * r_d
        c_10 = at(r0, g1, b0) * (1.0 - r_d) + at(r1, g1, b0) * r_d
        c_11 = at(r0, g1, b1) * (1.0 - r_d) + at(r1, g1, b1) * r_d
        c_0 = c_00 * (1.0 - g_d) + c_10 * g_d
        c_1 = c_01 * (1.0 - g_d) + c_11 * g_d
        return c_0 * (1.0 - b_d) + c_1 * b_d

    def measure_error_against(self, reference: ColorLUT) -> ErrorResult:
        """
        Estimate how much this LUT deviates from a reference LUT.

        The reference's grid points are taken as ground truth: at each one, the
        reference's stored output is compared against this LUT's interpolated
        output.  Both sides are normalized to a [0, 1] output range so LUTs with
        different domains (e.g. a float Cube and an 8-bit Hald CLUT) can be
        compared.
        """
        if np is not None:
            rc = reference.sample_count
            ref_grid = np.asarray(reference.data, dtype=np.float64).reshape(
                rc, rc, rc, 3
            )
            if not reference.red_increments_fastest:
                # Re-order [r, g, b] storage into the [b, g, r] enumeration that
                # _grid_coordinates produces, matching the scalar path's r/b swap.
                ref_grid = ref_grid.transpose(2, 1, 0, 3)
            ref_out = ref_grid.reshape(-1, 3) / reference.input_domain
            coords = self._grid_coordinates(reference.sample_count)
            out = self._interpolate_coords_numpy(coords) / self.input_domain
            error = np.abs(ref_out - out)
            return ErrorResult(
                max=float(error.max()),
                rms=float(np.sqrt(np.mean(error * error))),
            )

        sd = reference.sample_distance
        scale = self.input_domain / reference.input_domain
        squared_error = 0.0
        max_error = 0.0
        count = 0
        for b_idx in range(reference.sample_count):
            for g_idx in range(reference.sample_count):
                for r_idx in range(reference.sample_count):
                    ref = reference.get_color_value_from_index(r_idx, g_idx, b_idx)
                    out = self.get_interpolated_color_value(
                        r_idx * sd * scale,
                        g_idx * sd * scale,
                        b_idx * sd * scale,
                    )
                    for ref_c, out_c in zip(ref, out):
                        error = abs(
                            ref_c / reference.input_domain - out_c / self.input_domain
                        )
                        squared_error += error * error
                        if error > max_error:
                            max_error = error
                        count += 1
        return ErrorResult(
            max=max_error,
            rms=(squared_error / count) ** 0.5,
        )

    def render_haldclut(self, level: int, bit_depth: int):
        """
        Resample this LUT onto a Hald CLUT of the given level and bit depth.

        Returns (data, width, height, output_domain), where data is a flat array
        of quantized RGB samples ready to be written as a PNG.
        """
        output_sample_count = level**2
        width = height = level**3
        output_domain = 2**bit_depth - 1
        typecode = 'B' if bit_depth == 8 else 'H'
        if np is not None:
            coords = self._grid_coordinates(output_sample_count)
            out = self._interpolate_coords_numpy(coords)
            out *= output_domain / self.input_domain
            out = np.clip(np.rint(out), 0, output_domain).astype(np.int64)
            data = array(typecode, out.ravel().tolist())
        else:
            data = array(typecode)
            for color in self.get_values_translated(
                increment_red_fastest=True,
                output_sample_count=output_sample_count,
                output_domain=output_domain,
            ):
                for component in color:
                    # Clamp out-of-gamut LUT values.
                    data.append(min(output_domain, max(0, int(round(component)))))
        return data, width, height, output_domain

    def _haldclut_error(self, data, level: int, output_domain):
        """Measure the error of a rendered Hald CLUT against this LUT."""
        written = ColorLUT(data, sample_count=level**2, input_domain=output_domain)
        return written.measure_error_against(self)

    def find_haldclut_level(
        self,
        max_error: float,
        bit_depth: int = 8,
    ) -> tuple[int, ErrorResult, bool, tuple]:
        """
        Find the smallest Hald CLUT level whose max error is within max_error.

        Searches increasing levels up to the point where the output has at least
        as many samples per axis as this LUT (beyond which only quantization
        noise remains).  Returns (level, error, satisfied, render), where render
        is the chosen level's (data, width, height, output_domain) tuple so the
        caller can write it without resampling again; when no level meets the
        target, returns the best level found with satisfied=False.
        """
        max_level = int(self.sample_count**0.5)
        if max_level**2 < self.sample_count:
            max_level += 1
        max_level = max(2, min(max_level, 16))
        print(
            f'find_haldclut_level(): searching levels 2..{max_level} for max error <= {max_error:.5f}'
        )
        best: tuple[int, ErrorResult, tuple] | None = None
        for level in range(2, max_level + 1):
            render = self.render_haldclut(level, bit_depth)
            data, _, _, output_domain = render
            error = self._haldclut_error(data, level, output_domain)
            print(
                f"find_haldclut_level(): level {level} -> max error {error.max:.5f} (target {max_error:.5f}){' OK' if error.max <= max_error else ''}"
            )
            if best is None or error.max < best[1].max:
                best = (level, error, render)
            if error.max <= max_error:
                return level, error, True, render
        assert best
        return best[0], best[1], False, best[2]

    def write_haldclut(
        self,
        dest: PathLike | str,
        *,
        level=None,
        bit_depth: int = 8,
        max_error=None,
        measure_error: bool = True,
    ):
        if bit_depth not in (8, 16):
            raise ValidationError('Hald CLUT bit depth must be 8 or 16.')
        error = None
        if max_error is not None:
            level, error, satisfied, render = self.find_haldclut_level(
                max_error, bit_depth
            )
            if not satisfied:
                print(
                    f"write_haldclut(): could not meet max error {max_error:.5f}; using best level {level} (max error {error.max:.5f})."
                )
        else:
            if level is None:
                root = int(round(self.sample_count**0.5))
                if root**2 != self.sample_count:
                    raise ValidationError(
                        'Cannot infer a Hald CLUT level because the LUT sample count '
                        f'({self.sample_count}) is not a perfect square. Please specify a level or '
                        '--max-error.'
                    )
                level = root
            render = self.render_haldclut(level, bit_depth)
        data, width, height, output_domain = render
        print(f'write_haldclut(): level = {level}')
        print(f'write_haldclut(): PNG dimensions = {width}×{height}')
        print(f'write_haldclut(): PNG bit depth = {bit_depth}')
        write_png(dest, data, width, height, bit_depth)
        if error is None and measure_error:
            error = self._haldclut_error(data, level, output_domain)
        if error is not None:
            print(
                f"write_haldclut(): max error = {error.max:.5f} ({error.max * output_domain:.1f} of {output_domain} levels), rms error = {error.rms:.5f}"
            )

    def write_3dl(self, dest: PathLike | str) -> None:
        output_domain = 1023
        sample_intervals = uniform_intervals(output_domain, self.sample_count)
        color_value_gen = self.get_values_translated(
            increment_red_fastest=False,
            output_sample_count=self.sample_count,
            output_domain=output_domain,
        )
        with open(dest, 'w') as destfile:
            destfile.write('   '.join(str(v) for v in sample_intervals))
            destfile.write('\n')
            for color in color_value_gen:
                line = ' '.join(f'{v:.0f}' for v in color)
                destfile.write(line)
                destfile.write('\n')

    def write_cube(self, dest: PathLike | str) -> None:
        output_domain = 1.0
        output_sample_count = self.sample_count
        color_value_gen = self.get_values_translated(
            output_sample_count=output_sample_count,
            output_domain=output_domain,
        )
        with open(dest, 'w') as destfile:
            destfile.write(f'LUT_3D_SIZE {output_sample_count}')
            destfile.write('\n')
            for color in color_value_gen:
                line = ' '.join(f'{v:.7g}' for v in color)
                destfile.write(line)
                destfile.write('\n')
