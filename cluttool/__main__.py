from __future__ import annotations

import click

from cluttool.cluttool import ColorLUT


@click.command()
@click.argument(
    'src',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.argument(
    'dest',
    type=click.Path(exists=False),
    required=True,
)
@click.option(
    '--dest-type',
    help='Type of color LUT.  If this argument is not provided, the output type is inferred from the destination filename extension.',
    type=click.Choice(['3dl', 'haldclut', 'cube']),
)
@click.option(
    '--haldclut-level',
    type=int,
    default=None,
    help='Hald CLUT level for PNG output. The cube has level**2 samples per axis. '
    'If omitted, it is inferred from the source LUT (requires a perfect-square sample count).',
)
@click.option(
    '--haldclut-bit-depth',
    type=click.Choice(['8', '16']),
    default='8',
    help='Bit depth for Hald CLUT PNG output.',
)
@click.option(
    '--max-error',
    type=float,
    default=None,
    help='Automatically choose the smallest Hald CLUT level whose max error '
    '(in a normalized 0..1 output range) stays within this value. '
    'Mutually exclusive with --haldclut-level.',
)
@click.option(
    '--measure-error/--no-measure-error',
    default=True,
    help='Report the resampling/quantization error after writing a Hald CLUT. '
    'Disable to skip the (potentially slow) measurement for large source LUTs.',
)
def cli(
    src: str,
    dest: str,
    dest_type: str | None,
    haldclut_level: int | None,
    haldclut_bit_depth: str,
    max_error: float | None,
    measure_error: bool,
):
    if haldclut_level is not None and max_error is not None:
        raise click.UsageError('Use either --haldclut-level or --max-error, not both.')
    if not dest_type:
        dest_type = dest.lower().split('.')[-1]
        if dest_type == 'png':
            dest_type = 'haldclut'
    dest_type = dest_type.lower()
    src_ext = src.lower().split('.')[-1]
    if src_ext == 'png':
        clut = ColorLUT.from_haldclut(src)
    elif src_ext == '3dl':
        clut = ColorLUT.from_3dl(src)
    elif src_ext == 'cube':
        clut = ColorLUT.from_cube(src)
    else:  # pragma: no cover
        raise ValueError(f'Not an appropriate Color LUT file type: {src_ext}')
    if dest_type == 'haldclut':
        clut.write_haldclut(
            dest,
            level=haldclut_level,
            bit_depth=int(haldclut_bit_depth),
            max_error=max_error,
            measure_error=measure_error,
        )
    elif dest_type == '3dl':
        clut.write_3dl(dest)
    elif dest_type == 'cube':
        clut.write_cube(dest)
    else:  # pragma: no cover
        raise ValueError(f'Not an appropriate Color LUT file type: {dest_type}')


if __name__ == '__main__':
    cli()
