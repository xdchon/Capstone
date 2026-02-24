#!/usr/bin/env python3
"""
Extract random Z slices from a 3D TIFF stack and save them as individual TIFF files.

Example:
    python3 extract_random_tiff_slices.py \
        --input timepoint_0000.tif \
        --count 25 \
        --output-dir random_slices \
        --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from types import ModuleType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a random set of slices from a 3D TIFF stack."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input 3D TIFF file.",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        required=True,
        help="Number of random slices to extract.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory where extracted slices will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--prefix",
        default="slice",
        help="Output filename prefix (default: slice).",
    )
    return parser.parse_args()


def load_tifffile() -> ModuleType:
    try:
        import tifffile  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'tifffile'. Install it with: python3 -m pip install tifffile"
        ) from exc
    return tifffile


def validate_3d_series(path: Path, tifffile: ModuleType) -> tuple[int, str]:
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise SystemExit(f"No image data found in TIFF: {path}")

        series = tif.series[0]
        shape = series.shape
        axes = series.axes

        if len(shape) != 3:
            raise SystemExit(
                f"Expected a 3D TIFF stack, but got shape={shape}, axes={axes!r}."
            )

        z_count = shape[0]
        if z_count <= 0:
            raise SystemExit(f"Invalid number of slices in TIFF: {z_count}")

        return z_count, axes


def sample_indices(z_count: int, count: int, seed: int | None) -> list[int]:
    if count <= 0:
        raise SystemExit(f"--count must be > 0, got {count}")
    if count > z_count:
        raise SystemExit(
            f"Requested {count} slices, but TIFF only has {z_count} slices."
        )

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(z_count), count))
    return indices


def extract_slices(
    input_path: Path,
    output_dir: Path,
    indices: list[int],
    prefix: str,
    tifffile: ModuleType,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(input_path) as tif:
        pages = tif.pages
        if len(pages) < max(indices) + 1:
            raise SystemExit(
                "TIFF page count is smaller than expected for this 3D stack. "
                "This file may not be stored as one slice per page."
            )

        for idx in indices:
            slice_image = pages[idx].asarray()
            out_name = f"{prefix}_z{idx:04d}.tif"
            out_path = output_dir / out_name
            tifffile.imwrite(out_path, slice_image, photometric="minisblack")


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input TIFF does not exist: {input_path}")
    if input_path.suffix.lower() not in {".tif", ".tiff"}:
        raise SystemExit(
            f"Input file does not look like TIFF (expected .tif/.tiff): {input_path}"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    tifffile = load_tifffile()
    z_count, axes = validate_3d_series(input_path, tifffile)
    indices = sample_indices(z_count, args.count, args.seed)

    extract_slices(input_path, output_dir, indices, args.prefix, tifffile)

    print(f"Input:   {input_path}")
    print(f"Axes:    {axes}")
    print(f"Slices:  {z_count}")
    print(f"Chosen:  {indices}")
    print(f"Output:  {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
