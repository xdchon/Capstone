#!/usr/bin/env python3
"""
Randomly sample slices from a SlideBook .sldy file and save them as TIFF images.

Samples are drawn from (timepoint, z-slice) pairs and exported as one TIFF per sample.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile

from cellpose.io import LazySldy


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Pick N random slices from a .sldy file and export each slice to TIFF "
            "for training/annotation workflows."
        )
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Path to input .sldy file.",
    )
    ap.add_argument(
        "--num-slices",
        type=int,
        required=True,
        help="Number of random slices to export.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output folder for TIFF files. Defaults to "
            "<input_stem>_random_slices beside the input file."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    ap.add_argument(
        "--capture-index",
        type=int,
        default=0,
        help="SlideBook capture index (default: 0).",
    )
    ap.add_argument(
        "--position-index",
        type=int,
        default=0,
        help="SlideBook position index (default: 0).",
    )
    ap.add_argument(
        "--time-start",
        type=int,
        default=0,
        help="First timepoint (inclusive) to sample from (default: 0).",
    )
    ap.add_argument(
        "--time-end",
        type=int,
        default=-1,
        help="Last timepoint (inclusive) to sample from; -1 means last available.",
    )
    ap.add_argument(
        "--z-start",
        type=int,
        default=0,
        help="First z-slice (inclusive) to sample from (default: 0).",
    )
    ap.add_argument(
        "--z-end",
        type=int,
        default=-1,
        help="Last z-slice (inclusive) to sample from; -1 means last available.",
    )
    ap.add_argument(
        "--channel",
        type=int,
        default=-1,
        help=(
            "Channel index to export as single-channel TIFF (0-based). "
            "Use -1 to export all channels as YXC (default: -1)."
        ),
    )
    ap.add_argument(
        "--with-replacement",
        action="store_true",
        help="Allow selecting the same (T,Z) pair more than once.",
    )
    return ap.parse_args()


def _clamp_range(start: int, end: int, n: int) -> Tuple[int, int]:
    if n <= 0:
        return 0, -1
    s = max(0, min(int(start), n - 1))
    if int(end) < 0:
        e = n - 1
    else:
        e = max(0, min(int(end), n - 1))
    if e < s:
        s, e = e, s
    return s, e


def _choose_indices(
    candidates: List[Tuple[int, int]],
    n_pick: int,
    rng: np.random.Generator,
    with_replacement: bool,
) -> List[Tuple[int, int]]:
    if n_pick <= 0:
        return []
    n_total = len(candidates)
    if n_total == 0:
        return []

    if not with_replacement and n_pick > n_total:
        raise ValueError(
            f"requested {n_pick} slices but only {n_total} unique (T,Z) pairs are available"
        )

    if with_replacement:
        idx = rng.integers(0, n_total, size=n_pick)
    else:
        idx = rng.choice(n_total, size=n_pick, replace=False)
    return [candidates[int(i)] for i in idx]


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"input file does not exist: {input_path}")
    if input_path.suffix.lower() != ".sldy":
        raise SystemExit(f"input must be a .sldy file: {input_path}")
    if args.num_slices <= 0:
        raise SystemExit("--num-slices must be > 0")

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = input_path.parent / f"{input_path.stem}_random_slices"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"INFO: opening {input_path}")
    reader = LazySldy(
        str(input_path),
        capture_index=int(args.capture_index),
        position_index=int(args.position_index),
    )

    nt = int(reader.nt)
    nz = int(reader.nz)
    nc = int(reader.nc)
    print(f"INFO: source dims -> T={nt}, Z={nz}, C={nc}, Y={reader.ny}, X={reader.nx}")

    t0, t1 = _clamp_range(args.time_start, args.time_end, nt)
    z0, z1 = _clamp_range(args.z_start, args.z_end, nz)
    if t1 < t0 or z1 < z0:
        raise SystemExit("no valid time/z range available after clamping")

    if args.channel >= nc and args.channel >= 0:
        raise SystemExit(
            f"--channel {args.channel} is out of range for source channels C={nc}"
        )

    candidates = [(t, z) for t in range(t0, t1 + 1) for z in range(z0, z1 + 1)]
    print(
        "INFO: sampling from range "
        f"T={t0}..{t1}, Z={z0}..{z1} ({len(candidates)} candidates)"
    )

    rng = np.random.default_rng(int(args.seed))
    picks = _choose_indices(
        candidates=candidates,
        n_pick=int(args.num_slices),
        rng=rng,
        with_replacement=bool(args.with_replacement),
    )

    manifest_path = out_dir / "sample_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["output_file", "timepoint", "z_index", "channel", "dtype", "shape"])

        for i, (t, z) in enumerate(picks):
            plane = reader.get_plane(int(t), int(z))  # YXC (padded to <=3 channels by reader)
            if args.channel >= 0:
                img = np.asarray(plane[..., int(args.channel)])
                channel_tag = f"C{int(args.channel):02d}"
            else:
                img = np.asarray(plane)
                channel_tag = "CALL"

            out_name = (
                f"{input_path.stem}_sample_{i:04d}_T{int(t):04d}_Z{int(z):04d}_{channel_tag}.tif"
            )
            out_path = out_dir / out_name
            tifffile.imwrite(
                str(out_path),
                img,
                compression="zlib",
            )

            writer.writerow(
                [
                    out_name,
                    int(t),
                    int(z),
                    int(args.channel) if args.channel >= 0 else "all",
                    str(img.dtype),
                    "x".join(str(v) for v in img.shape),
                ]
            )

            if (i + 1) % 10 == 0 or i + 1 == len(picks):
                print(f"INFO: wrote {i + 1}/{len(picks)}")

    print(f"INFO: done. wrote {len(picks)} TIFF files to {out_dir}")
    print(f"INFO: manifest: {manifest_path}")


if __name__ == "__main__":
    main()
