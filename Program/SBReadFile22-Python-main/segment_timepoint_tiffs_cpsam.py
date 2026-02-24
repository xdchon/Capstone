#!/usr/bin/env python3
"""
Run CPSAM segmentation on timepoint TIFF stacks.

This script is designed for stacks like:
  - timepoint_0000_ZCYX.ome.tif  (Z, C, Y, X)
  - timepoint_0000.tif           (Z, Y, X)

By default it searches recursively for files named `timepoint_*.tif*`,
runs Cellpose/CPSAM in 3D, and saves mask outputs as `.npy`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import tifffile
from cellpose import models


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run CPSAM segmentation on timepoint TIFF stacks."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input TIFF file or folder containing timepoint TIFF files.",
    )
    p.add_argument(
        "--pattern",
        default="timepoint_*.tif*",
        help="Glob pattern used when --input is a folder (default: timepoint_*.tif*).",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search folders recursively (default: on).",
    )
    p.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Search only the top level of --input folder.",
    )
    p.add_argument(
        "--model",
        default="cpsam",
        help="Pretrained model name or path (default: cpsam).",
    )
    p.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    p.add_argument(
        "--output-root",
        default=None,
        help="Optional output root. If omitted, outputs are written beside each input TIFF.",
    )
    p.add_argument(
        "--save-ome-tiff",
        action="store_true",
        help="Also save mask stacks as OME-TIFF (ZYX).",
    )
    p.add_argument(
        "--save-flows",
        action="store_true",
        help="Also save Cellpose flow outputs as .npy.",
    )
    p.add_argument(
        "--invert",
        action="store_true",
        help="Invert image intensities before segmentation.",
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable Cellpose normalization.",
    )
    p.add_argument(
        "--diameter",
        type=float,
        default=0.0,
        help="Object diameter in pixels (<=0 uses auto/default).",
    )
    p.add_argument(
        "--cellprob-threshold",
        type=float,
        default=0.0,
        help="Cell probability threshold.",
    )
    p.add_argument(
        "--flow-threshold",
        type=float,
        default=0.4,
        help="Flow error threshold.",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=15,
        help="Minimum mask size in pixels.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask outputs.",
    )
    return p.parse_args()


def collect_inputs(input_path: Path, pattern: str, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise SystemExit(f"Input path does not exist: {input_path}")
    files: Iterable[Path]
    files = input_path.rglob(pattern) if recursive else input_path.glob(pattern)
    out = sorted(p for p in files if p.is_file())
    if not out:
        raise SystemExit(f"No files found under {input_path} with pattern {pattern!r}")
    return out


def to_zcyx(img: np.ndarray, src: Path) -> np.ndarray:
    """
    Convert common stack layouts to ZCYX.
    """
    if img.ndim == 2:
        # YX -> ZCYX
        return img[np.newaxis, np.newaxis, :, :]

    if img.ndim == 3:
        # Assume ZYX -> ZCYX
        return img[:, np.newaxis, :, :]

    if img.ndim == 4:
        # Likely already ZCYX
        if img.shape[1] <= 4:
            return img
        # ZYXC -> ZCYX
        if img.shape[-1] <= 4:
            return np.moveaxis(img, -1, 1)
        # CZYX -> ZCYX
        if img.shape[0] <= 4:
            return np.moveaxis(img, 0, 1)
        raise ValueError(
            f"Could not infer channel axis for 4D array shape {img.shape} in {src}"
        )

    if img.ndim == 5:
        # Accept single-timepoint TZCYX
        if img.shape[0] == 1 and img.shape[2] <= 4:
            return img[0]
        raise ValueError(
            f"5D input must be single-timepoint TZCYX (T=1); got shape {img.shape} in {src}"
        )

    raise ValueError(f"Unsupported image ndim={img.ndim} for {src}")


def pad_or_trim_to_three_channels(img_zcyx: np.ndarray) -> np.ndarray:
    c = img_zcyx.shape[1]
    if c == 3:
        return img_zcyx
    if c == 2:
        pad = np.zeros(
            (img_zcyx.shape[0], 1, img_zcyx.shape[2], img_zcyx.shape[3]),
            dtype=img_zcyx.dtype,
        )
        return np.concatenate([img_zcyx, pad], axis=1)
    if c == 1:
        return np.repeat(img_zcyx, 3, axis=1)
    return img_zcyx[:, :3, :, :]


def choose_mask_dtype(mask: np.ndarray) -> np.dtype:
    max_label = int(mask.max()) if mask.size else 0
    return np.uint16 if max_label < 2**16 else np.uint32


def target_dir_for_file(
    file_path: Path,
    input_root: Path,
    output_root: Path | None,
) -> Path:
    if output_root is None:
        return file_path.parent
    if input_root.is_file():
        return output_root
    rel = file_path.relative_to(input_root)
    return output_root / rel.parent


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve() if args.output_root else None
    )
    files = collect_inputs(input_path, args.pattern, args.recursive)

    diameter = None if args.diameter is None or args.diameter <= 0 else float(args.diameter)
    model = models.CellposeModel(gpu=bool(args.gpu), pretrained_model=args.model)

    print(f"[info] Found {len(files)} TIFF file(s).")
    print(f"[info] Model: {args.model}")
    print(f"[info] GPU: {'on' if args.gpu else 'off'}")

    for i, f in enumerate(files, start=1):
        out_dir = target_dir_for_file(f, input_path, output_root)
        out_dir.mkdir(parents=True, exist_ok=True)

        npy_path = out_dir / f"{f.stem}_seg.npy"
        tiff_path = out_dir / f"{f.stem}_masks.ome.tif"
        flows_path = out_dir / f"{f.stem}_flows.npy"

        if not args.overwrite and npy_path.exists():
            print(f"[skip] ({i}/{len(files)}) {f} -> {npy_path} already exists")
            continue

        print(f"[run ] ({i}/{len(files)}) {f}")
        img = tifffile.imread(str(f))
        img_zcyx = to_zcyx(np.asarray(img), f)
        img_zcyx = pad_or_trim_to_three_channels(img_zcyx)

        masks, flows, _styles = model.eval(
            img_zcyx,
            channel_axis=1,  # Z, C, Y, X
            z_axis=0,
            do_3D=True,
            normalize=not args.no_normalize,
            invert=bool(args.invert),
            diameter=diameter,
            cellprob_threshold=float(args.cellprob_threshold),
            flow_threshold=float(args.flow_threshold),
            min_size=int(args.min_size),
            compute_masks=True,
        )

        np.save(npy_path, masks)
        print(f"[save] {npy_path}")

        if args.save_ome_tiff:
            mdtype = choose_mask_dtype(np.asarray(masks))
            tifffile.imwrite(
                tiff_path,
                np.asarray(masks, dtype=mdtype),
                photometric="minisblack",
                metadata={"axes": "ZYX"},
                ome=True,
                bigtiff=True,
            )
            print(f"[save] {tiff_path}")

        if args.save_flows:
            np.save(flows_path, flows)
            print(f"[save] {flows_path}")

    print("[done] CPSAM batch segmentation complete.")


if __name__ == "__main__":
    main()
