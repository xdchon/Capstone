#!/usr/bin/env python3
"""
Batch segment TIFF stacks with a trained Cellpose model and save masks as .npy.
Assumes data shaped (Z, C, Y, X); pads channels to 3 for CPSAM.
"""

from pathlib import Path
import argparse
import numpy as np
from cellpose import io, models


def pad_to_three_channels(img: np.ndarray, channel_axis: int) -> np.ndarray:
    if img.shape[channel_axis] == 2:
        pad_shape = list(img.shape)
        pad_shape[channel_axis] = 1
        pad = np.zeros(pad_shape, dtype=img.dtype)
        img = np.concatenate([img, pad], axis=channel_axis)
    elif img.shape[channel_axis] == 1:
        img = np.repeat(img, 3, axis=channel_axis)
    elif img.shape[channel_axis] > 3:
        slc = [slice(None)] * img.ndim
        slc[channel_axis] = slice(0, 3)
        img = img[tuple(slc)]
    return img


def main():
    ap = argparse.ArgumentParser(description="Batch Cellpose segmentation and .npy mask export.")
    ap.add_argument("--input-dir", required=True, help="Folder containing .tif/.tiff (default non-recursive).")
    ap.add_argument("--model", required=True, help="Path or name of your pretrained model.")
    ap.add_argument("--output-dir", help="Where to write .npy masks; defaults to input folder structure.")
    ap.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    ap.add_argument("--invert", action="store_true", help="Invert intensities before segmentation.")
    ap.add_argument("--no-normalize", action="store_true", help="Disable Cellpose normalization.")
    ap.add_argument("--cellprob-threshold", type=float, default=0.0, help="Cellprob threshold.")
    ap.add_argument("--flow-threshold", type=float, default=0.4, help="Flow error threshold.")
    ap.add_argument("--min-size", type=int, default=15, help="Remove masks smaller than this (pixels).")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    args = ap.parse_args()

    in_root = Path(args.input_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else in_root

    files = (in_root.rglob("*.tif*") if args.recursive else in_root.glob("*.tif*"))
    files = sorted(f for f in files if f.is_file())

    if not files:
        raise SystemExit(f"No TIFF files found in {in_root}")

    model = models.CellposeModel(gpu=args.gpu, pretrained_model=str(Path(args.model)))

    for f in files:
        img = io.imread(str(f))

        # Ensure (Z, C, Y, X)
        if img.ndim == 2:
            img = img[np.newaxis, np.newaxis, ...]
        elif img.ndim == 3:
            # assume (Z, Y, X) with no channels
            img = np.stack([img] * 3, axis=1)  # now (Z, C=3, Y, X)
        elif img.ndim == 4:
            pass  # assume (Z, C, Y, X) already
        else:
            raise ValueError(f"Unsupported ndim={img.ndim} for {f}")

        # Pad/trim channels to 3 for CPSAM
        img = pad_to_three_channels(img, channel_axis=1)

        masks, flows, _ = model.eval(
            img,
            channel_axis=1,  # (Z, C, Y, X)
            z_axis=0,
            do_3D=True,
            normalize=not args.no_normalize,
            invert=args.invert,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold,
            min_size=args.min_size,
        )

        rel = f.relative_to(in_root)
        target_dir = (out_root / rel.parent)
        target_dir.mkdir(parents=True, exist_ok=True)
        npy_path = target_dir / (f.stem + "_seg.npy")
        np.save(npy_path, masks)

        # Optional: also save flow data per image if you want
        # np.save(target_dir / (f.stem + "_flows.npy"), flows)

    print("Done.")


if __name__ == "__main__":
    main()
