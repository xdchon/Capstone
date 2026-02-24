#!/usr/bin/env python3
"""
Merge per-timepoint labeled mask TIFFs into one OME-TIFF (TZYX).

Example:
  python Program/cellpose/merge_timepoint_mask_tiffs_to_ome.py \
    --input "E:/.../my_cp_masks_by_time" \
    --output "E:/.../my_cp_masks.ome.tif"
"""

import argparse
import os
import sys
from pathlib import Path


def _ensure_local_cellpose_on_path():
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def parse_args():
    p = argparse.ArgumentParser(
        description="Combine timepoint mask TIFFs from a folder into one OME-TIFF (TZYX)."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Folder containing per-timepoint mask TIFFs (e.g. *_cp_masks_by_time).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output OME-TIFF path. Default: inferred from input folder name.",
    )
    p.add_argument(
        "--strict-shape",
        action="store_true",
        help="Require all input TIFFs to have identical ZYX shape.",
    )
    p.add_argument(
        "--compression",
        default="zlib",
        help="TIFF compression (default: zlib).",
    )
    return p.parse_args()


def main():
    _ensure_local_cellpose_on_path()
    from cellpose.gui.io import combine_timepoint_masks_folder_to_ome

    args = parse_args()
    out_path, nt, shape, out_dtype = combine_timepoint_masks_folder_to_ome(
        args.input,
        output_path=args.output,
        strict_shape=bool(args.strict_shape),
        compression=args.compression,
    )
    print(
        f"[done] wrote {out_path}\n"
        f"       shape=(T={nt}, Z={shape[0]}, Y={shape[1]}, X={shape[2]}) "
        f"dtype={out_dtype.__name__ if hasattr(out_dtype, '__name__') else str(out_dtype)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

