"""
Export a SlideBook .sldyz file to a TIFF stack.

Defaults: timepoint 0 only, single channel, Z-stack ready for Cellpose/Fiji.
"""

import argparse
import os
import sys
import numpy as np
import tifffile
from tqdm import tqdm

from SBReadFile import SBReadFile


def parse_args():
    p = argparse.ArgumentParser(description="Export SlideBook .sldyz to TIFF stack")
    p.add_argument("slide", help="Path to .sldyz file")
    p.add_argument("--capture", type=int, default=0, help="Capture index (default: 0)")
    p.add_argument("--position", type=int, default=0, help="Montage position (default: 0)")
    p.add_argument("--channel", type=int, default=0, help="Channel index to export (default: 0)")
    p.add_argument("--timepoint", type=int, default=0, help="Timepoint index to export (default: 0)")
    p.add_argument("--all-time", action="store_true", help="Export all timepoints (outputs TZYX)")
    p.add_argument(
        "--outfile",
        help="Output TIFF path. If omitted, writes alongside the slide with name <slide>_T<tp>_C<ch>.ome.tif",
    )
    return p.parse_args()


def default_outfile(slide_path, timepoint, channel):
    stem = os.path.splitext(os.path.basename(slide_path))[0]
    return os.path.join(
        os.path.dirname(slide_path),
        f"{stem}_T{timepoint}_C{channel}.ome.tif",
    )


def main():
    args = parse_args()

    slide_path = os.path.abspath(args.slide)
    reader = SBReadFile()
    if not reader.Open(slide_path):
        raise SystemExit(f"Could not open slide: {slide_path}")

    n_t = reader.GetNumTimepoints(args.capture)
    n_z = reader.GetNumZPlanes(args.capture)
    n_y = reader.GetNumYRows(args.capture)
    n_x = reader.GetNumXColumns(args.capture)

    outfile = args.outfile or default_outfile(slide_path, args.timepoint, args.channel)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    if args.all_time:
        stack = np.zeros((n_t, n_z, n_y, n_x), dtype=np.uint16)
        for t in tqdm(range(n_t), desc="Timepoints"):
            for z in tqdm(range(n_z), desc=f"T{t} Z-planes", leave=False):
                img = reader.ReadImagePlaneBuf(
                    args.capture, args.position, t, z, args.channel, True
                )
                stack[t, z] = img
        axes = "TZYX"
    else:
        if not 0 <= args.timepoint < n_t:
            raise SystemExit(f"TIMEPOINT {args.timepoint} is out of range (0-{n_t-1})")
        stack = np.zeros((n_z, n_y, n_x), dtype=np.uint16)
        for z in tqdm(range(n_z), desc=f"T{args.timepoint} Z-planes"):
            img = reader.ReadImagePlaneBuf(
                args.capture, args.position, args.timepoint, z, args.channel, True
            )
            stack[z] = img
        axes = "ZYX"

    tifffile.imwrite(
        outfile,
        stack,
        photometric="minisblack",
        metadata={"axes": axes},
    )
    print(f"Wrote {outfile}")


if __name__ == "__main__":
    sys.exit(main())
