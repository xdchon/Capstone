"""
Build 4D or 5D TIFF Z-stacks from a SlideBook .sldyz or from extracted plane TIFFs.

- 5D mode (default): one file per capture, axes TZCYX (Time, Z, Channel, Y, X), streamed to disk as ImageJ hyperstack BigTIFF (.ome.btf).
- 4D mode: one file per timepoint, axes ZCYX (Z, Channel, Y, X) as OME-TIFF.

Output layout:
- 5D: Images/OUTPUTIMAGES/<stem>/capture_XXX/capture_XXX_TZCYX.ome.btf
- 4D: Images/OUTPUTIMAGES/<stem>/capture_XXX/timepoint_TTTT_ZCYX.ome.tif
"""

import argparse
import os
import sys
from glob import glob

import numpy as np
import tifffile
from tqdm import tqdm

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.append(HERE)
from SBReadFile import SBReadFile  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="Export TIFF Z-stacks (4D ZCYX per timepoint or 5D TZCYX per capture) from .sldyz or extracted planes."
    )
    p.add_argument(
        "input",
        help="Path to .sldyz file OR to a capture root under OUTPUTIMAGES (e.g. Images/OUTPUTIMAGES/NIben...DCN)",
    )
    p.add_argument("--capture", type=int, help="Capture index (default: all captures)")
    p.add_argument("--position", type=int, default=0, help="Montage position index (default: 0)")
    p.add_argument(
        "--mode",
        choices=["4d", "5d"],
        default="5d",
        help="4d = ZCYX per timepoint; 5d = TZCYX single file per capture (default: 5d)",
    )
    p.add_argument(
        "--outdir",
        help="Override output root. By default uses OUTPUTIMAGES/<stem> (for .sldyz) or the provided frames root.",
    )
    p.add_argument(
        "--from-frames",
        action="store_true",
        help="Force treating input as already-extracted planes instead of .sldyz",
    )
    return p.parse_args()


def find_images_root(path):
    cur = os.path.abspath(path)
    while cur and cur != os.path.dirname(cur):
        if os.path.basename(cur) == "Images":
            return cur
        cur = os.path.dirname(cur)
    return None


def default_outdir_for_slide(slide_path):
    stem = os.path.splitext(os.path.basename(slide_path))[0]
    images_root = find_images_root(slide_path)
    if images_root:
        return os.path.join(images_root, "OUTPUTIMAGES", stem)
    return os.path.join(os.path.dirname(slide_path), f"{stem}_OUTPUT")


# ------------------ SlideBook path -----------------------------------------
def export_from_slidebook(slide_path, capture_idx, position_idx, outdir, mode):
    reader = SBReadFile()
    if not reader.Open(slide_path):
        raise SystemExit(f"Could not open slide: {slide_path}")

    captures = [capture_idx] if capture_idx is not None else range(reader.GetNumCaptures())
    for cap in captures:
        _export_capture_from_slidebook(reader, cap, position_idx, outdir or default_outdir_for_slide(slide_path), mode)


def _export_capture_from_slidebook(reader, capture_idx, position_idx, root_outdir, mode):
    n_t = reader.GetNumTimepoints(capture_idx)
    n_z = reader.GetNumZPlanes(capture_idx)
    n_c = reader.GetNumChannels(capture_idx)
    n_y = reader.GetNumYRows(capture_idx)
    n_x = reader.GetNumXColumns(capture_idx)

    capture_dir = os.path.join(root_outdir, f"capture_{capture_idx:03d}")
    os.makedirs(capture_dir, exist_ok=True)

    if mode == "5d":
        outfile = os.path.join(capture_dir, f"capture_{capture_idx:03d}_TCZYX.ome.btf")

        def plane_iter():
            # order: T, C, Z to match axes TCZYX
            for t in range(n_t):
                for c in range(n_c):
                    for z in range(n_z):
                        yield reader.ReadImagePlaneBuf(capture_idx, position_idx, t, z, c, True)

        tifffile.imwrite(
            outfile,
            data=plane_iter(),
            shape=(n_t, n_c, n_z, n_y, n_x),
            dtype=np.uint16,
            photometric="minisblack",
            metadata={"axes": "TCZYX"},
            ome=True,
            bigtiff=True,
        )
    else:  # 4d -> one file per timepoint
        for t in tqdm(range(n_t), desc=f"[SlideBook] capture {capture_idx} T"):
            stack = np.zeros((n_z, n_c, n_y, n_x), dtype=np.uint16)
            for z in range(n_z):
                for c in range(n_c):
                    stack[z, c] = reader.ReadImagePlaneBuf(capture_idx, position_idx, t, z, c, True)
            outfile = os.path.join(capture_dir, f"timepoint_{t:04d}_ZCYX.ome.tif")
            tifffile.imwrite(outfile, stack, photometric="minisblack", metadata={"axes": "ZCYX"}, ome=True, bigtiff=True)


# ------------------ Already-extracted frames path -------------------------
def export_from_frames(frames_root, capture_idx, position_idx, outdir_override, mode):
    if capture_idx is None:
        captures = sorted(d for d in os.listdir(frames_root) if d.startswith("capture_"))
    else:
        captures = [f"capture_{capture_idx:03d}"]

    for cap_name in captures:
        cap_dir = os.path.join(frames_root, cap_name)
        if not os.path.isdir(cap_dir):
            print(f"Skipping missing {cap_dir}")
            continue
        pos_dir = os.path.join(cap_dir, f"position_{position_idx:03d}")
        if not os.path.isdir(pos_dir):
            print(f"Skipping missing {pos_dir}")
            continue
        channel_dirs = sorted(d for d in os.listdir(pos_dir) if d.startswith("channel_"))
        if not channel_dirs:
            print(f"No channels in {pos_dir}, skipping")
            continue

        timepoints = sorted(
            d for d in os.listdir(os.path.join(pos_dir, channel_dirs[0])) if d.startswith("timepoint_")
        )
        out_root = outdir_override or cap_dir
        os.makedirs(out_root, exist_ok=True)

        first_tp_dir = os.path.join(pos_dir, channel_dirs[0], timepoints[0])
        first_planes = sorted(glob(os.path.join(first_tp_dir, "z_*.tiff")))
        if not first_planes:
            raise RuntimeError(f"No z_*.tiff files in {first_tp_dir}")
        sample = tifffile.imread(first_planes[0])
        n_z = len(first_planes)
        n_c = len(channel_dirs)
        n_y, n_x = sample.shape

        if mode == "5d":
            outfile = os.path.join(out_root, f"{cap_name}_TCZYX.ome.btf")

            def plane_iter():
                # order: T, C, Z to match axes TCZYX
                for tp in timepoints:
                    for ci, ch_dir in enumerate(channel_dirs):
                        for z in range(n_z):
                            pf = os.path.join(pos_dir, ch_dir, tp, f"z_{z:04d}.tiff")
                            yield tifffile.imread(pf)

            tifffile.imwrite(
                outfile,
                data=plane_iter(),
                shape=(len(timepoints), n_c, n_z, n_y, n_x),
                dtype=sample.dtype,
                photometric="minisblack",
                metadata={"axes": "TCZYX"},
                ome=True,
                bigtiff=True,
            )
        else:  # 4d
            for tp in tqdm(timepoints, desc=f"[Frames] {cap_name} T"):
                stack = np.zeros((n_z, n_c, n_y, n_x), dtype=sample.dtype)
                for ci, ch_dir in enumerate(channel_dirs):
                    tp_dir = os.path.join(pos_dir, ch_dir, tp)
                    plane_files = sorted(glob(os.path.join(tp_dir, "z_*.tiff")))
                    if len(plane_files) != n_z:
                        raise RuntimeError(f"Mismatched Z count in {tp_dir}")
                    for zi, pf in enumerate(plane_files):
                        stack[zi, ci] = tifffile.imread(pf)
                outfile = os.path.join(out_root, f"{tp}_ZCYX.ome.tif")
                tifffile.imwrite(outfile, stack, photometric="minisblack", metadata={"axes": "ZCYX"}, ome=True, bigtiff=True)


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input)

    if args.from_frames or not input_path.lower().endswith(".sldyz"):
        export_from_frames(
            frames_root=input_path,
            capture_idx=args.capture,
            position_idx=args.position,
            outdir_override=os.path.abspath(args.outdir) if args.outdir else None,
            mode=args.mode,
        )
    else:
        export_from_slidebook(
            slide_path=input_path,
            capture_idx=args.capture,
            position_idx=args.position,
            outdir=os.path.abspath(args.outdir) if args.outdir else None,
            mode=args.mode,
        )


if __name__ == "__main__":
    main()
