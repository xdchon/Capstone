"""
Export every plane in a SlideBook 7 file to individual TIFF images.

Each output file is named with capture/position/channel/timepoint/z indices and
is stored under:
    <output_root>/<slide_stem>/capture_<id>/

Example:
python export_all_planes.py -i "/path/to/Slide.sldyz"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import tifffile
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from SBReadFile import SBReadFile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SlideBook planes as TIFF images.")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the .sldy/.sldyz file to export.",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        default=None,
        help="Root directory for exported images. Defaults to Images/OUTPUTIMAGES if an Images parent exists, otherwise <slide_dir>/OUTPUTIMAGES.",
    )
    parser.add_argument(
        "--captures",
        type=int,
        nargs="+",
        help="Capture indices to export (default: all captures).",
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        help="Position indices to export (default: all positions per capture).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        help="Channel indices to export (default: all channels).",
    )
    parser.add_argument(
        "--timepoints",
        type=int,
        nargs="+",
        help="Timepoint indices to export (default: all timepoints).",
    )
    parser.add_argument(
        "--zplanes",
        type=int,
        nargs="+",
        help="Z-plane indices to export (default: all planes).",
    )
    return parser.parse_args()


def default_output_root(slide_path: Path) -> Path:
    for parent in slide_path.parents:
        if parent.name.lower() == "images":
            return parent / "OUTPUTIMAGES"
    return slide_path.parent / "OUTPUTIMAGES"


def normalize_indices(
    requested: Optional[Iterable[int]], total: int, label: str
) -> List[int]:
    if total == 0:
        return []
    if requested is None:
        return list(range(total))

    result: List[int] = []
    for idx in requested:
        if 0 <= idx < total:
            result.append(idx)
        else:
            print(f"[warn] Skipping {label} index {idx}: valid range 0-{total-1}")
    if not result:
        print(f"[warn] No valid {label} indices requested; defaulting to full range.")
        return list(range(total))
    return sorted(set(result))


def export_slide(args: argparse.Namespace) -> None:
    slide_path = Path(args.input)
    if not slide_path.exists():
        raise SystemExit(f"Slide not found: {slide_path}")

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else default_output_root(slide_path)
    )
    slide_output_dir = output_root / slide_path.stem
    slide_output_dir.mkdir(parents=True, exist_ok=True)

    reader = SBReadFile()
    if not reader.Open(str(slide_path)):
        raise SystemExit(f"Failed to open slide: {slide_path}")

    num_captures = reader.GetNumCaptures()
    capture_indices = normalize_indices(args.captures, num_captures, "capture")

    for capture in capture_indices:
        print(f"[info] Starting capture {capture}")
        cap_dir = slide_output_dir / f"capture_{capture:03d}"
        cap_dir.mkdir(parents=True, exist_ok=True)

        num_positions = reader.GetNumPositions(capture)
        num_channels = reader.GetNumChannels(capture)
        num_timepoints = reader.GetNumTimepoints(capture)
        num_planes = reader.GetNumZPlanes(capture)

        position_indices = normalize_indices(args.positions, num_positions, "position")
        channel_indices = normalize_indices(args.channels, num_channels, "channel")
        time_indices = normalize_indices(args.timepoints, num_timepoints, "timepoint")
        plane_indices = normalize_indices(args.zplanes, num_planes, "z-plane")

        total_planes = (
            len(position_indices)
            * len(channel_indices)
            * len(time_indices)
            * len(plane_indices)
        )

        print(
            f"[info] Capture {capture}: "
            f"{len(position_indices)} positions, "
            f"{len(channel_indices)} channels, "
            f"{len(time_indices)} timepoints, "
            f"{len(plane_indices)} planes",
        )

        with tqdm(total=total_planes, desc=f"Capture {capture}", unit="plane") as pbar:
            for pos in position_indices:
                pos_dir = cap_dir / f"position_{pos:03d}"
                pos_dir.mkdir(exist_ok=True)

                for channel in channel_indices:
                    channel_dir = pos_dir / f"channel_{channel:02d}"
                    channel_dir.mkdir(exist_ok=True)

                    for timepoint in time_indices:
                        tp_dir = channel_dir / f"timepoint_{timepoint:04d}"
                        tp_dir.mkdir(exist_ok=True)

                        for plane in plane_indices:
                            plane_path = tp_dir / f"z_{plane:04d}.tiff"
                            image = reader.ReadImagePlaneBuf(
                                capture, pos, timepoint, plane, channel, True
                            )
                            tifffile.imwrite(
                                plane_path,
                                image,
                                photometric="minisblack",
                            )
                            pbar.set_postfix(
                                pos=pos,
                                ch=channel,
                                t=timepoint,
                                z=plane,
                            )
                            pbar.update(1)
        print(f"[info] Finished capture {capture}, outputs under {cap_dir}")


def main() -> None:
    args = parse_args()
    export_slide(args)


if __name__ == "__main__":
    main()
