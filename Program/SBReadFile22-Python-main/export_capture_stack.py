"""Export a SlideBook capture as one OME-TIFF stack.

Output ordering is written as T, P, C, Z, Y, X planes.
"""

from pathlib import Path
import argparse
import sys
from tifffile import TiffWriter

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from SBReadFile import SBReadFile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one SlideBook capture to a single OME-TIFF stack."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input .sldy/.sldyz slide.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output OME-TIFF path. Defaults next to input slide.",
    )
    parser.add_argument(
        "--capture",
        type=int,
        default=0,
        help="Capture index to export.",
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=[0],
        help="Position indices to include (default: 0).",
    )
    parser.add_argument(
        "--first-timepoint",
        type=int,
        default=0,
        help="First timepoint to export (inclusive).",
    )
    parser.add_argument(
        "--last-timepoint",
        type=int,
        default=None,
        help="Last timepoint to export (inclusive). Defaults to end.",
    )
    return parser.parse_args()


def export_capture(args: argparse.Namespace) -> None:
    slide_path = Path(args.input).expanduser().resolve()
    if not slide_path.exists():
        raise SystemExit(f"Slide not found: {slide_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else slide_path.with_name(f"{slide_path.stem}_capture_{args.capture:03d}.ome.tif")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = SBReadFile()
    if not reader.Open(str(slide_path)):
        raise SystemExit(f"Could not open slide: {slide_path}")

    capture_idx = int(args.capture)
    num_captures = int(reader.GetNumCaptures())
    if capture_idx < 0 or capture_idx >= num_captures:
        raise SystemExit(
            f"Capture index {capture_idx} out of range [0, {num_captures - 1}]"
        )

    n_t = int(reader.GetNumTimepoints(capture_idx))
    n_z = int(reader.GetNumZPlanes(capture_idx))
    n_c = int(reader.GetNumChannels(capture_idx))
    n_pos = int(reader.GetNumPositions(capture_idx))

    start_t = max(0, int(args.first_timepoint))
    end_t = n_t - 1 if args.last_timepoint is None else min(int(args.last_timepoint), n_t - 1)
    if start_t > end_t:
        raise SystemExit(
            f"Invalid timepoint range: first={start_t}, last={end_t}, available=0..{n_t - 1}"
        )

    positions = []
    for pos in args.positions:
        if 0 <= pos < n_pos:
            positions.append(pos)
        else:
            print(f"[warn] Skipping out-of-range position index {pos} (valid: 0..{n_pos - 1})")
    if not positions:
        raise SystemExit("No valid positions selected.")

    with TiffWriter(str(output_path), ome=True) as tif:
        for idx_t, t in enumerate(range(start_t, end_t + 1)):
            for idx_p, pos in enumerate(positions):
                for channel in range(n_c):
                    for z in range(n_z):
                        plane = reader.ReadImagePlaneBuf(capture_idx, pos, t, z, channel, True)
                        tif.write(
                            plane,
                            photometric="minisblack",
                            contiguous=True,
                            metadata={
                                "axes": "TPCZYX",
                                "TimePoint": idx_t,
                                "Position": idx_p,
                                "Channel": channel,
                                "Slice": z,
                            },
                        )

    print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    export_capture(args)


if __name__ == "__main__":
    main()
