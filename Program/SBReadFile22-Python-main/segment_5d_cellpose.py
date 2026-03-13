"""
Segment 5D SlideBook (.sldyz) data with Cellpose / CPSAM.

This script reads volumetric time-lapse data from SlideBook files using the
SBReadFile API, converts it to the array layouts expected by Cellpose,
runs 3D or 2D+stitch segmentation, and writes mask volumes to disk.

Key features:
    - Supports 5D TZCYX data (Time, Z, Channel, Y, X) via lazy loading.
    - Uses Cellpose 3D mode when possible, with automatic fallback to
      2D-per-slice + 3D stitching if 3D fails (e.g. out of memory).
    - Lets you pick which channels and timepoints to segment.
    - Can automatically derive anisotropy (Z spacing / XY pixel size)
      from SlideBook metadata when available.
    - Outputs masks as NumPy arrays and/or OME-TIFF label stacks.

Usage example (single capture, all timepoints, default channel 0):

    python segment_5d_cellpose.py \\
        --slide /path/to/data.sldyz \\
        --capture 0 \\
        --position 0 \\
        --all-time \\
        --gpu \\
        --mode auto \\
        --stitch-threshold 0.4 \\
        --output-root /path/to/OUTPUTMASKS

Requires:
    - SBReadFile22-Python-main (this folder) on the Python path.
    - Cellpose >= 4.0 (CPSAM) installed and importable as `cellpose`.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)

from SBReadFile import SBReadFile  # noqa: E402

try:
    from cellpose import models as cp_models  # type: ignore
except Exception as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "Could not import `cellpose`. Make sure Cellpose is installed "
        "in the current environment (e.g. `pip install cellpose`)."
    ) from exc


class SlideBookTimeStack:
    """
    Lazy 5D SlideBook source exposing one Z-stack per timepoint.

    Cellpose detects this object via its `get_time_stack` method and will
    iterate over timepoints internally. Each call returns a 4D array with
    shape (Z, Y, X, C_selected), where C_selected is the number of channels
    you requested.
    """

    def __init__(
        self,
        reader: SBReadFile,
        capture: int,
        position: int = 0,
        channels: Optional[Sequence[int]] = None,
        time_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.reader = reader
        self.capture = int(capture)
        self.position = int(position)

        self._nt = int(self.reader.GetNumTimepoints(self.capture))
        self._nz = int(self.reader.GetNumZPlanes(self.capture))
        self._ny = int(self.reader.GetNumYRows(self.capture))
        self._nx = int(self.reader.GetNumXColumns(self.capture))
        self._nc = int(self.reader.GetNumChannels(self.capture))

        if channels is None:
            channels = [0]
        self.channels: List[int] = sorted({int(c) for c in channels})
        for c in self.channels:
            if c < 0 or c >= self._nc:
                raise ValueError(f"Channel index {c} is out of range [0, {self._nc-1}]")

        if time_indices is None:
            time_indices = list(range(self._nt))
        indices = [int(t) for t in time_indices]
        for t in indices:
            if t < 0 or t >= self._nt:
                raise ValueError(f"Timepoint index {t} is out of range [0, {self._nt-1}]")
        self.time_indices: List[int] = indices

        # Attributes used by Cellpose's time-lapse helper path
        self.nt: int = len(self.time_indices)
        self.axes: str = "TZCYX"
        self.shape: Tuple[int, int, int, int, int] = (
            self.nt,
            self._nz,
            len(self.channels),
            self._ny,
            self._nx,
        )

    def get_time_stack(self, t_index: int) -> np.ndarray:
        """
        Return a 4D stack for a given logical time index.

        Output shape is (Z, Y, X, C), with C=len(self.channels).
        """
        if t_index < 0 or t_index >= len(self.time_indices):
            raise IndexError(f"t_index {t_index} out of range [0, {len(self.time_indices)-1}]")

        t = self.time_indices[t_index]
        n_ch = len(self.channels)
        stack = np.zeros((self._nz, self._ny, self._nx, n_ch), dtype=np.uint16)

        for zi in range(self._nz):
            for ci, ch in enumerate(self.channels):
                plane = self.reader.ReadImagePlaneBuf(
                    self.capture,
                    self.position,
                    t,
                    zi,
                    ch,
                    True,  # return as 2D (Y, X)
                )
                # `plane` is (Y, X), uint16
                stack[zi, :, :, ci] = plane

        return stack


def find_images_root(path: str) -> Optional[str]:
    """
    Walk up the directory tree until an `Images` folder is found.

    This mirrors the layout used by the SlideBook export utilities in this
    repository and allows segmentation outputs to live alongside exported
    TIFFs when desired.
    """
    cur = os.path.abspath(path)
    while cur and cur != os.path.dirname(cur):
        if os.path.basename(cur) == "Images":
            return cur
        cur = os.path.dirname(cur)
    return None


def default_output_root(slide_path: str) -> str:
    """
    Default output root for segmentation masks, given a SlideBook slide path.

    If the slide lives under an `Images` tree, outputs will be placed under
    `Images/OUTPUTMASKS/<stem>`. Otherwise, outputs will be placed next to
    the slide as `<stem>_SEG`.
    """
    stem = os.path.splitext(os.path.basename(slide_path))[0]
    images_root = find_images_root(slide_path)
    if images_root:
        return os.path.join(images_root, "OUTPUTMASKS", stem)
    return os.path.join(os.path.dirname(slide_path), f"{stem}_SEG")


def build_normalize_param(args: argparse.Namespace) -> bool | dict:
    """
    Build the `normalize` argument for CellposeModel.eval from CLI args.
    """
    if args.no_normalize:
        return False

    norm_dict = {}
    # Tri-state norm3d: None = use Cellpose default; True/False override.
    if args.norm3d is not None:
        norm_dict["norm3D"] = bool(args.norm3d)
    if args.tile_norm_blocksize is not None and args.tile_norm_blocksize > 0:
        norm_dict["tile_norm_blocksize"] = int(args.tile_norm_blocksize)

    if norm_dict:
        return norm_dict
    return True


def choose_label_dtype(mask: np.ndarray) -> np.dtype:
    """
    Choose an unsigned integer dtype suitable for writing label images.
    """
    max_label = int(mask.max()) if mask.size else 0
    if max_label < 2**16:
        return np.uint16
    return np.uint32


def release_cellpose_memory() -> None:
    """
    Prompt Python and torch to release transient inference memory.
    """
    gc.collect()
    try:
        torch_mod = getattr(cp_models, "torch", None)
        if torch_mod is None:
            return
        if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()
        mps_mod = getattr(torch_mod, "mps", None)
        if mps_mod is not None and hasattr(mps_mod, "empty_cache"):
            mps_mod.empty_cache()
    except Exception:
        # Best-effort cleanup only.
        pass


def segment_timepoint_stack(
    model: cp_models.CellposeModel,
    stack: np.ndarray,
    args: argparse.Namespace,
    anisotropy: Optional[float],
    common_eval_kwargs: dict,
    timepoint: int,
) -> Tuple[np.ndarray, object, object, str]:
    """
    Segment one timepoint stack and return masks plus bookkeeping info.
    """
    if args.mode in ("3d", "auto"):
        try:
            masks_t, flows_t, styles_t = model.eval(
                stack,
                channel_axis=-1,
                z_axis=0,
                do_3D=True,
                stitch_threshold=0.0,
                anisotropy=anisotropy,
                **common_eval_kwargs,
            )
            return masks_t, flows_t, styles_t, "3d"
        except RuntimeError as exc:
            if args.mode == "3d":
                raise
            print(
                "3D Cellpose segmentation failed for "
                f"timepoint {timepoint}; falling back to 2D+stitch.\n"
                f"Error was: {exc!r}"
            )
            release_cellpose_memory()

    masks_t, flows_t, styles_t = model.eval(
        stack,
        # For 2D/2D+stitch with 4D ZYXC input, keep explicit axes so Cellpose
        # can correctly convert the stack layout before running per-slice dynamics.
        channel_axis=-1,
        z_axis=0,
        do_3D=False,
        stitch_threshold=float(args.stitch_threshold),
        anisotropy=None,
        **common_eval_kwargs,
    )
    used_mode = "2d-stitch" if args.stitch_threshold > 0 else "2d"
    return masks_t, flows_t, styles_t, used_mode


def ensure_position_output_dir(out_root: str, capture: int, position: int) -> str:
    """
    Create and return the capture/position output directory for one run.
    """
    os.makedirs(out_root, exist_ok=True)
    capture_dir = os.path.join(out_root, f"capture_{capture:03d}")
    position_dir = os.path.join(capture_dir, f"position_{position:03d}")
    os.makedirs(position_dir, exist_ok=True)
    return position_dir


def save_timepoint_outputs(
    position_dir: str,
    tp: int,
    masks_t: np.ndarray,
    flows_t,
    save_npy: bool,
    save_ome_tiff: bool,
    save_per_z: bool,
    save_flows: bool,
) -> None:
    """
    Save one timepoint's masks (and optionally flows) to disk.
    """
    if masks_t.ndim == 2:
        masks_t = masks_t[np.newaxis, ...]

    n_z, _n_y, _n_x = masks_t.shape
    t_dir = os.path.join(position_dir, f"timepoint_{tp:04d}")
    os.makedirs(t_dir, exist_ok=True)

    if save_npy:
        np.save(os.path.join(t_dir, f"masks_T{tp:04d}_ZYX.npy"), masks_t)

    if save_ome_tiff:
        dtype = choose_label_dtype(masks_t)
        tifffile.imwrite(
            os.path.join(t_dir, f"masks_T{tp:04d}_ZYX.ome.tif"),
            masks_t.astype(dtype, copy=False),
            dtype=dtype,
            photometric="minisblack",
            metadata={"axes": "ZYX"},
            ome=True,
        )

    if save_per_z:
        dtype = choose_label_dtype(masks_t)
        for z in range(n_z):
            tifffile.imwrite(
                os.path.join(t_dir, f"mask_T{tp:04d}_Z{z:04d}.tif"),
                masks_t[z].astype(dtype, copy=False),
                dtype=dtype,
                photometric="minisblack",
            )

    if save_flows and flows_t is not None:
        np.save(os.path.join(t_dir, f"flows_T{tp:04d}.npy"), flows_t)


def summarize_modes_by_timepoint(mode_by_timepoint: Dict[int, str]) -> str:
    """
    Collapse per-timepoint modes into one run-level summary string.
    """
    unique_modes = sorted(set(mode_by_timepoint.values()))
    if len(unique_modes) == 1:
        return unique_modes[0]
    return "mixed"


def write_segmentation_metadata(
    position_dir: str,
    slide_path: str,
    capture: int,
    position: int,
    time_indices: Sequence[int],
    out_root: str,
    eval_args: dict,
    mode_by_timepoint: Dict[int, str],
) -> None:
    """
    Write a small JSON metadata summary at the capture/position level.
    """
    meta = {
        "slide": slide_path,
        "capture": int(capture),
        "position": int(position),
        "time_indices": list(int(t) for t in time_indices),
        "mode_used": summarize_modes_by_timepoint(mode_by_timepoint),
        "mode_used_by_timepoint": {str(int(tp)): mode for tp, mode in mode_by_timepoint.items()},
        "output_root": out_root,
        "eval_args": eval_args,
    }
    with open(os.path.join(position_dir, "segmentation_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def segment_slidebook(args: argparse.Namespace) -> None:
    """
    Top-level segmentation driver.

    Opens the SlideBook file, creates a lazy 5D source, runs Cellpose, and
    writes masks (and optionally flows) to disk.
    """
    slide_path = os.path.abspath(args.slide)
    reader = SBReadFile()
    if not reader.Open(slide_path):
        raise SystemExit(f"Could not open SlideBook file: {slide_path}")

    capture = int(args.capture)
    position = int(args.position)

    n_captures = reader.GetNumCaptures()
    if capture < 0 or capture >= n_captures:
        raise SystemExit(f"Capture index {capture} out of range [0, {n_captures-1}]")

    nt = reader.GetNumTimepoints(capture)
    if args.all_time:
        time_indices = list(range(nt))
    else:
        tp = int(args.timepoint)
        if tp < 0 or tp >= nt:
            raise SystemExit(f"Timepoint {tp} out of range [0, {nt-1}]")
        time_indices = [tp]

    source = SlideBookTimeStack(
        reader=reader,
        capture=capture,
        position=position,
        channels=args.channels,
        time_indices=time_indices,
    )

    # Diameter: <=0 or None => let Cellpose auto-estimate / use default.
    diameter: Optional[float]
    if args.diameter is None or args.diameter <= 0:
        diameter = None
    else:
        diameter = float(args.diameter)

    # Anisotropy: if not provided, try to infer from voxel size.
    anisotropy: Optional[float] = None
    if args.anisotropy is not None:
        anisotropy = float(args.anisotropy)
    else:
        try:
            vx, vy, vz = reader.GetVoxelSize(capture)
            if vx > 0 and vz > 0:
                anisotropy = float(vz / vx)
        except Exception:
            anisotropy = None

    normalize_param = build_normalize_param(args)

    model = cp_models.CellposeModel(
        gpu=bool(args.gpu),
        pretrained_model=args.model,
    )

    common_eval_kwargs = dict(
        normalize=normalize_param,
        invert=bool(args.invert),
        cellprob_threshold=float(args.cellprob_threshold),
        flow_threshold=float(args.flow_threshold),
        min_size=int(args.min_size),
        diameter=diameter,
        compute_masks=True,
    )

    out_root = args.output_root or default_output_root(slide_path)
    position_dir = ensure_position_output_dir(out_root, capture, position)
    mode_by_timepoint: Dict[int, str] = {}

    for logical_t_index, tp in enumerate(source.time_indices):
        print(
            "Segmenting timepoint "
            f"{tp} ({logical_t_index + 1}/{len(source.time_indices)})"
        )
        stack_t = source.get_time_stack(logical_t_index)
        masks_t = flows_t = styles_t = None
        try:
            masks_t, flows_t, styles_t, used_mode = segment_timepoint_stack(
                model=model,
                stack=stack_t,
                args=args,
                anisotropy=anisotropy,
                common_eval_kwargs=common_eval_kwargs,
                timepoint=int(tp),
            )
            mode_by_timepoint[int(tp)] = used_mode
            save_timepoint_outputs(
                position_dir=position_dir,
                tp=int(tp),
                masks_t=np.asarray(masks_t),
                flows_t=flows_t,
                save_npy=bool(args.save_npy),
                save_ome_tiff=bool(args.save_ome_tiff),
                save_per_z=bool(args.save_per_z),
                save_flows=bool(args.save_flows),
            )
            print(f"Finished timepoint {tp} with mode={used_mode}")
        finally:
            del stack_t
            del masks_t
            del flows_t
            del styles_t
            release_cellpose_memory()

    write_segmentation_metadata(
        position_dir=position_dir,
        slide_path=slide_path,
        capture=capture,
        position=position,
        time_indices=source.time_indices,
        out_root=out_root,
        eval_args=dict(
            mode=args.mode,
            channels=list(source.channels),
            normalize=normalize_param,
            anisotropy=anisotropy,
            diameter=diameter,
            gpu=bool(args.gpu),
            cellprob_threshold=float(args.cellprob_threshold),
            flow_threshold=float(args.flow_threshold),
            min_size=int(args.min_size),
            stitch_threshold=float(args.stitch_threshold),
        ),
        mode_by_timepoint=mode_by_timepoint,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the segmentation script.
    """
    p = argparse.ArgumentParser(
        description="Segment 5D SlideBook (.sldyz) data with Cellpose / CPSAM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--slide",
        required=True,
        help="Path to the SlideBook .sldyz file.",
    )
    p.add_argument(
        "--capture",
        type=int,
        default=0,
        help="Capture (image group) index to segment.",
    )
    p.add_argument(
        "--position",
        type=int,
        default=0,
        help="Montage position index to segment.",
    )
    p.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[0],
        help="Channel indices to include in the input stack (up to 3 will be used).",
    )
    p.add_argument(
        "--timepoint",
        type=int,
        default=0,
        help="Single timepoint index to segment (ignored if --all-time is set).",
    )
    p.add_argument(
        "--all-time",
        action="store_true",
        help="Segment all timepoints instead of a single timepoint.",
    )
    p.add_argument(
        "--model",
        default="cpsam",
        help="Cellpose pretrained model to use (path or model name).",
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available.",
    )
    p.add_argument(
        "--mode",
        choices=["auto", "3d", "2d"],
        default="auto",
        help="Segmentation mode: 3d (pure 3D), 2d (per-slice + optional stitching), or auto (3D with 2D fallback).",
    )
    p.add_argument(
        "--stitch-threshold",
        type=float,
        default=0.4,
        help="3D stitching threshold for 2D mode (set >0 to merge masks across Z).",
    )
    p.add_argument(
        "--diameter",
        type=float,
        default=0.0,
        help="Approximate cell diameter in pixels. Set <=0 to let Cellpose auto-estimate.",
    )
    p.add_argument(
        "--anisotropy",
        type=float,
        default=None,
        help="Z anisotropy (Z spacing / XY pixel size). If omitted, attempts to infer from SlideBook metadata.",
    )
    p.add_argument(
        "--cellprob-threshold",
        type=float,
        default=0.0,
        help="Cell probability threshold (higher = fewer, more confident masks).",
    )
    p.add_argument(
        "--flow-threshold",
        type=float,
        default=0.4,
        help="Flow error threshold (used in 2D mode).",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=15,
        help="Discard masks smaller than this number of pixels.",
    )
    p.add_argument(
        "--invert",
        action="store_true",
        help="Invert image intensities before segmentation.",
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable Cellpose intensity normalization.",
    )
    p.add_argument(
        "--norm3d",
        dest="norm3d",
        action="store_true",
        help="Force 3D normalization across Z.",
    )
    p.add_argument(
        "--norm2d",
        dest="norm3d",
        action="store_false",
        help="Force per-slice (2D) normalization along Z.",
    )
    p.set_defaults(norm3d=None)
    p.add_argument(
        "--tile-norm-blocksize",
        type=int,
        default=0,
        help="Enable tile-based normalization with this block size (0 = disabled).",
    )
    p.add_argument(
        "--output-root",
        help="Root directory for segmentation outputs. If omitted, a default under Images/OUTPUTMASKS or alongside the slide is used.",
    )
    p.add_argument(
        "--no-npy",
        dest="save_npy",
        action="store_false",
        help="Disable saving masks as .npy arrays.",
    )
    p.add_argument(
        "--save-ome-tiff",
        action="store_true",
        help="Save masks as OME-TIFF label stacks (ZYX).",
    )
    p.add_argument(
        "--save-per-z",
        action="store_true",
        help="Also save one 2D TIFF per Z-slice.",
    )
    p.add_argument(
        "--save-flows",
        action="store_true",
        help="Save Cellpose flow fields per timepoint as .npy.",
    )

    args = p.parse_args(argv)
    # default: save_npy is True unless explicitly disabled
    if not hasattr(args, "save_npy"):
        args.save_npy = True
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    segment_slidebook(args)


if __name__ == "__main__":  # pragma: no cover
    main()
