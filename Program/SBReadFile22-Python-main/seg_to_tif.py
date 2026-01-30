import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tifffile


def load_masks(
    npy_path: Path, key: str, time_index: Optional[int] = None
) -> Tuple[np.ndarray, bool]:
    """
    Load a mask array from a Cellpose-style .npy file.

    Supports:
      - Plain numpy arrays saved via np.save(masks)
      - Dict-like seg files with a top-level key (e.g. {'masks': ...})
      - Time-lapse seg files with an aggregated structure:
            {'per_time': {t: {'masks': ..., ...}, ...}, 'filename': ..., 'NT': ...}
        In this case, a specific timepoint is selected via `time_index`.
    """
    data = np.load(npy_path, allow_pickle=True)
    multi_time = False

    # Unwrap dict-like objects saved via np.save on an object array.
    if isinstance(data, np.ndarray) and data.dtype == object:
        try:
            data = data.item()
        except Exception:
            # fall back to treating as a plain array below
            pass

    # Aggregated time-lapse seg file with per-time dictionaries
    # produced by the .sldy-aware Cellpose GUI.
    if isinstance(data, dict) and "per_time" in data and isinstance(data["per_time"], dict):
        per_time = data["per_time"]

        # If a specific timepoint is requested, behave like the original
        # single-time loader and return only that entry.
        if time_index is not None:
            t_sel = time_index
            entry = None
            # keys might be ints, numpy ints, or strings
            if t_sel in per_time:
                entry = per_time[t_sel]
            elif str(t_sel) in per_time:
                entry = per_time[str(t_sel)]
            else:
                # search by stored 'time_index' field inside each dict
                for v in per_time.values():
                    if isinstance(v, dict):
                        try:
                            if int(v.get("time_index", -1)) == int(t_sel):
                                entry = v
                                break
                        except Exception:
                            continue
            if entry is None:
                # no usable entry for the requested timepoint
                raise KeyError(
                    f"time_index {t_sel} not found in per_time mapping of {npy_path}"
                )
            if not isinstance(entry, dict):
                raise ValueError(f"per_time entry in {npy_path} is not a dict")
            if key not in entry:
                raise KeyError(f"Key '{key}' not found in per_time entry of {npy_path}")
            data = entry[key]
            return np.asarray(data), False

        # Otherwise, stack ALL timepoints into a single array with a leading
        # T dimension so we can export a 5D TIFF (TZCYX).
        if len(per_time) == 0:
            raise ValueError(f"No per_time entries found in {npy_path}")

        # Build a sorted list of (time_index, entry) pairs.
        entries = []
        for k, v in per_time.items():
            if not isinstance(v, dict):
                continue
            try:
                t_val = int(v.get("time_index", k))
            except Exception:
                continue
            entries.append((t_val, v))
        if not entries:
            # fall back to insertion order if no usable time_index values
            entries = []
            for idx, (k, v) in enumerate(per_time.items()):
                if isinstance(v, dict):
                    entries.append((idx, v))

        entries.sort(key=lambda kv: kv[0])

        arrays = []
        for _, entry in entries:
            if key not in entry:
                raise KeyError(f"Key '{key}' not found in per_time entry of {npy_path}")
            arr = np.asarray(entry[key])
            # ensure we always have at least Z,Y,X for stacking
            if arr.ndim == 2:  # Y,X
                arr = arr[np.newaxis, ...]  # 1,Z,Y,X
            arrays.append(arr)

        if not arrays:
            raise ValueError(f"No valid per_time entries with key '{key}' in {npy_path}")

        first_shape = arrays[0].shape
        for arr in arrays[1:]:
            if arr.shape != first_shape:
                raise ValueError(
                    f"Per-time arrays have inconsistent shapes: {first_shape} vs {arr.shape}"
                )

        # Final stacked shape: (T, Z, Y, X) or (T, Z, ..., Y, X)
        data = np.stack(arrays, axis=0)
        multi_time = True

    elif isinstance(data, dict):
        # Classic seg dict produced by Cellpose CLI / scripts.
        # Here `data[key]` may be:
        #   - a single 2D/3D array (single timepoint)
        #   - a list/tuple of 2D/3D arrays (one entry per timepoint)
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {npy_path}")
        value = data[key]

        # List-of-arrays format: "one mask after another" for each timepoint.
        if isinstance(value, (list, tuple)):
            arrays = []
            for arr in value:
                arr = np.asarray(arr)
                if arr.ndim == 2:
                    # Y,X -> 1,Z,Y,X
                    arr = arr[np.newaxis, ...]
                elif arr.ndim != 3:
                    raise ValueError(
                        f"Unsupported mask array with {arr.ndim} dimensions in list from {npy_path}"
                    )
                arrays.append(arr)

            if not arrays:
                raise ValueError(f"No mask arrays found in list for key '{key}' in {npy_path}")

            first_shape = arrays[0].shape
            for arr in arrays[1:]:
                if arr.shape != first_shape:
                    raise ValueError(
                        f"Inconsistent mask shapes in list for key '{key}' in {npy_path}: "
                        f"{first_shape} vs {arr.shape}"
                    )

            # Stack along leading axis to form (T, Z, Y, X)
            data = np.stack(arrays, axis=0)
            multi_time = True
        else:
            data = value

    return np.asarray(data), multi_time


def infer_axes(array):
    if array.ndim == 2:
        return "YX"
    if array.ndim == 3:
        return "ZYX"
    if array.ndim == 4:
        return "ZCYX"
    raise ValueError(f"Unsupported array with {array.ndim} dimensions")


def main():
    parser = argparse.ArgumentParser(description="Convert segmentation .npy files to TIFF.")
    parser.add_argument("npy_path", help="Path to the segmentation .npy file (dict with 'masks' supported)")
    parser.add_argument("-o", "--output", help="Output TIFF path. Defaults to input name with .tif extension.")
    parser.add_argument("--key", default="masks", help="Dictionary key to read when the .npy stores multiple entries.")
    parser.add_argument(
        "--time-index",
        type=int,
        help=(
            "Time index to export when the .npy file contains an aggregated "
            "time-lapse structure with a 'per_time' mapping (as used for .sldy files). "
            "If omitted and only a single timepoint is present, that entry is used."
        ),
    )
    args = parser.parse_args()

    npy_path = Path(args.npy_path)
    output_path = Path(args.output) if args.output else npy_path.with_suffix(".tif")

    masks, multi_time = load_masks(npy_path, args.key, time_index=args.time_index)

    # For aggregated time-lapse seg files with no explicit --time-index,
    # we stack all timepoints and write a single 4D TIFF with axes TZYX
    # (time, z, y, x) so you get a Z-stack for each timepoint.
    if multi_time:
        # Expected common case for Cellpose .sldy seg: (T, Z, Y, X)
        if masks.ndim == 4:
            # already (T, Z, Y, X): nothing to change
            pass
        elif masks.ndim == 3:
            # (T, Y, X) -> (T, 1, Y, X) with a singleton Z
            masks = masks[:, np.newaxis, ...]
        else:
            raise ValueError(
                f"Expected masks with shape (T,Z,Y,X) or (T,Y,X) for multi-time export, got {masks.shape}"
            )

        data = masks.astype(np.uint16, copy=False)
        axes = "TZYX"
    else:
        data = masks.astype(np.uint16, copy=False)
        axes = infer_axes(data)

    tifffile.imwrite(output_path, data, ome=True, metadata={"axes": axes})


if __name__ == "__main__":
    main()
