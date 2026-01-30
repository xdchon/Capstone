__copyright__  = "Copyright (c) 2022-2025, Intelligent Imaging Innovations, Inc. All rights reserved.  All rights reserved."
__license__  = "This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree."

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import argparse
import os
import re
import sys
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    import matlab.engine  # type: ignore
except Exception:  # pragma: no cover
    matlab = None


from CNpyHeader import CNpyHeader
from CCompressionBase import CCompressionBase

# In[25]:

def _windows_to_wsl_path(p: str) -> str:
    """
    Convert 'E:\\path\\to\\file' -> '/mnt/e/path/to/file' when running on POSIX.
    If the path doesn't look like a Windows drive path, return as-is.
    """
    if os.name == "nt":
        return p
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if not m:
        return p
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def read_npyz_plane(file_path: str, block: int = 0) -> np.ndarray:
    """Return one (planes, rows, cols) block from a compressed .npyz file."""
    with open(file_path, "rb") as stream:
        header = CNpyHeader()
        if not header.ParseNpyHeader(stream):
            raise RuntimeError(f"Cannot parse header from {file_path}")

        # Determine dimensions
        shape = header.mShape
        num_dim = len(shape)
        if num_dim == 4:
            num_blocks, num_planes, num_rows, num_cols = shape
            if block < 0 or block >= num_blocks:
                raise IndexError(f"Block out of range: {block} (valid: 0..{num_blocks-1})")
        else:
            num_blocks = 1
            num_planes, num_rows, num_cols = shape
            block = 0

        # Initialize compressor and read dictionary
        compressor = CCompressionBase()
        if num_dim == 4:
            compressor.InitializeEx(header.mHeaderSize, header.mCompressionFlag,
                                    num_cols, num_rows, num_planes, num_blocks, 0)
        else:
            compressor.Initialize(header.mHeaderSize, header.mCompressionFlag,
                                  num_cols, num_rows, num_planes, 0)
        compressor.ReadDictionary(stream)

        # Locate compressed block and decompress
        offset = compressor.GetDataOffsetForBlock(block)
        size = compressor.GetDataSizeForBlock(block)
        stream.seek(offset, 0)
        compressed = stream.read(size)
        uncompressed = compressor.DecompressBuffer(compressed)

    arr = np.frombuffer(uncompressed, dtype=np.uint16)
    arr = arr.reshape(num_planes, num_rows, num_cols)
    return arr


def _list_capture_titles(slide_root: str) -> list[str]:
    try:
        entries = os.listdir(slide_root)
    except FileNotFoundError:
        return []
    titles: list[str] = []
    for e in entries:
        if e.endswith(".imgdir"):
            titles.append(e[:-6])
    return sorted(titles)


def open_npy_file(path: str, cap_number: str, channel: str, *, block: int = 0) -> np.ndarray:
    """Open a specific channel array from a capture directory.

    - path: directory ending with .dir for the slide root
    - cap_number: exact capture title as used on disk (without .imgdir suffix)
    - channel: channel label, e.g. 'Ch0', 'Ch1', 'Ch2'
    """
    # Find the desired capture directory exactly
    capture_dir_name = f"{cap_number}.imgdir"
    try:
        entries = os.listdir(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Slide root not found: {path}")

    b = None
    for i in entries:
        if i == capture_dir_name:
            b = i
            break
    if b is None:
        titles = _list_capture_titles(path)
        suggestions = get_close_matches(cap_number, titles, n=5, cutoff=0.2)
        hint = ""
        if suggestions:
            hint = "\nDid you mean:\n  - " + "\n  - ".join(suggestions)
        raise FileNotFoundError(f"Capture not found: {capture_dir_name} in {path}{hint}")

    new_path = os.path.join(path, str(b))

    # Find files for the channel, prefer lowest timepoint if multiple
    candidates = []
    for q in os.listdir(new_path):
        if not q.startswith("ImageData"):
            continue
        if f"_{channel}_" not in q:
            continue
        if not (q.endswith(".npy") or q.endswith(".npyz")):
            continue
        m = re.search(r"_TP(\d{7})", q)
        tp = int(m.group(1)) if m else 0
        candidates.append((tp, q))

    if not candidates:
        raise FileNotFoundError(f"No .npy files found for {channel} in {new_path}")

    candidates.sort(key=lambda x: x[0])
    c = candidates[0][1]
    # opens the npy file that has your channel data in it
    full_path = os.path.join(new_path, c)
    print("Opening", full_path)
    if full_path.endswith(".npyz"):
        im = read_npyz_plane(full_path, block=block)
    else:
        im = np.load(full_path)
    return im


# In[29]:


def to2d(a: np.ndarray) -> np.ndarray:
    # Ensure a 2D (rows, cols) plane; if more dims, take first plane/timepoint
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        return a[0, :, :]
    if a.ndim >= 4:
        return a.reshape((-1, a.shape[-2], a.shape[-1]))[0]
    return a


def _connect_matlab(session: Optional[str]) -> "matlab.engine.MatlabEngine":
    if matlab is None:  # type: ignore[name-defined]
        raise RuntimeError(
            "matlab.engine is not importable in this Python environment.\n"
            "If you're on WSL, you typically cannot use the Windows MATLAB Engine from Linux Python.\n"
            "Run this script with the same Windows Python that has the MATLAB Engine installed, or install MATLAB for Linux."
        )

    if session:
        return matlab.engine.connect_matlab(session)  # type: ignore[attr-defined]

    names = matlab.engine.find_matlab()  # type: ignore[attr-defined]
    if not names:
        raise RuntimeError(
            "No shared MATLAB sessions found.\n"
            "In MATLAB, run: matlab.engine.shareEngine('Python')\n"
            "Then run this script with: --matlab-session Python"
        )
    return matlab.engine.connect_matlab(names[0])  # type: ignore[attr-defined]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load channels from a SlideBook .dir export and push an RGB stack into MATLAB."
    )
    parser.add_argument("--slide-dir", default=None, help="Path to the slide '.dir' root.")
    parser.add_argument("--capture-title", default=None, help="Capture title (folder name without '.imgdir').")
    parser.add_argument("--channels", nargs=3, default=["Ch0", "Ch1", "Ch2"], help="3 channel labels for RGB.")
    parser.add_argument("--block", type=int, default=0, help="Block/timepoint index for .npyz (default: 0).")
    parser.add_argument("--matlab-session", default="Python", help="Shared MATLAB session name (default: Python).")
    parser.add_argument("--matlab-var", default="rgb", help="Variable name to create in MATLAB (default: rgb).")
    parser.add_argument("--no-matlab", action="store_true", help="Only load/compose RGB; don't connect to MATLAB.")
    args = parser.parse_args()

    # Keep the original variables as a fallback for users who edit the script directly.
    slide_dir = args.slide_dir or r"E:\Capstone Project - 1 Batch Files\Images\John\NIben_19_TriPer_CAT_TS_ON_v2_Capt3_DCN.dir"
    capture_title = args.capture_title or "Capture 3 MLS Decon MIP XY-1759482556-506"

    slide_dir = _windows_to_wsl_path(slide_dir)
    slide_dir = str(Path(slide_dir))

    # Load channels and compose RGB
    r_arr = open_npy_file(slide_dir, capture_title, args.channels[0], block=args.block)
    g_arr = open_npy_file(slide_dir, capture_title, args.channels[1], block=args.block)
    b_arr = open_npy_file(slide_dir, capture_title, args.channels[2], block=args.block)

    r2 = to2d(r_arr)
    g2 = to2d(g_arr)
    b2 = to2d(b_arr)

    rgb = np.dstack([r2, g2, b2]).astype(np.uint16, copy=False)
    rgb = np.ascontiguousarray(rgb)

    if args.no_matlab:
        print(f"Loaded RGB array: shape={rgb.shape} dtype={rgb.dtype}")
        return 0

    eng = _connect_matlab(args.matlab_session)
    eng.workspace[args.matlab_var] = rgb
    print(f"Sent to MATLAB workspace as variable '{args.matlab_var}': shape={rgb.shape} dtype={rgb.dtype}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
