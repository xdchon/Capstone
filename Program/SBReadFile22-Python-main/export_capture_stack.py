"""
Export a SlideBook capture as a single ImageJ-compatible TIFF stack.
"""

import sys
import numpy as np
import tifffile
from tifffile import TiffWriter

# --- configuration ---------------------------------------------------------
SLIDE_PATH = r"E:\Capstone Project - 1 Batch Files\Images\John\NIben_19_TriPer_CAT_TS_ON_v2_Capt3_DCN.sldyz"
OUTPUT_PATH = r"E:\Capstone Project - 1 Batch Files\Images\OUTPUTIMAGES\NIben_19_TriPer_CAT_TS_ON_v2_Capt3_DCN_TIFF.ome.tiff"

CAPTURE_INDEX = 0              # use 0 unless your slide has multiple captures
POSITION_INDICES = [0]         # e.g. range(num_positions) if you want all positions
FIRST_TIMEPOINT = 0            # inclusive
LAST_TIMEPOINT = None          # inclusive; None means “all the way to the end”

# --- main -----------------------------------------------------------------
sys.path.append(r"E:\Capstone Project - 1 Batch Files\Program\SBReadFile22-Python-main")
from SBReadFile import SBReadFile  # noqa: E402


def export_capture():
    reader = SBReadFile()
    if not reader.Open(SLIDE_PATH):
        raise SystemExit(f"Could not open slide: {SLIDE_PATH}")

    n_t = reader.GetNumTimepoints(CAPTURE_INDEX)
    n_z = reader.GetNumZPlanes(CAPTURE_INDEX)
    n_c = reader.GetNumChannels(CAPTURE_INDEX)
    n_y = reader.GetNumYRows(CAPTURE_INDEX)
    n_x = reader.GetNumXColumns(CAPTURE_INDEX)

    start_t = FIRST_TIMEPOINT
    end_t = n_t - 1 if LAST_TIMEPOINT is None else min(LAST_TIMEPOINT, n_t - 1)

    with TiffWriter(OUTPUT_PATH, ome=True) as tif:
        for idx_t, t in enumerate(range(start_t, end_t + 1)):
            for idx_p, pos in enumerate(POSITION_INDICES):
                for channel in range(n_c):
                    for z in range(n_z):
                        plane = reader.ReadImagePlaneBuf(
                            CAPTURE_INDEX, pos, t, z, channel, True
                        )
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
    print(f"Wrote {OUTPUT_PATH}")



if __name__ == "__main__":
    export_capture()
