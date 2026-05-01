# Local Cellpose SlideBook Workflow

## 1. Project Overview

This project uses a local fork of Cellpose to segment large microscopy datasets
from SlideBook.

The main goal is to support 5D image data:

- time
- z-stack
- channel
- y
- x

The standard Cellpose repository is mainly designed around images or stacks that
are already loaded into memory. This project adds tools for SlideBook files,
lazy loading, timepoint-by-timepoint segmentation, mask export, Fiji/ImageJ
analysis, and CSV analysis through a natural language interface.

## 2. Installation Instructions

Create the conda environment from the repository root:

```powershell
conda env create -f Program\cellpose_env.yml
conda activate cellpose-local
```

Install the local Cellpose fork:

```powershell
python -m pip install -e .\Program\cellpose
```

Install the Qt packages needed for the GUI:

```powershell
conda install -n cellpose-local -c conda-forge pyqt=6 qtpy pyqtgraph superqt qtbase -y
```

On Windows, run the Qt setup script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\Program\setup_windows_qt_env.ps1 -PersistForCondaEnv
```

Check the install:

```powershell
python Program\check_install.py
python -m cellpose --version
```

More detailed Windows setup notes are in:

```text
Program/INSTALL_OTHER_COMPUTER.md
```

## 3. Required Dependencies

Main dependencies:

- Python
- conda
- Cellpose/CPSAM from the local `Program/cellpose` fork
- PyTorch
- NumPy
- tifffile
- SBReadFile
- PyQt6
- qtpy
- pyqtgraph
- superqt
- pyzstd

Optional dependencies:

- Fiji/ImageJ for TrackMate and viewing OME-TIFF outputs
- OpenAI Python package for the natural language CSV interface
- Streamlit for the optional web interface

Install the optional NLI tools with:

```powershell
python -m pip install openai streamlit
```

## 4. Folder Structure

Important folders and files:

```text
Program/cellpose/
```

Local Cellpose fork. This contains the modified GUI, lazy 5D loading, SlideBook
support, timepoint mask saving, and OME-TIFF mask merging.

```text
Program/SBReadFile22-Python-main/
```

SlideBook reader and command-line scripts for exporting and segmenting
SlideBook data.

```text
Program/NLI_DB/
```

Natural language interface for CSV files.

```text
extract_random_tiff_slices.py
```

Random slice extraction script for normal 3D TIFF stacks.

```text
Program/check_install.py
```

Environment check script.

## 5. How To Load SlideBook Data

SlideBook files can be loaded directly by the local Cellpose fork.

Supported extensions:

- `.sldy`
- `.sldyz`

The local code uses `SBReadFile` through a lazy reader. This means the full
SlideBook file is not loaded into memory at once. Instead, the code loads the
requested timepoint and z-stack when needed.

To export one SlideBook timepoint as a TIFF stack:

```powershell
python Program\SBReadFile22-Python-main\export_Z_stack.py C:\path\to\file.sldyz --capture 0 --position 0 --channel 0 --timepoint 0
```

To export all timepoints from one channel:

```powershell
python Program\SBReadFile22-Python-main\export_Z_stack.py C:\path\to\file.sldyz --capture 0 --position 0 --channel 0 --all-time
```

To export a capture as an OME-TIFF:

```powershell
python Program\SBReadFile22-Python-main\export_capture_stack.py --input C:\path\to\file.sldyz --capture 0 --positions 0
```

## 6. How To Run Random Slice Extraction

For a normal 3D TIFF stack:

```powershell
python extract_random_tiff_slices.py --input C:\path\to\stack.tif --count 25 --output-dir C:\path\to\random_slices --seed 42
```

For a SlideBook `.sldy` file:

```powershell
python Program\cellpose\sample_random_sldy_slices_to_tiff.py --input C:\path\to\file.sldy --num-slices 25 --output-dir C:\path\to\random_sldy_slices --seed 42
```

The SlideBook sampler writes a `sample_manifest.csv` file that records which
timepoint and z-slice each exported image came from.

## 7. How To Run Segmentation In The GUI

Start the local Cellpose GUI:

```powershell
python -m cellpose
```

Basic workflow:

1. Open a TIFF, OME-TIFF, `.sldy`, or `.sldyz` file.
2. Choose the Cellpose/CPSAM model.
3. Set segmentation options such as diameter, thresholds, and normalization.
4. Use `downscale XY` if the stack is too large or slow.
5. For time-lapse data, tick `segment timepoint range` if multiple timepoints
   should be segmented.
6. Set the start and end timepoint.
7. Run segmentation.
8. Save the masks.

The modified GUI can keep masks separate for each timepoint and can reload the
correct masks when the visible timepoint changes.

## 8. How To Run Batch Segmentation

To segment SlideBook data from the command line:

```powershell
python Program\SBReadFile22-Python-main\segment_5d_cellpose.py --slide C:\path\to\file.sldyz --capture 0 --position 0 --all-time --mode auto --gpu --save-ome-tiff
```

Useful options:

- `--timepoint 0` segments one timepoint.
- `--all-time` segments every timepoint.
- `--channels 0` selects the channel.
- `--mode 3d` runs 3D segmentation only.
- `--mode 2d` runs per-slice segmentation.
- `--mode auto` tries 3D and falls back to 2D plus stitching if 3D fails.
- `--save-ome-tiff` saves labelled OME-TIFF masks.
- `--save-per-z` saves one TIFF per z-plane.
- `--save-flows` saves Cellpose flow outputs.

To segment exported timepoint TIFF stacks:

```powershell
python Program\SBReadFile22-Python-main\segment_timepoint_tiffs_cpsam.py --input C:\path\to\timepoint_tiffs --gpu --save-ome-tiff
```

## 9. How Masks Are Saved Per Time Point

For command-line SlideBook segmentation, masks are saved in folders like:

```text
OUTPUTMASKS/<slide_name>/capture_000/position_000/timepoint_0000/
```

Common files:

```text
masks_T0000_ZYX.npy
masks_T0000_ZYX.ome.tif
flows_T0000.npy
```

The run also writes:

```text
segmentation_metadata.json
```

In the GUI, time-lapse masks can also be saved as:

- per-timepoint `_seg.npy` files
- one aggregated `_seg.npy` file with a `per_time` section
- a folder of per-timepoint labelled TIFF stacks, usually ending in
  `_cp_masks_by_time`

The `per_time` structure is project-specific. It stores masks by timepoint so
the GUI does not mix up segmentation results across time.

## 10. How To Merge Masks Into OME-TIFF

If the GUI saved a folder of per-timepoint mask TIFFs, merge them into one
labelled OME-TIFF with:

```powershell
python Program\cellpose\merge_timepoint_mask_tiffs_to_ome.py --input C:\path\to\file_cp_masks_by_time --output C:\path\to\merged_masks.ome.tif
```

You can also use the GUI menu item:

```text
Build OME-TIFF from by-time mask folder (no stitching)
```

The merged mask file is written as `TZYX`: time, z, y, x.

## 11. How To Import Outputs Into Fiji/ImageJ

Open Fiji/ImageJ, then use:

```text
File > Open
```

Open the exported image OME-TIFF and the merged mask OME-TIFF.

Recommended checks:

- confirm that the time axis is correct
- confirm that the z-axis is correct
- confirm that mask labels line up with the original image
- use the same voxel/time calibration as the original dataset

For labelled masks, Fiji can display the file as an indexed label image. If
needed, adjust brightness/contrast or use a label lookup table.

## 12. How To Run TrackMate LAP Tracker

In Fiji/ImageJ:

1. Open the merged mask OME-TIFF or the image data to be tracked.
2. Start TrackMate:

```text
Plugins > Tracking > TrackMate
```

3. Choose a detector that matches the data type. For labelled masks, use a
   label/image-based detector if available in your Fiji installation.
4. Check that detected objects match the Cellpose masks.
5. Choose `LAP Tracker`.
6. Set linking distance, gap-closing distance, and frame gap based on the
   expected movement between timepoints.
7. Run tracking.
8. Inspect tracks manually before exporting.

TrackMate settings depend strongly on object speed, frame interval, and mask
quality, so the LAP parameters should be checked for each dataset.

## 13. How To Export CSV Files

After TrackMate tracking:

1. In TrackMate, go to the results/export section.
2. Export spot statistics, edge statistics, and track statistics as CSV files.
3. Save the CSV files with clear names that include the dataset and channel.

These CSV files can then be used by the natural language interface.

## 14. How To Upload CSVs Into The Natural Language Interface

Set an OpenAI API key:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Then start the desktop app:

```powershell
python Program\NLI_DB\nli_gui.py
```

Workflow:

1. Click `New Project` or `Open Project`.
2. Click `Add CSV Files`.
3. Select one or more TrackMate CSV files.
4. Select one CSV in `Dataset scope`.
5. Ask a question, show the schema, find correlations, or generate a report.

You can also ask about a CSV directly from the command line:

```powershell
python Program\NLI_DB\nli_cli.py C:\path\to\tracks.csv --question "summarise the tracking columns"
```

To generate a report:

```powershell
python Program\NLI_DB\nli_cli.py C:\path\to\tracks.csv --report --out C:\path\to\report_output
```

Do not commit or share `Program/NLI_DB/openai_api_key.txt`.

## 15. Known Limitations

- SlideBook loading depends on the bundled `SBReadFile` code.
- The SlideBook GUI loader currently uses channel 0 by default.
- Very large OME-TIFF exports can use a lot of disk space.
- Large 3D segmentation can run out of GPU or system memory.
- `--mode auto` can fall back to 2D plus stitching, but this may not behave the
  same as true 3D segmentation.
- TrackMate LAP settings must be tuned per dataset.
- The natural language interface analyses CSV content; it does not prove
  biological conclusions by itself.
- OpenAI-based NLI features require an API key and may send selected CSV data to
  OpenAI.

## 16. Troubleshooting

If Cellpose imports from the wrong location, reinstall the local fork:

```powershell
python -m pip install -e .\Program\cellpose
```

If the GUI has Qt errors on Windows, rerun:

```powershell
.\Program\setup_windows_qt_env.ps1 -PersistForCondaEnv
```

If segmentation runs out of memory:

- use `downscale XY`
- segment fewer timepoints at once
- use `--mode auto`
- use `--mode 2d`
- reduce output options such as flow saving

If Fiji does not show the axes correctly:

- check whether the TIFF is `ZYX` or `TZYX`
- reopen with Bio-Formats Importer
- confirm that the file is an OME-TIFF

If the NLI app cannot call OpenAI:

- check that `OPENAI_API_KEY` is set
- check internet access
- check that `openai` is installed in the active environment

### Adopted

- Cellpose/CPSAM as the base segmentation method.
- The Cellpose GUI and model evaluation pipeline.
- SBReadFile for reading SlideBook data.
- Fiji/ImageJ and TrackMate LAP Tracker for downstream object tracking.

### Modified

- Added `.sldy` and `.sldyz` loading to the local Cellpose fork.
- Added lazy 5D timepoint loading through `get_time_stack(t_index)`.
- Changed model evaluation so lazy time-lapse data can be segmented one
  timepoint at a time.
- Added GUI controls for timepoint range segmentation.
- Added GUI support for XY downscaling before segmentation.
- Added per-timepoint mask caching and reloading.
- Added saving of time-lapse masks as per-timepoint files and aggregated
  `per_time` data.
- Added OME-TIFF mask export and merging for time-lapse masks.

### Created

- `segment_5d_cellpose.py` for command-line SlideBook segmentation.
- `segment_timepoint_tiffs_cpsam.py` for batch segmentation of exported TIFF
  stacks.
- `seg_to_tif.py` for converting Cellpose `.npy` outputs to TIFF/OME-TIFF.
- `extract_random_tiff_slices.py` for random slice extraction from 3D TIFFs.
- `sample_random_sldy_slices_to_tiff.py` for random slice extraction from
  SlideBook files.
- `merge_timepoint_mask_tiffs_to_ome.py` for merging per-timepoint masks.
- `check_install.py` and `setup_windows_qt_env.ps1` for setup checks.
- `Program/NLI_DB` tools for uploading CSVs and asking natural-language
  questions about tracking data.
