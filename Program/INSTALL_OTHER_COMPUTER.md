# Install On Another Windows Computer (Conda)

This project uses a local `cellpose` fork in `Program/cellpose`, so install from local source, not from PyPI.

## 0) Preflight on the source computer

Before you move to a second computer, verify what is actually tracked vs local-only:

- Tracked and expected in a normal clone:
  - `Program/cellpose`
  - `Program/SBReadFile22-Python-main`
  - `Program/cellpose_env.yml`
  - `Program/check_install.py`
  - `Program/setup_windows_qt_env.ps1`
- Local-only by `.gitignore` (not present after a normal clone):
  - `Program/NLI_DB`
  - `Program/*_env` (for example `Program/cellpose_env`, `Program/matlab_env`)
- Currently untracked local script:
  - `Program/SBReadFile22-Python-main/segment_timepoint_tiffs_cpsam.py`

If you need any local-only files on the new PC, either commit them or copy them manually.

Important security note:
- `Program/NLI_DB/openai_api_key.txt` should not be transferred as-is.
- Prefer setting `OPENAI_API_KEY` on the new machine instead.

Important size note:
- The `Images/` tree can be very large. Copy only the test data you need for validation.

## 1) Base install on target Windows PC (PowerShell)

Open **Anaconda Prompt** or **PowerShell with conda available**, then run from repository root:

```powershell
conda env remove -n cellpose-local -y
conda env create -f Program\cellpose_env.yml
conda activate cellpose-local

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .\Program\cellpose

conda install -n cellpose-local -c conda-forge pyqt=6 qtpy pyqtgraph superqt qtbase -y

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\Program\setup_windows_qt_env.ps1 -PersistForCondaEnv
```

## 2) Optional GPU torch install

If you want GPU acceleration, replace torch after base install:

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Use the CUDA index URL that matches your NVIDIA driver/toolkit.

## 3) Verify install

Run these checks in the activated `cellpose-local` env:

```powershell
python Program\check_install.py
python -m cellpose --version
python Program\SBReadFile22-Python-main\segment_5d_cellpose.py --help
python Program\SBReadFile22-Python-main\seg_to_tif.py --help
```

Expected outcomes:
- `check_install.py` prints `All required checks passed.`
- `cellpose` import path points into this repo (`Program/cellpose/...`).
- CLI `--help` commands print usage text without import errors.

## 4) Run commands

Cellpose GUI:

```powershell
python -m cellpose
```

SlideBook 5D segmentation:

```powershell
python Program\SBReadFile22-Python-main\segment_5d_cellpose.py --slide C:\path\to\file.sldyz --capture 0 --position 0 --all-time --mode auto --save-ome-tiff
```

## 5) Optional components

### NLI CSV OpenAI tools (`Program/NLI_DB`)

This folder is local-only by default (`.gitignore`), so copy it manually if needed.

Install extra packages:

```powershell
python -m pip install openai streamlit
```

Run:

```powershell
python Program\NLI_DB\nli_cli.py path\to\data.csv --question "summarise the tracking columns"
python Program\NLI_DB\nli_gui.py
streamlit run Program\NLI_DB\nli_streamlit.py
```

The recommended interface is now the desktop app:

```powershell
python Program\NLI_DB\nli_gui.py
```

In the app, create/open a CSV project folder, add one or more CSV files, select one CSV in `Dataset scope`, then ask analysis-mode questions or use `Generate Report`. Chat and report generation send the selected CSV directly to OpenAI.

### MATLAB bridge (`SldyToMATLAB.py`)

`Program/SBReadFile22-Python-main/SldyToMATLAB.py` additionally requires MATLAB Engine for Python in the active environment and a shared MATLAB session.

## 6) Common fixes

If Qt plugin errors appear (`qwindows.dll` not found):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\Program\setup_windows_qt_env.ps1 -PersistForCondaEnv
```

If still failing, reinstall one coherent Qt stack:

```powershell
conda install -n cellpose-local -c conda-forge pyqt=6 qtpy pyqtgraph superqt qtbase -y
# OR
python -m pip install --force-reinstall PyQt6 PyQt6-Qt6 qtpy pyqtgraph superqt
```

If editable install fails with `setuptools-scm was unable to detect version`:

```powershell
$env:SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CELLPOSE="4.0.0"
python -m pip install -e .\Program\cellpose
Remove-Item Env:\SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CELLPOSE
```
