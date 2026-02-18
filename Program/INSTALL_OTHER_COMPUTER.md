# Install On Another Computer (Conda)

This repo has a local `cellpose` fork, so installation must use the local source code, not PyPI `cellpose`.

## Why your old env file failed

`Program/cellpose_env.yml` previously contained a machine-specific export:
- Windows build-string pins
- `python=3.14`
- local dev version pin (`cellpose==...dev...`)
- CUDA-specific torch wheel pin

That combination is not portable and often fails to solve/install on a second machine.

You also hit two independent Windows-specific issues:
- Editable install error from `setuptools-scm` when `Program/cellpose` has no usable git metadata in the copied folder.
- Qt runtime plugin resolution error (`qwindows.dll`) where Qt imports succeed but GUI startup fails with:
  `Could not find the Qt platform plugin "windows" in ""`.

## Clean install steps

Run these commands from the repository root (the folder that contains `Program/`):

```bash
conda env remove -n cellpose-local -y
conda env create -f Program/cellpose_env.yml
conda activate cellpose-local
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ./Program/cellpose
conda install -n cellpose-local -c conda-forge pyqt=6 qtpy pyqtgraph superqt qtbase -y
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\Program\setup_windows_qt_env.ps1 -PersistForCondaEnv
```

## Optional GPU torch install

If you want GPU acceleration, replace torch/torchvision after the above install:

```bash
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Use the CUDA index URL that matches your NVIDIA driver/toolkit.

## Verify install

```bash
python Program/check_install.py
python -m cellpose --version
python Program/SBReadFile22-Python-main/segment_5d_cellpose.py --help
```

If `check_install.py` reports missing Qt Windows plugin, rerun:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\Program\setup_windows_qt_env.ps1 -PersistForCondaEnv
```

If it still fails and says `qwindows.dll` is missing, your environment has Qt Python packages but not the Windows platform plugin binaries. Repair by reinstalling one coherent Qt stack:

```powershell
conda install -n cellpose-local -c conda-forge pyqt=6 qtpy pyqtgraph superqt qtbase -y
# OR
python -m pip install --force-reinstall PyQt6 PyQt6-Qt6 qtpy pyqtgraph superqt
```

## Launch commands

Cellpose GUI:

```bash
python -m cellpose
```

SlideBook 5D segmentation CLI:

```bash
python Program/SBReadFile22-Python-main/segment_5d_cellpose.py --slide /path/to/file.sldyz --capture 0 --position 0 --all-time --mode auto --save-ome-tiff
```

## Common checks if something still fails

```bash
python -c "import cellpose; print(cellpose.__file__)"
```

Path should point into this repo (`Program/cellpose/...`), not a global site-packages copy.

If editable install fails with `setuptools-scm was unable to detect version`, your copy is missing git metadata for `Program/cellpose`. Use:

```powershell
$env:SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CELLPOSE="4.0.0"
python -m pip install -e ./Program/cellpose
Remove-Item Env:\SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CELLPOSE
```
