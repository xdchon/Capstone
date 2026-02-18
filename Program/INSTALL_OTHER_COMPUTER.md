# Install On Another Computer (Conda)

This repo has a local `cellpose` fork, so installation must use the local source code, not PyPI `cellpose`.

## Why your old env file failed

`Program/cellpose_env.yml` previously contained a machine-specific export:
- Windows build-string pins
- `python=3.14`
- local dev version pin (`cellpose==...dev...`)
- CUDA-specific torch wheel pin

That combination is not portable and often fails to solve/install on a second machine.

## Clean install steps

Run these commands from the repository root (the folder that contains `Program/`):

```bash
conda env remove -n cellpose-local -y
conda env create -f Program/cellpose_env.yml
conda activate cellpose-local
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ./Program/cellpose
python -m pip install pyqt6 pyqt6-sip pyqtgraph qtpy superqt
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
