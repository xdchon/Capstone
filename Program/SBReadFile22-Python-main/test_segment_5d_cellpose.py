import argparse
import importlib.util
import json
import sys
import types
import uuid
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("segment_5d_cellpose.py")


def load_segment_module(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        mps=types.SimpleNamespace(empty_cache=lambda: None),
    )
    fake_models = types.SimpleNamespace(CellposeModel=None, torch=fake_torch)
    fake_cellpose = types.ModuleType("cellpose")
    fake_cellpose.models = fake_models
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models)

    module_name = f"segment_5d_cellpose_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeReader:
    def Open(self, _path):
        return True

    def GetNumCaptures(self):
        return 1

    def GetNumTimepoints(self, _capture):
        return 3

    def GetNumZPlanes(self, _capture):
        return 2

    def GetNumYRows(self, _capture):
        return 4

    def GetNumXColumns(self, _capture):
        return 5

    def GetNumChannels(self, _capture):
        return 1

    def GetVoxelSize(self, _capture):
        return (1.0, 1.0, 2.0)

    def ReadImagePlaneBuf(self, _capture, _position, timepoint, z_index, _channel, _as_2d):
        return np.full((4, 5), timepoint * 10 + z_index, dtype=np.uint16)


def make_args(tmp_path, mode="auto"):
    return argparse.Namespace(
        slide=str(tmp_path / "fake_slide.sldyz"),
        capture=0,
        position=0,
        channels=[0],
        timepoint=0,
        all_time=True,
        model="cpsam",
        gpu=False,
        mode=mode,
        stitch_threshold=0.4,
        diameter=0.0,
        anisotropy=None,
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        min_size=15,
        invert=False,
        no_normalize=False,
        norm3d=None,
        tile_norm_blocksize=0,
        output_root=str(tmp_path / "out"),
        save_npy=True,
        save_ome_tiff=False,
        save_per_z=False,
        save_flows=False,
    )


def test_segment_slidebook_streams_each_timepoint(monkeypatch, tmp_path):
    module = load_segment_module(monkeypatch)
    fake_model = StreamingModel()

    monkeypatch.setattr(module, "SBReadFile", FakeReader)
    monkeypatch.setattr(module.cp_models, "CellposeModel", lambda **_kwargs: fake_model)

    args = make_args(tmp_path, mode="auto")
    module.segment_slidebook(args)

    assert [call["time_marker"] for call in fake_model.calls] == [0, 10, 20]
    assert all(call["is_ndarray"] for call in fake_model.calls)
    assert all(call["do_3D"] for call in fake_model.calls)

    position_dir = tmp_path / "out" / "capture_000" / "position_000"
    for tp in range(3):
        mask_path = position_dir / f"timepoint_{tp:04d}" / f"masks_T{tp:04d}_ZYX.npy"
        masks = np.load(mask_path)
        assert masks.shape == (2, 4, 5)
        assert int(masks.max()) == tp + 1

    meta = json.loads((position_dir / "segmentation_metadata.json").read_text(encoding="utf-8"))
    assert meta["mode_used"] == "3d"
    assert meta["mode_used_by_timepoint"] == {"0": "3d", "1": "3d", "2": "3d"}


def test_segment_slidebook_falls_back_per_timepoint(monkeypatch, tmp_path):
    module = load_segment_module(monkeypatch)
    fake_model = FallbackOnSecondTimepointModel()

    monkeypatch.setattr(module, "SBReadFile", FakeReader)
    monkeypatch.setattr(module.cp_models, "CellposeModel", lambda **_kwargs: fake_model)

    args = make_args(tmp_path, mode="auto")
    module.segment_slidebook(args)

    assert [(call["time_marker"], call["do_3D"]) for call in fake_model.calls] == [
        (0, True),
        (10, True),
        (10, False),
        (20, True),
    ]
    fallback_call = fake_model.calls[2]
    assert fallback_call["channel_axis"] == -1
    assert fallback_call["z_axis"] == 0

    position_dir = tmp_path / "out" / "capture_000" / "position_000"
    meta = json.loads((position_dir / "segmentation_metadata.json").read_text(encoding="utf-8"))
    assert meta["mode_used"] == "mixed"
    assert meta["mode_used_by_timepoint"] == {
        "0": "3d",
        "1": "2d-stitch",
        "2": "3d",
    }

    for tp in range(3):
        assert (position_dir / f"timepoint_{tp:04d}" / f"masks_T{tp:04d}_ZYX.npy").exists()


class StreamingModel:
    def __init__(self):
        self.calls = []

    def eval(self, stack, **kwargs):
        stack = np.asarray(stack)
        time_marker = int(stack[0, 0, 0, 0])
        self.calls.append(
            {
                "is_ndarray": isinstance(stack, np.ndarray),
                "time_marker": time_marker,
                "do_3D": bool(kwargs["do_3D"]),
                "channel_axis": kwargs.get("channel_axis"),
                "z_axis": kwargs.get("z_axis"),
            }
        )
        label = time_marker // 10 + 1
        masks = np.full(stack.shape[:3], label, dtype=np.uint16)
        flows = {"label": label, "do_3D": bool(kwargs["do_3D"])}
        styles = {"label": label}
        return masks, flows, styles


class FallbackOnSecondTimepointModel(StreamingModel):
    def eval(self, stack, **kwargs):
        stack = np.asarray(stack)
        time_marker = int(stack[0, 0, 0, 0])
        self.calls.append(
            {
                "is_ndarray": isinstance(stack, np.ndarray),
                "time_marker": time_marker,
                "do_3D": bool(kwargs["do_3D"]),
                "channel_axis": kwargs.get("channel_axis"),
                "z_axis": kwargs.get("z_axis"),
            }
        )
        if time_marker == 10 and kwargs["do_3D"]:
            raise RuntimeError("synthetic 3D failure")
        label = time_marker // 10 + 1
        masks = np.full(stack.shape[:3], label, dtype=np.uint16)
        flows = {"label": label, "do_3D": bool(kwargs["do_3D"])}
        styles = {"label": label}
        return masks, flows, styles
