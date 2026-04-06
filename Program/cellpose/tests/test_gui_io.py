import numpy as np
import tifffile

import cellpose.gui.io as gui_io


def test_read_image_for_gui_2d_rgb_ignores_stale_3d_request(tmp_path):
    path = tmp_path / "rgb_2d.tif"
    tifffile.imwrite(path, np.zeros((32, 24, 3), dtype=np.uint16))

    image, load_3d = gui_io._read_image_for_gui(
        str(path),
        requested_load_3d=True,
    )

    assert load_3d is False
    assert image.shape == (32, 24, 3)


def test_read_image_for_gui_keeps_z_stack_loading(tmp_path):
    path = tmp_path / "stack_zyx.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((4, 32, 24), dtype=np.uint16),
        ome=True,
        metadata={"axes": "ZYX"},
    )

    image, load_3d = gui_io._read_image_for_gui(str(path))

    assert load_3d is True
    assert image.shape == (4, 32, 24, 3)


def test_read_image_for_gui_keeps_time_stack_loading(tmp_path):
    path = tmp_path / "stack_tzyx.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((2, 4, 32, 24), dtype=np.uint16),
        ome=True,
        metadata={"axes": "TZYX"},
    )

    image, load_3d = gui_io._read_image_for_gui(str(path))

    assert load_3d is True
    assert image.shape == (2, 4, 32, 24, 3)
