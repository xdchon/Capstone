"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.
"""

import sys, os, pathlib, warnings, datetime, time, copy

from qtpy import QtGui, QtCore
from superqt import QRangeSlider, QCollapsible
from qtpy.QtWidgets import QScrollArea, QMainWindow, QApplication, QWidget, QScrollBar, \
    QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, \
        QLineEdit, QMessageBox, QGroupBox, QMenu, QAction
import pyqtgraph as pg

import numpy as np
from scipy.stats import mode
import cv2

from . import guiparts, menus, io
from .. import models, core, dynamics, metrics, version, train
from ..utils import download_url_to_file, masks_to_outlines, diameters
from ..io import get_image_files, imsave, imread
from ..transforms import resize_image, normalize99, normalize99_tile, smooth_sharpen_img
from ..models import normalize_default
from ..plot import disk

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False

Horizontal = QtCore.Qt.Orientation.Horizontal


class SegmentationWorker(QtCore.QObject):
    """Background worker to run Cellpose segmentation without blocking the GUI."""

    finished = QtCore.Signal(object)  # result dict
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int)

    def __init__(self, model, data, cfg, geom, parent=None):
        super().__init__(parent)
        self.model = model
        self.data = data
        self.cfg = cfg
        self.geom = geom

    def run(self):
        try:
            data = self.data
            cfg = self.cfg
            geom = self.geom

            tic = cfg.get("tic", time.time())
            do_3D = cfg["do_3D"]
            stitch_threshold = cfg["stitch_threshold"]
            anisotropy = cfg["anisotropy"]
            flow3D_smooth = cfg["flow3D_smooth"]
            min_size = cfg["min_size"]
            flow_threshold = cfg["flow_threshold"]
            cellprob_threshold = cfg["cellprob_threshold"]
            diameter = cfg["diameter"]
            niter = cfg["niter"]
            downscale = cfg["downscale"]
            normalize_params = cfg["normalize_params"]
            z_axis = cfg["z_axis"]

            self.progress.emit(25)
            masks, flows = self.model.eval(
                data,
                diameter=diameter,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                do_3D=do_3D,
                niter=niter,
                normalize=normalize_params,
                stitch_threshold=stitch_threshold,
                anisotropy=anisotropy,
                flow3D_smooth=flow3D_smooth,
                min_size=min_size,
                channel_axis=-1,
                progress=None,
                z_axis=z_axis,
            )[:2]

            self.progress.emit(60)

            masks, flows_display, recompute_masks = _process_segmentation_outputs(
                masks,
                flows,
                load_3D=geom["load_3D"],
                do_3D=do_3D,
                stitch_threshold=stitch_threshold,
                downscale=downscale,
                Ly=geom["Ly"],
                Lx=geom["Lx"],
                Ly0=geom["Ly0"],
                Lx0=geom["Lx0"],
                Lyr=geom.get("Lyr", geom["Ly"]),
                Lxr=geom.get("Lxr", geom["Lx"]),
                NZ=geom["NZ"],
                restore=geom.get("restore", None),
            )

            elapsed = time.time() - tic
            result = {
                "masks": masks,
                "flows": flows_display,
                "recompute_masks": recompute_masks,
                "do_3D": do_3D,
                "stitch_threshold": stitch_threshold,
                "downscale": downscale,
                "elapsed": elapsed,
                "time_index": cfg.get("time_index", None),
            }
            self.progress.emit(80)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


def _process_segmentation_outputs(
    masks,
    flows,
    *,
    load_3D,
    do_3D,
    stitch_threshold,
    downscale,
    Ly,
    Lx,
    Ly0,
    Lx0,
    Lyr,
    Lxr,
    NZ,
    restore,
):
    """Post-process raw Cellpose outputs into GUI-ready masks and flows."""

    flows_new = []
    flows_new.append(flows[0].copy())  # RGB flow
    flows_new.append(
        (np.clip(normalize99(flows[2].copy()), 0, 1) * 255).astype("uint8")
    )  # cellprob
    flows_new.append(flows[1].copy())  # XY flows
    flows_new.append(flows[2].copy())  # original cellprob

    if load_3D:
        if stitch_threshold == 0.0:
            flows_new.append((flows[1][0] / 10 * 127 + 127).astype("uint8"))
        else:
            flows_new.append(np.zeros(flows[1][0].shape, dtype="uint8"))

    flows_display = []

    # 2D / 2D+stitch: treat flows as per-slice XY overlays only
    if (not load_3D) or (not do_3D and stitch_threshold > 0.0):
        if restore and "upsample" in str(restore):
            out_Ly, out_Lx = Lyr, Lxr
        else:
            out_Ly, out_Lx = Ly, Lx

        if flows_new[0].shape[-3:-1] != (out_Ly, out_Lx):
            for j in range(len(flows_new)):
                try:
                    flows_display.append(
                        resize_image(
                            flows_new[j],
                            Ly=out_Ly,
                            Lx=out_Lx,
                            interpolation=cv2.INTER_NEAREST,
                        )
                    )
                except Exception as e:
                    print(f"GUI_WARNING: could not resize flow[{j}] for display ({e})")
                    flows_display.append(flows_new[j])
        else:
            flows_display = flows_new
    else:
        # true 3D segmentation: preserve Z and resize XY (and optionally Z) for display
        flows_display = []
        Lz, target_Ly, target_Lx = NZ, Ly, Lx
        Lz0, Ly0_flow, Lx0_flow = flows_new[0].shape[:3]
        print("GUI_INFO: resizing flows to original image size")
        for j in range(len(flows_new)):
            flow0 = flows_new[j]
            try:
                if Ly0_flow != target_Ly or Lx0_flow != target_Lx:
                    flow0 = resize_image(
                        flow0,
                        Ly=target_Ly,
                        Lx=target_Lx,
                        no_channels=flow0.ndim == 3,
                        interpolation=cv2.INTER_NEAREST,
                    )
                if Lz0 != Lz and flow0.ndim >= 3:
                    flow0 = np.swapaxes(
                        resize_image(
                            np.swapaxes(flow0, 0, 1),
                            Ly=Lz,
                            Lx=target_Lx,
                            no_channels=flow0.ndim == 3,
                            interpolation=cv2.INTER_NEAREST,
                        ),
                        0,
                        1,
                    )
            except Exception as e:
                print(f"GUI_WARNING: could not resize 3D flow[{j}] for display ({e})")
            flows_display.append(flow0)

    # add first axis for 2D images
    if NZ == 1:
        masks = masks[np.newaxis, ...]
        flows_display = [f[np.newaxis, ...] for f in flows_display]

    # if image was downscaled, resize masks back to original XY for display / saving
    if downscale is not None and downscale < 1.0:
        try:
            if masks.ndim == 3:
                masks = resize_image(
                    masks,
                    Ly=Ly0,
                    Lx=Lx0,
                    interpolation=cv2.INTER_NEAREST,
                    no_channels=True,
                )
            elif masks.ndim == 2:
                masks = resize_image(
                    masks[np.newaxis, ...],
                    Ly=Ly0,
                    Lx=Lx0,
                    interpolation=cv2.INTER_NEAREST,
                    no_channels=True,
                ).squeeze(0)
        except Exception as e:
            print(f"GUI_WARNING: could not resize masks back to original size ({e})")

    recompute_masks = not do_3D and not (stitch_threshold > 0.0)
    return masks, flows_display, recompute_masks

class Slider(QRangeSlider):

    def __init__(self, parent, name, color):
        super().__init__(Horizontal)
        self.setEnabled(False)
        self.valueChanged.connect(lambda: self.levelChanged(parent))
        self.name = name

        self.setStyleSheet(""" QSlider{
                             background-color: transparent;
                             }
        """)
        self.show()

    def levelChanged(self, parent):
        parent.level_change(self.name)


class QHLine(QFrame):

    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setLineWidth(8)


def make_bwr():
    # make a bwr colormap
    b = np.append(255 * np.ones(128), np.linspace(0, 255, 128)[::-1])[:, np.newaxis]
    r = np.append(np.linspace(0, 255, 128), 255 * np.ones(128))[:, np.newaxis]
    g = np.append(np.linspace(0, 255, 128),
                  np.linspace(0, 255, 128)[::-1])[:, np.newaxis]
    color = np.concatenate((r, g, b), axis=-1).astype(np.uint8)
    bwr = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return bwr


def make_spectral():
    # make spectral colormap
    r = np.array([
        0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80,
        84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 128, 128, 128, 128, 128,
        128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 120, 112, 104, 96, 88,
        80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 11, 15, 19, 23,
        27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103,
        107, 111, 115, 119, 123, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167,
        171, 175, 179, 183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231,
        235, 239, 243, 247, 251, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255
    ])
    g = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 5, 4, 4, 3, 3,
        2, 2, 1, 1, 0, 0, 0, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111,
        119, 127, 135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239,
        247, 255, 247, 239, 231, 223, 215, 207, 199, 191, 183, 175, 167, 159, 151, 143,
        135, 128, 129, 131, 132, 134, 135, 137, 139, 140, 142, 143, 145, 147, 148, 150,
        151, 153, 154, 156, 158, 159, 161, 162, 164, 166, 167, 169, 170, 172, 174, 175,
        177, 178, 180, 181, 183, 185, 186, 188, 189, 191, 193, 194, 196, 197, 199, 201,
        202, 204, 205, 207, 208, 210, 212, 213, 215, 216, 218, 220, 221, 223, 224, 226,
        228, 229, 231, 232, 234, 235, 237, 239, 240, 242, 243, 245, 247, 248, 250, 251,
        253, 255, 251, 247, 243, 239, 235, 231, 227, 223, 219, 215, 211, 207, 203, 199,
        195, 191, 187, 183, 179, 175, 171, 167, 163, 159, 155, 151, 147, 143, 139, 135,
        131, 127, 123, 119, 115, 111, 107, 103, 99, 95, 91, 87, 83, 79, 75, 71, 67, 63,
        59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11, 7, 3, 0, 8, 16, 24, 32, 41,
        49, 57, 65, 74, 82, 90, 98, 106, 115, 123, 131, 139, 148, 156, 164, 172, 180,
        189, 197, 205, 213, 222, 230, 238, 246, 254
    ])
    b = np.array([
        0, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127, 135, 143,
        151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239, 247, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 251, 247,
        243, 239, 235, 231, 227, 223, 219, 215, 211, 207, 203, 199, 195, 191, 187, 183,
        179, 175, 171, 167, 163, 159, 155, 151, 147, 143, 139, 135, 131, 128, 126, 124,
        122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90,
        88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50,
        48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10,
        8, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 16, 24, 32, 41, 49, 57, 65, 74,
        82, 90, 98, 106, 115, 123, 131, 139, 148, 156, 164, 172, 180, 189, 197, 205,
        213, 222, 230, 238, 246, 254
    ])
    color = (np.vstack((r, g, b)).T).astype(np.uint8)
    spectral = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return spectral


def make_cmap(cm=0):
    # make a single channel colormap
    r = np.arange(0, 256)
    color = np.zeros((256, 3))
    color[:, cm] = r
    color = color.astype(np.uint8)
    cmap = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return cmap


def _ensure_qt_plugin_paths():
    """Fill in Qt plugin paths when environment variables are missing/empty."""
    plugin_path = os.environ.get("QT_PLUGIN_PATH", "").strip()
    platform_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "").strip()
    if plugin_path and platform_path:
        return

    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        candidates.append(pathlib.Path(conda_prefix) / "Library" / "plugins")

    try:
        from qtpy.QtCore import QLibraryInfo

        qt_plugins = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
        if qt_plugins:
            candidates.append(pathlib.Path(qt_plugins))
    except Exception:
        pass

    chosen = None
    for candidate in candidates:
        if candidate and candidate.exists():
            chosen = candidate.resolve()
            break
    if chosen is None:
        return

    if not plugin_path:
        os.environ["QT_PLUGIN_PATH"] = str(chosen)

    platform_dir = chosen / "platforms"
    if not platform_path and platform_dir.exists():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_dir)


def run(image=None):
    from ..io import logger_setup
    logger, log_file = logger_setup()
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    _ensure_qt_plugin_paths()
    app = QApplication(sys.argv)
    icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
    guip_path = pathlib.Path.home().joinpath(".cellpose", "cellposeSAM_gui.png")
    if not icon_path.is_file():
        cp_dir = pathlib.Path.home().joinpath(".cellpose")
        cp_dir.mkdir(exist_ok=True)
        print("downloading logo")
        download_url_to_file(
            "https://www.cellpose.org/static/images/cellpose_transparent.png",
            icon_path, progress=True)
    if not guip_path.is_file():
        print("downloading help window image")
        download_url_to_file("https://www.cellpose.org/static/images/cellposeSAM_gui.png",
                             guip_path, progress=True)
    icon_path = str(icon_path.resolve())
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(64, 64))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    app.setStyle("Fusion")
    app.setPalette(guiparts.DarkPalette())
    MainW(image=image, logger=logger)
    ret = app.exec_()
    sys.exit(ret)


class MainW(QMainWindow):

    def __init__(self, image=None, logger=None):
        super(MainW, self).__init__()

        self.logger = logger
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(50, 50, 1200, 1000)
        self.setWindowTitle(f"cellpose v{version}")
        self.cp_path = os.path.dirname(os.path.realpath(__file__))
        app_icon = QtGui.QIcon()
        icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
        icon_path = str(icon_path.resolve())
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(64, 64))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)
        # rgb(150,255,150)
        self.setStyleSheet(guiparts.stylesheet())

        menus.mainmenu(self)
        menus.editmenu(self)
        menus.modelmenu(self)
        menus.helpmenu(self)

        self.stylePressed = """QPushButton {Text-align: center; 
                             background-color: rgb(150,50,150); 
                             border-color: white;
                             color:white;}
                            QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }"""
        self.styleUnpressed = """QPushButton {Text-align: center; 
                               background-color: rgb(50,50,50);
                                border-color: white;
                               color:white;}
                                QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }"""
        self.loaded = False

        # ---- MAIN WIDGET LAYOUT ---- #
        self.cwidget = QWidget(self)
        self.lmain = QGridLayout()
        self.cwidget.setLayout(self.lmain)
        self.setCentralWidget(self.cwidget)
        self.lmain.setVerticalSpacing(0)
        self.lmain.setContentsMargins(0, 0, 0, 10)

        self.imask = 0
        self.scrollarea = QScrollArea()
        self.scrollarea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollarea.setStyleSheet("""QScrollArea { border: none }""")
        self.scrollarea.setWidgetResizable(True)
        self.swidget = QWidget(self)
        self.scrollarea.setWidget(self.swidget)
        self.l0 = QGridLayout()
        self.swidget.setLayout(self.l0)
        b = self.make_buttons()
        self.lmain.addWidget(self.scrollarea, 0, 0, 39, 9)

        # ---- drawing area ---- #
        self.win = pg.GraphicsLayoutWidget()

        self.lmain.addWidget(self.win, 0, 9, 40, 30)
        # placeholder for bottom widgets
        self.loadingLabel = QLabel("")
        self.loadingLabel.setFont(QtGui.QFont("Arial", 9))
        self.loadingLabel.setStyleSheet("color: rgb(200,200,50);")
        self.loadingLabel.hide()
        self.lmain.addWidget(self.loadingLabel, 41, 9, 1, 10)

        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.win.scene().sigMouseMoved.connect(self.mouse_moved)
        self.make_viewbox()
        self.lmain.setColumnStretch(10, 1)
        bwrmap = make_bwr()
        self.bwr = bwrmap.getLookupTable(start=0.0, stop=255.0, alpha=False)
        self.cmap = []
        # spectral colormap
        self.cmap.append(make_spectral().getLookupTable(start=0.0, stop=255.0,
                                                        alpha=False))
        # single channel colormaps
        for i in range(3):
            self.cmap.append(
                make_cmap(i).getLookupTable(start=0.0, stop=255.0, alpha=False))

        if MATPLOTLIB:
            self.colormap = (plt.get_cmap("gist_ncar")(np.linspace(0.0, .9, 1000000)) *
                             255).astype(np.uint8)
            np.random.seed(42)  # make colors stable
            self.colormap = self.colormap[np.random.permutation(1000000)]
        else:
            np.random.seed(42)  # make colors stable
            self.colormap = ((np.random.rand(1000000, 3) * 0.8 + 0.1) * 255).astype(
                np.uint8)
        self.time_stack = None
        self.has_time = False
        self.NT = 1
        self.currentT = 0
        self.NZ = 1
        self.restore = None
        self.ratio = 1.
        self._segmentation_running = False
        self._segmentation_thread = None
        self._segmentation_worker = None
        self._segmentation_all_running = False
        self._seg_all_queue = []
        self._seg_all_total = 0
        self._seg_all_current_t = None
        self._seg_all_original_T = 0
        self._seg_all_custom = False
        self._seg_all_model_name = None
        self._seg_all_first = True
        self._seg_all_time_stitch_enabled = False
        self._seg_all_time_stitch_threshold = 0.25
        self._seg_all_time_prev_masks = None
        self._seg_all_time_next_label = 1
        self.seg_time_data = None
        self._current_seg_preserve_labels = False
        self.reset()

        # This needs to go after .reset() is called to get state fully set up:
        self.autobtn.checkStateChanged.connect(self.compute_saturation_if_checked)

        self.load_3D = False

        # if called with image, load it
        if image is not None:
            self.filename = image
            io._load_image(self, self.filename)

        # training settings
        d = datetime.datetime.now()
        self.training_params = {
            "model_index": 0,
            "initial_model": "cpsam",
            "learning_rate": 1e-5,
            "weight_decay": 0.1,
            "n_epochs": 100,
            "model_name": "cpsam" + d.strftime("_%Y%m%d_%H%M%S"),
        }

        self.stitch_threshold = 0.
        self.flow3D_smooth = 0.
        self.anisotropy = 1.
        self.min_size = 15

        self.setAcceptDrops(True)
        self.win.show()
        self.show()

    def help_window(self):
        HW = guiparts.HelpWindow(self)
        HW.show()

    def train_help_window(self):
        THW = guiparts.TrainHelpWindow(self)
        THW.show()

    def gui_window(self):
        EG = guiparts.ExampleGUI(self)
        EG.show()

    def make_buttons(self):
        self.boldfont = QtGui.QFont("Arial", 11, QtGui.QFont.Bold)
        self.boldmedfont = QtGui.QFont("Arial", 9, QtGui.QFont.Bold)
        self.medfont = QtGui.QFont("Arial", 9)
        self.smallfont = QtGui.QFont("Arial", 8)

        b = 0
        self.satBox = QGroupBox("Views")
        self.satBox.setFont(self.boldfont)
        self.satBoxG = QGridLayout()
        self.satBox.setLayout(self.satBoxG)
        self.l0.addWidget(self.satBox, b, 0, 1, 9)

        widget_row = 0
        self.view = 0  # 0=image, 1=flowsXY, 2=flowsZ, 3=cellprob
        self.color = 0  # 0=RGB, 1=gray, 2=R, 3=G, 4=B
        self.RGBDropDown = QComboBox()
        self.RGBDropDown.addItems(
            ["RGB", "red=R", "green=G", "blue=B", "gray", "spectral"])
        self.RGBDropDown.setFont(self.medfont)
        self.RGBDropDown.currentIndexChanged.connect(self.color_choose)
        self.satBoxG.addWidget(self.RGBDropDown, widget_row, 0, 1, 3)

        label = QLabel("<p>[&uarr; / &darr; or W/S]</p>")
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, widget_row, 3, 1, 3)
        label = QLabel("[R / G / B \n toggles color ]")
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, widget_row, 6, 1, 3)

        widget_row += 1
        self.ViewDropDown = QComboBox()
        self.ViewDropDown.addItems(["image", "gradXY", "cellprob", "restored"])
        self.ViewDropDown.setFont(self.medfont)
        self.ViewDropDown.model().item(3).setEnabled(False)
        self.ViewDropDown.currentIndexChanged.connect(self.update_plot)
        self.satBoxG.addWidget(self.ViewDropDown, widget_row, 0, 2, 3)

        label = QLabel("[pageup / pagedown]")
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, widget_row, 3, 1, 5)

        widget_row += 2
        label = QLabel("")
        label.setToolTip(
            "NOTE: manually changing the saturation bars does not affect normalization in segmentation"
        )
        self.satBoxG.addWidget(label, widget_row, 0, 1, 5)

        self.autobtn = QCheckBox("auto-adjust saturation")
        self.autobtn.setToolTip("sets scale-bars as normalized for segmentation")
        self.autobtn.setFont(self.medfont)
        self.autobtn.setChecked(True)
        self.satBoxG.addWidget(self.autobtn, widget_row, 1, 1, 8)

        widget_row += 1
        self.sliders = []
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [100, 100, 100]]
        colornames = ["red", "Chartreuse", "DodgerBlue"]
        names = ["red", "green", "blue"]
        for r in range(3):
            widget_row += 1
            if r == 0:
                label = QLabel('<font color="gray">gray/</font><br>red')
            else:
                label = QLabel(names[r] + ":")
            label.setStyleSheet(f"color: {colornames[r]}")
            label.setFont(self.boldmedfont)
            self.satBoxG.addWidget(label, widget_row, 0, 1, 2)
            self.sliders.append(Slider(self, names[r], colors[r]))
            self.sliders[-1].setMinimum(-.1)
            self.sliders[-1].setMaximum(255.1)
            self.sliders[-1].setValue([0, 255])
            self.sliders[-1].setToolTip(
                "NOTE: manually changing the saturation bars does not affect normalization in segmentation"
            )
            self.satBoxG.addWidget(self.sliders[-1], widget_row, 2, 1, 7)

        widget_row += 1
        # time controls (for 5D TCZYX)
        self.timeLabel = QLabel("time (t):")
        self.timeLabel.setFont(self.medfont)
        self.timeLabel.hide()
        self.timeScrollTop = QScrollBar(QtCore.Qt.Horizontal)
        self.timeScrollTop.setMinimum(0)
        self.timeScrollTop.setMaximum(0)
        self.timeScrollTop.setFixedWidth(140)
        self.timeScrollTop.valueChanged.connect(self.move_in_T)
        self.timeScrollTop.hide()
        self.timePos = QLineEdit("0")
        self.timePos.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.timePos.setFixedWidth(40)
        self.timePos.setFont(self.medfont)
        self.timePos.returnPressed.connect(self.update_tpos)
        self.timePos.hide()
        self.satBoxG.addWidget(self.timeLabel, widget_row, 0, 1, 2)
        self.satBoxG.addWidget(self.timeScrollTop, widget_row, 2, 1, 5)
        self.satBoxG.addWidget(self.timePos, widget_row, 7, 1, 2)

        b += 1
        self.drawBox = QGroupBox("Drawing")
        self.drawBox.setFont(self.boldfont)
        self.drawBoxG = QGridLayout()
        self.drawBox.setLayout(self.drawBoxG)
        self.l0.addWidget(self.drawBox, b, 0, 1, 9)
        self.autosave = True

        widget_row = 0
        self.brush_size = 3
        self.BrushChoose = QComboBox()
        self.BrushChoose.addItems(["1", "3", "5", "7", "9"])
        self.BrushChoose.currentIndexChanged.connect(self.brush_choose)
        self.BrushChoose.setFixedWidth(40)
        self.BrushChoose.setFont(self.medfont)
        self.drawBoxG.addWidget(self.BrushChoose, widget_row, 3, 1, 2)
        label = QLabel("brush size:")
        label.setFont(self.medfont)
        self.drawBoxG.addWidget(label, widget_row, 0, 1, 3)

        widget_row += 1
        # turn off masks
        self.layer_off = False
        self.masksOn = True
        self.MCheckBox = QCheckBox("MASKS ON [X]")
        self.MCheckBox.setFont(self.medfont)
        self.MCheckBox.setChecked(True)
        self.MCheckBox.toggled.connect(self.toggle_masks)
        self.drawBoxG.addWidget(self.MCheckBox, widget_row, 0, 1, 5)

        widget_row += 1
        # turn off outlines
        self.outlinesOn = False  # turn off by default
        self.OCheckBox = QCheckBox("outlines on [Z]")
        self.OCheckBox.setFont(self.medfont)
        self.drawBoxG.addWidget(self.OCheckBox, widget_row, 0, 1, 5)
        self.OCheckBox.setChecked(False)
        self.OCheckBox.toggled.connect(self.toggle_masks)

        widget_row += 1
        self.SCheckBox = QCheckBox("single stroke")
        self.SCheckBox.setFont(self.medfont)
        self.SCheckBox.setChecked(True)
        self.SCheckBox.toggled.connect(self.autosave_on)
        self.SCheckBox.setEnabled(True)
        self.drawBoxG.addWidget(self.SCheckBox, widget_row, 0, 1, 5)

        # buttons for deleting multiple cells
        self.deleteBox = QGroupBox("delete multiple ROIs")
        self.deleteBox.setStyleSheet("color: rgb(200, 200, 200)")
        self.deleteBox.setFont(self.medfont)
        self.deleteBoxG = QGridLayout()
        self.deleteBox.setLayout(self.deleteBoxG)
        self.drawBoxG.addWidget(self.deleteBox, 0, 5, 4, 4)
        self.MakeDeletionRegionButton = QPushButton("region-select")
        self.MakeDeletionRegionButton.clicked.connect(self.remove_region_cells)
        self.deleteBoxG.addWidget(self.MakeDeletionRegionButton, 0, 0, 1, 4)
        self.MakeDeletionRegionButton.setFont(self.smallfont)
        self.MakeDeletionRegionButton.setFixedWidth(70)
        self.DeleteMultipleROIButton = QPushButton("click-select")
        self.DeleteMultipleROIButton.clicked.connect(self.delete_multiple_cells)
        self.deleteBoxG.addWidget(self.DeleteMultipleROIButton, 1, 0, 1, 4)
        self.DeleteMultipleROIButton.setFont(self.smallfont)
        self.DeleteMultipleROIButton.setFixedWidth(70)
        self.DoneDeleteMultipleROIButton = QPushButton("done")
        self.DoneDeleteMultipleROIButton.clicked.connect(
            self.done_remove_multiple_cells)
        self.deleteBoxG.addWidget(self.DoneDeleteMultipleROIButton, 2, 0, 1, 2)
        self.DoneDeleteMultipleROIButton.setFont(self.smallfont)
        self.DoneDeleteMultipleROIButton.setFixedWidth(35)
        self.CancelDeleteMultipleROIButton = QPushButton("cancel")
        self.CancelDeleteMultipleROIButton.clicked.connect(self.cancel_remove_multiple)
        self.deleteBoxG.addWidget(self.CancelDeleteMultipleROIButton, 2, 2, 1, 2)
        self.CancelDeleteMultipleROIButton.setFont(self.smallfont)
        self.CancelDeleteMultipleROIButton.setFixedWidth(35)

        b += 1
        widget_row = 0
        self.segBox = QGroupBox("Segmentation")
        self.segBoxG = QGridLayout()
        self.segBox.setLayout(self.segBoxG)
        self.l0.addWidget(self.segBox, b, 0, 1, 9)
        self.segBox.setFont(self.boldfont)

        widget_row += 1

        # use GPU
        self.useGPU = QCheckBox("use GPU")
        self.useGPU.setToolTip(
            "if you have specially installed the <i>cuda</i> version of torch, then you can activate this"
        )
        self.useGPU.setFont(self.medfont)
        self.check_gpu()
        self.segBoxG.addWidget(self.useGPU, widget_row, 0, 1, 3)

        # compute segmentation with general models
        self.net_text = ["run CPSAM"]
        nett = ["cellpose super-generalist model"]

        self.StyleButtons = []
        jj = 4
        for j in range(len(self.net_text)):
            self.StyleButtons.append(
                guiparts.ModelButton(self, self.net_text[j], self.net_text[j]))
            w = 5
            self.segBoxG.addWidget(self.StyleButtons[-1], widget_row, jj, 1, w)
            jj += w
            self.StyleButtons[-1].setToolTip(nett[j])

        widget_row += 1
        self.ncells = guiparts.ObservableVariable(0)
        self.roi_count = QLabel()
        self.roi_count.setFont(self.boldfont)
        self.roi_count.setAlignment(QtCore.Qt.AlignLeft)
        self.ncells.valueChanged.connect(
            lambda n: self.roi_count.setText(f'{str(n)} ROIs')
        )

        self.segBoxG.addWidget(self.roi_count, widget_row, 0, 1, 4)

        self.progress = QProgressBar(self)
        self.segBoxG.addWidget(self.progress, widget_row, 4, 1, 5)

        widget_row += 1

        ############################### Segmentation settings ###############################
        self.additional_seg_settings_qcollapsible = QCollapsible("additional settings")
        self.additional_seg_settings_qcollapsible.setFont(self.medfont)
        self.additional_seg_settings_qcollapsible._toggle_btn.setFont(self.medfont)
        self.segmentation_settings = guiparts.SegmentationSettings(self.medfont)
        self.additional_seg_settings_qcollapsible.setContent(self.segmentation_settings)
        self.segBoxG.addWidget(self.additional_seg_settings_qcollapsible, widget_row, 0, 1, 9)

        # connect edits to image processing steps: 
        self.segmentation_settings.diameter_box.editingFinished.connect(self.update_scale)
        self.segmentation_settings.flow_threshold_box.returnPressed.connect(self.compute_cprob)
        self.segmentation_settings.cellprob_threshold_box.returnPressed.connect(self.compute_cprob)
        self.segmentation_settings.niter_box.returnPressed.connect(self.compute_cprob)

        # Needed to do this for the drop down to not be open on startup
        self.additional_seg_settings_qcollapsible._toggle_btn.setChecked(True)
        self.additional_seg_settings_qcollapsible._toggle_btn.setChecked(False)

        b += 1
        self.modelBox = QGroupBox("user-trained models")
        self.modelBoxG = QGridLayout()
        self.modelBox.setLayout(self.modelBoxG)
        self.l0.addWidget(self.modelBox, b, 0, 1, 9)
        self.modelBox.setFont(self.boldfont)
        # choose models
        self.ModelChooseC = QComboBox()
        self.ModelChooseC.setFont(self.medfont)
        current_index = 0
        self.ModelChooseC.addItems(["custom models"])
        if len(self.model_strings) > 0:
            self.ModelChooseC.addItems(self.model_strings)
        self.ModelChooseC.setFixedWidth(175)
        self.ModelChooseC.setCurrentIndex(current_index)
        tipstr = 'add or train your own models in the "Models" file menu and choose model here'
        self.ModelChooseC.setToolTip(tipstr)
        self.ModelChooseC.activated.connect(lambda: self.model_choose(custom=True))
        self.modelBoxG.addWidget(self.ModelChooseC, widget_row, 0, 1, 8)

        # compute segmentation w/ custom model
        self.ModelButtonC = QPushButton(u"run")
        self.ModelButtonC.setFont(self.medfont)
        self.ModelButtonC.setFixedWidth(35)
        # run custom model segmentation asynchronously for responsiveness
        self.ModelButtonC.clicked.connect(self.run_custom_segmentation)
        self.modelBoxG.addWidget(self.ModelButtonC, widget_row, 8, 1, 1)
        self.ModelButtonC.setEnabled(False)


        b += 1
        self.filterBox = QGroupBox("Image filtering")
        self.filterBox.setFont(self.boldfont)
        self.filterBox_grid_layout = QGridLayout()
        self.filterBox.setLayout(self.filterBox_grid_layout)
        self.l0.addWidget(self.filterBox, b, 0, 1, 9)

        widget_row = 0
        
        # Filtering
        self.FilterButtons = []
        nett = [
            "clear restore/filter",
            "filter image (settings below)",
        ]
        self.filter_text = ["none", 
                             "filter", 
                             ]
        self.restore = None
        self.ratio = 1.
        jj = 0
        w = 3
        for j in range(len(self.filter_text)):
            self.FilterButtons.append(
                guiparts.FilterButton(self, self.filter_text[j]))
            self.filterBox_grid_layout.addWidget(self.FilterButtons[-1], widget_row, jj, 1, w)
            self.FilterButtons[-1].setFixedWidth(75)
            self.FilterButtons[-1].setToolTip(nett[j])
            self.FilterButtons[-1].setFont(self.medfont)
            widget_row += 1 if j%2==1 else 0
            jj = 0 if j%2==1 else jj + w

        self.save_norm = QCheckBox("save restored/filtered image")
        self.save_norm.setFont(self.medfont)
        self.save_norm.setToolTip("save restored/filtered image in _seg.npy file")
        self.save_norm.setChecked(True)

        widget_row += 2

        self.filtBox = QCollapsible("custom filter settings")
        self.filtBox._toggle_btn.setFont(self.medfont)
        self.filtBoxG = QGridLayout()
        _content = QWidget()
        _content.setLayout(self.filtBoxG)
        _content.setMaximumHeight(0)
        _content.setMinimumHeight(0)
        self.filtBox.setContent(_content)
        self.filterBox_grid_layout.addWidget(self.filtBox, widget_row, 0, 1, 9)

        self.filt_vals = [0., 0., 0., 0.]
        self.filt_edits = []
        labels = [
            "sharpen\nradius", "smooth\nradius", "tile_norm\nblocksize",
            "tile_norm\nsmooth3D"
        ]
        tooltips = [
            "set size of surround-subtraction filter for sharpening image",
            "set size of gaussian filter for smoothing image",
            "set size of tiles to use to normalize image",
            "set amount of smoothing of normalization values across planes"
        ]

        for p in range(4):
            label = QLabel(f"{labels[p]}:")
            label.setToolTip(tooltips[p])
            label.setFont(self.medfont)
            self.filtBoxG.addWidget(label, widget_row + p // 2, 4 * (p % 2), 1, 2)
            self.filt_edits.append(QLineEdit())
            self.filt_edits[p].setText(str(self.filt_vals[p]))
            self.filt_edits[p].setFixedWidth(40)
            self.filt_edits[p].setFont(self.medfont)
            self.filtBoxG.addWidget(self.filt_edits[p], widget_row + p // 2, 4 * (p % 2) + 2, 1,
                                    2)
            self.filt_edits[p].setToolTip(tooltips[p])

        widget_row += 3
        self.norm3D_cb = QCheckBox("norm3D")
        self.norm3D_cb.setFont(self.medfont)
        self.norm3D_cb.setChecked(True)
        self.norm3D_cb.setToolTip("run same normalization across planes")
        self.filtBoxG.addWidget(self.norm3D_cb, widget_row, 0, 1, 3)


        return b

    def configure_timebar(self):
        show_time = self.has_time and self.NT > 1
        for w in [self.timeLabel, getattr(self, "timeScrollTop", None), getattr(self, "timeScrollBottom", None), self.timePos]:
            if w is None:
                continue
            w.setVisible(show_time)
        if show_time:
            if hasattr(self, "timeScrollTop") and self.timeScrollTop is not None:
                self.timeScrollTop.setMaximum(self.NT - 1)
                self.timeScrollTop.blockSignals(True)
                self.timeScrollTop.setValue(self.currentT)
                self.timeScrollTop.blockSignals(False)
            if hasattr(self, "timeScrollBottom") and self.timeScrollBottom is not None:
                self.timeScrollBottom.setMaximum(self.NT - 1)
                self.timeScrollBottom.blockSignals(True)
                self.timeScrollBottom.setValue(self.currentT)
                self.timeScrollBottom.blockSignals(False)
            self.timePos.setText(str(self.currentT))

    def set_time_index(self, t):
        has_lazy = hasattr(self, "lazy_data") and self.lazy_data is not None
        if not (self.has_time and (self.time_stack is not None or has_lazy)):
            return
        last_z = getattr(self, "currentZ", 0)
        t_clamped = max(0, min(self.NT - 1, int(t)))
        if t_clamped == self.currentT and self.loaded:
            self.timePos.setText(str(t_clamped))
            return
        self.currentT = t_clamped
        self.timePos.setText(str(self.currentT))
        if has_lazy:
            self.show_loading(f"Loading T={self.currentT} ...")
        if hasattr(self, "timeScrollTop") and self.timeScrollTop is not None:
            self.timeScrollTop.blockSignals(True)
            self.timeScrollTop.setValue(self.currentT)
            self.timeScrollTop.blockSignals(False)
        if hasattr(self, "timeScrollBottom") and self.timeScrollBottom is not None:
            self.timeScrollBottom.blockSignals(True)
            self.timeScrollBottom.setValue(self.currentT)
            self.timeScrollBottom.blockSignals(False)
        # reload current time slice into the viewer
        target = self.lazy_data if has_lazy else self.time_stack[self.currentT]
        # preserve current view range (so time swaps don't reset zoom)
        view_range = None
        if hasattr(self, "p0"):
            try:
                view_range = self.p0.viewRange()
            except Exception:
                view_range = None
        try:
            io._initialize_images(self, target, load_3D=self.load_3D, time_stack=self.time_stack, currentT=self.currentT, use_lazy=has_lazy)
            self.loaded = True
            self.enable_buttons()
            self.update_plot()
            # automatically load matching segmentation for this timepoint, if present
            seg_loaded = False
            try:
                base, _ = os.path.splitext(self.filename)
                t_index = int(getattr(self, "currentT", 0))
                # prefer in-memory aggregated time-lapse seg if available
                per_time = getattr(self, "seg_time_data", None)
                if isinstance(per_time, dict):
                    dat_t = per_time.get(t_index)
                    if dat_t is None and str(t_index) in per_time:
                        dat_t = per_time[str(t_index)]
                    # as a fallback, search by stored 'time_index' field
                    if (dat_t is None or not isinstance(dat_t, dict)) and len(per_time) > 0:
                        for _k, _v in per_time.items():
                            try:
                                if isinstance(_v, dict) and int(_v.get("time_index", -1)) == t_index:
                                    dat_t = _v
                                    break
                            except Exception:
                                continue
                    if isinstance(dat_t, dict):
                        masks = dat_t.get("masks", None)
                        outlines = dat_t.get("outlines", None)
                        colors = dat_t.get("colors", None)
                        preserve_labels = bool(dat_t.get("preserve_labels", False))
                        if masks is not None:
                            io._masks_to_gui(
                                self,
                                masks,
                                outlines=outlines,
                                colors=colors,
                                preserve_labels=preserve_labels,
                            )
                            seg_loaded = True
                # if not cached, try loading aggregated seg file from disk once
                if not seg_loaded:
                    seg_all_path = base + "_seg.npy"
                    if os.path.isfile(seg_all_path):
                        try:
                            all_dat = np.load(seg_all_path, allow_pickle=True).item()
                            per_time = all_dat.get("per_time", {})
                            if isinstance(per_time, dict):
                                # Merge disk cache with in-memory cache instead of replacing it.
                                # This preserves freshly computed, not-yet-saved timepoints.
                                merged_per_time = {}
                                if isinstance(per_time, dict):
                                    merged_per_time.update(per_time)
                                mem_per_time = getattr(self, "seg_time_data", None)
                                if isinstance(mem_per_time, dict):
                                    merged_per_time.update(mem_per_time)
                                self.seg_time_data = merged_per_time

                                dat_t = merged_per_time.get(t_index)
                                if dat_t is None and str(t_index) in merged_per_time:
                                    dat_t = merged_per_time[str(t_index)]
                                if (dat_t is None or not isinstance(dat_t, dict)) and len(merged_per_time) > 0:
                                    for _k, _v in merged_per_time.items():
                                        try:
                                            if isinstance(_v, dict) and int(_v.get("time_index", -1)) == t_index:
                                                dat_t = _v
                                                break
                                        except Exception:
                                            continue
                                if isinstance(dat_t, dict):
                                    masks = dat_t.get("masks", None)
                                    outlines = dat_t.get("outlines", None)
                                    colors = dat_t.get("colors", None)
                                    preserve_labels = bool(
                                        dat_t.get(
                                            "preserve_labels",
                                            all_dat.get("stitch_over_time", False),
                                        )
                                    )
                                    if masks is not None:
                                        io._masks_to_gui(
                                            self,
                                            masks,
                                            outlines=outlines,
                                            colors=colors,
                                            preserve_labels=preserve_labels,
                                        )
                                        seg_loaded = True

                            # Fallback: OME-style TZYX arrays at top-level (no per_time mapping)
                            if (not seg_loaded and isinstance(all_dat, dict)
                                    and isinstance(all_dat.get("masks", None), np.ndarray)
                                    and isinstance(all_dat.get("axes", None), str)):
                                axes = all_dat.get("axes").upper()
                                masks_all = all_dat.get("masks")
                                outlines_all = all_dat.get("outlines", None)
                                if "T" in axes and "Z" in axes and "Y" in axes and "X" in axes:
                                    try:
                                        t_axis = axes.index("T")
                                        z_axis = axes.index("Z")
                                        y_axis = axes.index("Y")
                                        x_axis = axes.index("X")
                                        order = [t_axis, z_axis, y_axis, x_axis]
                                        arr_tzyx = np.transpose(masks_all, order)
                                        if 0 <= t_index < arr_tzyx.shape[0]:
                                            masks_t = arr_tzyx[t_index]
                                            outlines_t = None
                                            preserve_labels = bool(
                                                all_dat.get("stitch_over_time", False)
                                            )
                                            if isinstance(outlines_all, np.ndarray) and outlines_all.shape == masks_all.shape:
                                                out_arr = np.transpose(outlines_all, order)
                                                outlines_t = out_arr[t_index]
                                            io._masks_to_gui(
                                                self,
                                                masks_t,
                                                outlines=outlines_t,
                                                colors=None,
                                                preserve_labels=preserve_labels,
                                            )
                                            seg_loaded = True
                                    except Exception as ee:
                                        print(f"ERROR: could not interpret OME-style masks in {seg_all_path}: {ee}")
                        except Exception as e:
                            if isinstance(e, EOFError):
                                print(
                                    f"ERROR: seg_all file appears truncated or corrupted (EOF) {seg_all_path}: {e}"
                                )
                            else:
                                print(f"ERROR: could not read seg_all file {seg_all_path}: {e}")
                # fallback: per-timepoint seg file
                if not seg_loaded:
                    seg_path = base + f"_T{t_index:04d}_seg.npy"
                    if os.path.isfile(seg_path):
                        dat = np.load(seg_path, allow_pickle=True).item()
                        masks = dat.get("masks", None)
                        outlines = dat.get("outlines", None)
                        colors = dat.get("colors", None)
                        preserve_labels = bool(dat.get("preserve_labels", False))
                        if masks is not None:
                            io._masks_to_gui(
                                self,
                                masks,
                                outlines=outlines,
                                colors=colors,
                                preserve_labels=preserve_labels,
                            )
                            seg_loaded = True
                # final fallback: by-timepoint TIFF folder (e.g. *_cp_masks_by_time)
                if not seg_loaded:
                    seg_loaded = io._load_timepoint_mask_from_by_time(
                        self, base, t_index
                    )
            except Exception as e:
                print(f"ERROR: could not auto-load seg for T={self.currentT}: {e}")
            if not seg_loaded and hasattr(self, "cellpix") and hasattr(self, "NZ") and self.NZ is not None:
                # clear masks when no segmentation is available for this T
                try:
                    self.ncells.set(0)
                    self.cellpix = np.zeros((self.NZ, self.Ly0, self.Lx0), np.uint16)
                    self.outpix = np.zeros_like(self.cellpix)
                    self.masksOn = False
                    self.update_layer()
                except Exception:
                    pass
            if has_lazy:
                desired_z = max(0, min(self.NZ - 1, int(last_z)))
                if self.currentZ != desired_z:
                    self.currentZ = desired_z
                    if hasattr(self, "scroll"):
                        self.scroll.blockSignals(True)
                        self.scroll.setValue(self.currentZ)
                        self.scroll.blockSignals(False)
                    try:
                        plane = self.lazy_data.get_plane(self.currentT, self.currentZ)
                        plane = plane.astype(np.float32)
                        pmin, pmax = plane.min(), plane.max()
                        if pmax > pmin + 1e-3:
                            plane = (plane - pmin) / (pmax - pmin) * 255.
                        if hasattr(self, "time_cache") and self.time_cache is not None:
                            self.time_cache = {(self.currentT, self.currentZ): plane}
                        self.stack = plane[np.newaxis, ...]
                        self.update_plot()
                    except Exception as e:
                        print(f"ERROR fetching plane T{self.currentT} Z{self.currentZ}: {e}")
            if view_range is not None:
                try:
                    self.p0.blockSignals(True)
                    self.p0.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)
                finally:
                    self.p0.blockSignals(False)
        finally:
            if has_lazy:
                self.hide_loading()

    def move_in_T(self, value):
        if not (self.has_time and (self.time_stack is not None or hasattr(self, "lazy_data") and self.lazy_data is not None)):
            return
        self.set_time_index(value)

    def update_tpos(self):
        if not (self.has_time and (self.time_stack is not None or hasattr(self, "lazy_data") and self.lazy_data is not None)):
            return
        try:
            t = int(self.timePos.text())
        except Exception:
            t = self.currentT
        self.set_time_index(t)

    def level_change(self, r):
        r = ["red", "green", "blue"].index(r)
        if self.loaded:
            sval = self.sliders[r].value()
            self.saturation[r][self.currentZ] = sval
            if not self.autobtn.isChecked():
                for r in range(3):
                    for i in range(len(self.saturation[r])):
                        self.saturation[r][i] = self.saturation[r][self.currentZ]
            self.update_plot()

    def keyPressEvent(self, event):
        if self.loaded:
            if not (event.modifiers() &
                    (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier |
                     QtCore.Qt.AltModifier) or self.in_stroke):
                updated = False
                if len(self.current_point_set) > 0:
                    if event.key() == QtCore.Qt.Key_Return:
                        self.add_set()
                else:
                    nviews = self.ViewDropDown.count() - 1
                    nviews += int(
                        self.ViewDropDown.model().item(self.ViewDropDown.count() -
                                                       1).isEnabled())
                    if event.key() == QtCore.Qt.Key_X:
                        self.MCheckBox.toggle()
                    if event.key() == QtCore.Qt.Key_Z:
                        self.OCheckBox.toggle()
                    if event.key() == QtCore.Qt.Key_Left or event.key(
                    ) == QtCore.Qt.Key_A:
                        self.get_prev_image()
                    elif event.key() == QtCore.Qt.Key_Right or event.key(
                    ) == QtCore.Qt.Key_D:
                        self.get_next_image()
                    elif event.key() == QtCore.Qt.Key_PageDown:
                        self.view = (self.view + 1) % (nviews)
                        self.ViewDropDown.setCurrentIndex(self.view)
                    elif event.key() == QtCore.Qt.Key_PageUp:
                        self.view = (self.view - 1) % (nviews)
                        self.ViewDropDown.setCurrentIndex(self.view)

                # can change background or stroke size if cell not finished
                if event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_W:
                    self.color = (self.color - 1) % (6)
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_Down or event.key(
                ) == QtCore.Qt.Key_S:
                    self.color = (self.color + 1) % (6)
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_R:
                    if self.color != 1:
                        self.color = 1
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_G:
                    if self.color != 2:
                        self.color = 2
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_B:
                    if self.color != 3:
                        self.color = 3
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif (event.key() == QtCore.Qt.Key_Comma or
                      event.key() == QtCore.Qt.Key_Period):
                    count = self.BrushChoose.count()
                    gci = self.BrushChoose.currentIndex()
                    if event.key() == QtCore.Qt.Key_Comma:
                        gci = max(0, gci - 1)
                    else:
                        gci = min(count - 1, gci + 1)
                    self.BrushChoose.setCurrentIndex(gci)
                    self.brush_choose()
                if not updated:
                    self.update_plot()
        if event.key() == QtCore.Qt.Key_Minus or event.key() == QtCore.Qt.Key_Equal:
            self.p0.keyPressEvent(event)

    def autosave_on(self):
        if self.SCheckBox.isChecked():
            self.autosave = True
        else:
            self.autosave = False

    def check_gpu(self, torch=True):
        # also decide whether or not to use torch
        self.useGPU.setChecked(False)
        self.useGPU.setEnabled(False)
        if core.use_gpu(use_torch=True):
            self.useGPU.setEnabled(True)
            self.useGPU.setChecked(True)
        else:
            self.useGPU.setStyleSheet("color: rgb(80,80,80);")


    def model_choose(self, custom=False):
        index = self.ModelChooseC.currentIndex(
        ) if custom else self.ModelChooseB.currentIndex()
        if index > 0:
            if custom:
                model_name = self.ModelChooseC.currentText()
            else:
                model_name = self.net_names[index - 1]
            print(f"GUI_INFO: selected model {model_name}, loading now")
            self.initialize_model(model_name=model_name, custom=custom)

    def toggle_scale(self):
        if self.scale_on:
            self.p0.removeItem(self.scale)
            self.scale_on = False
        else:
            self.p0.addItem(self.scale)
            self.scale_on = True

    def enable_buttons(self):
        if len(self.model_strings) > 0:
            self.ModelButtonC.setEnabled(True)
        for i in range(len(self.StyleButtons)):
            self.StyleButtons[i].setEnabled(True)

        for i in range(len(self.FilterButtons)):
            self.FilterButtons[i].setEnabled(True)
        if self.load_3D:
            self.FilterButtons[-2].setEnabled(False)

        self.newmodel.setEnabled(True)
        self.loadMasks.setEnabled(True)

        for n in range(self.nchan):
            self.sliders[n].setEnabled(True)
        for n in range(self.nchan, 3):
            self.sliders[n].setEnabled(True)

        self.toggle_mask_ops()

        self.update_plot()
        self.setWindowTitle(self.filename)

    def disable_buttons_removeROIs(self):
        if len(self.model_strings) > 0:
            self.ModelButtonC.setEnabled(False)
        for i in range(len(self.StyleButtons)):
            self.StyleButtons[i].setEnabled(False)
        self.newmodel.setEnabled(False)
        self.loadMasks.setEnabled(False)
        self.saveSet.setEnabled(False)
        self.savePNG.setEnabled(False)
        self.saveFlows.setEnabled(False)
        self.saveOutlines.setEnabled(False)
        self.saveROIs.setEnabled(False)

        self.MakeDeletionRegionButton.setEnabled(False)
        self.DeleteMultipleROIButton.setEnabled(False)
        self.DoneDeleteMultipleROIButton.setEnabled(True)
        self.CancelDeleteMultipleROIButton.setEnabled(True)

    def toggle_mask_ops(self):
        self.update_layer()
        self.toggle_saving()
        self.toggle_removals()

    def toggle_saving(self):
        if self.ncells > 0:
            self.saveSet.setEnabled(True)
            self.savePNG.setEnabled(True)
            self.saveFlows.setEnabled(True)
            self.saveOutlines.setEnabled(True)
            self.saveROIs.setEnabled(True)
        else:
            self.saveSet.setEnabled(False)
            self.savePNG.setEnabled(False)
            self.saveFlows.setEnabled(False)
            self.saveOutlines.setEnabled(False)
            self.saveROIs.setEnabled(False)

    def toggle_removals(self):
        if self.ncells > 0:
            self.ClearButton.setEnabled(True)
            self.remcell.setEnabled(True)
            self.undo.setEnabled(True)
            self.MakeDeletionRegionButton.setEnabled(True)
            self.DeleteMultipleROIButton.setEnabled(True)
            self.DoneDeleteMultipleROIButton.setEnabled(False)
            self.CancelDeleteMultipleROIButton.setEnabled(False)
        else:
            self.ClearButton.setEnabled(False)
            self.remcell.setEnabled(False)
            self.undo.setEnabled(False)
            self.MakeDeletionRegionButton.setEnabled(False)
            self.DeleteMultipleROIButton.setEnabled(False)
            self.DoneDeleteMultipleROIButton.setEnabled(False)
            self.CancelDeleteMultipleROIButton.setEnabled(False)

    def remove_action(self):
        if self.selected > 0:
            self.remove_cell(self.selected)

    def undo_action(self):
        if (len(self.strokes) > 0 and self.strokes[-1][0][0] == self.currentZ):
            self.remove_stroke()
        else:
            # remove previous cell
            if self.ncells > 0:
                self.remove_cell(self.ncells.get())

    def undo_remove_action(self):
        self.undo_remove_cell()

    def get_files(self):
        folder = os.path.dirname(self.filename)
        mask_filter = "_masks"
        images = get_image_files(folder, mask_filter)
        fnames = [os.path.split(images[k])[-1] for k in range(len(images))]
        f0 = os.path.split(self.filename)[-1]
        idx = np.nonzero(np.array(fnames) == f0)[0][0]
        return images, idx

    def get_prev_image(self):
        images, idx = self.get_files()
        idx = (idx - 1) % len(images)
        io._load_image(self, filename=images[idx])

    def get_next_image(self, load_seg=True):
        images, idx = self.get_files()
        idx = (idx + 1) % len(images)
        io._load_image(self, filename=images[idx], load_seg=load_seg)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if os.path.splitext(files[0])[-1] == ".npy":
            io._load_seg(self, filename=files[0], load_3D=self.load_3D)
        else:
            io._load_image(self, filename=files[0], load_seg=True, load_3D=self.load_3D)

    def toggle_masks(self):
        if self.MCheckBox.isChecked():
            self.masksOn = True
        else:
            self.masksOn = False
        if self.OCheckBox.isChecked():
            self.outlinesOn = True
        else:
            self.outlinesOn = False
        if not self.masksOn and not self.outlinesOn:
            self.p0.removeItem(self.layer)
            self.layer_off = True
        else:
            if self.layer_off:
                self.p0.addItem(self.layer)
            self.draw_layer()
            self.update_layer()
        if self.loaded:
            self.update_plot()
            self.update_layer()

    def make_viewbox(self):
        self.p0 = guiparts.ViewBoxNoRightDrag(parent=self, lockAspect=True,
                                              name="plot1", border=[100, 100,
                                                                    100], invertY=True)
        self.p0.setCursor(QtCore.Qt.CrossCursor)
        self.brush_size = 3
        self.win.addItem(self.p0, 0, 0, rowspan=1, colspan=1)
        self.p0.setMenuEnabled(False)
        self.p0.setMouseEnabled(x=True, y=True)
        self.img = pg.ImageItem(viewbox=self.p0, parent=self)
        self.img.autoDownsample = False
        self.layer = guiparts.ImageDraw(viewbox=self.p0, parent=self)
        self.layer.setLevels([0, 255])
        self.scale = pg.ImageItem(viewbox=self.p0, parent=self)
        self.scale.setLevels([0, 255])
        self.p0.scene().contextMenuItem = self.p0
        self.Ly, self.Lx = 512, 512
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)

    def reset(self):
        # ---- start sets of points ---- #
        self.selected = 0
        self.nchan = 3
        self.loaded = False
        self.channel = [0, 1]
        self.current_point_set = []
        self.in_stroke = False
        self.strokes = []
        self.stroke_appended = True
        self.resize = False
        self.ncells.reset()
        self.zdraw = []
        self.removed_cell = []
        self.cellcolors = np.array([255, 255, 255])[np.newaxis, :]

        # -- zero out image stack -- #
        self.opacity = 128  # how opaque masks should be
        self.outcolor = [200, 200, 255, 200]
        self.NZ, self.Ly, self.Lx = 1, 256, 256
        self.saturation = self.saturation if hasattr(self, 'saturation') else []

        # only adjust the saturation if auto-adjust is on: 
        if self.autobtn.isChecked():
            for r in range(3):
                self.saturation.append([[0, 255] for n in range(self.NZ)])
                self.sliders[r].setValue([0, 255])
                self.sliders[r].setEnabled(False)
                self.sliders[r].show()
        self.currentZ = 0
        self.flows = [[], [], [], [], [[]]]
        # masks matrix
        # image matrix with a scale disk
        self.stack = np.zeros((1, self.Ly, self.Lx, 3))
        self.Lyr, self.Lxr = self.Ly, self.Lx
        self.Ly0, self.Lx0 = self.Ly, self.Lx
        self.radii = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
        self.layerz = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
        self.cellpix = np.zeros((1, self.Ly, self.Lx), np.uint16)
        self.outpix = np.zeros((1, self.Ly, self.Lx), np.uint16)
        self.ismanual = np.zeros(0, "bool")

        # -- set menus to default -- #
        self.color = 0
        self.RGBDropDown.setCurrentIndex(self.color)
        self.view = 0
        self.ViewDropDown.setCurrentIndex(0)
        self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(False)
        self.delete_restore()

        self.clear_all()

        self.filename = []
        self.loaded = False
        self.recompute_masks = False
        self.seg_time_data = None
        self._current_seg_time_index = None
        self._current_seg_preserve_labels = False
        self._seg_all_total = 0
        self._seg_all_current_t = None
        self._reset_time_stitch_state()
        self._time_mask_base = None
        self._time_mask_folder = None
        self._time_mask_folder_mtime = None
        self._time_mask_file_map = None
        self.available_mask_timepoints = []

        self.deleting_multiple = False
        self.removing_cells_list = []
        self.removing_region = False
        self.remove_roi_obj = None

    def show_loading(self, text="Loading..."):
        if hasattr(self, "loadingLabel"):
            self.loadingLabel.setText(text)
            self.loadingLabel.show()
            self.loadingLabel.repaint()

    def hide_loading(self):
        if hasattr(self, "loadingLabel"):
            self.loadingLabel.hide()

    def delete_restore(self):
        """ delete restored imgs but don't reset settings """
        if hasattr(self, "stack_filtered"):
            del self.stack_filtered
        if hasattr(self, "cellpix_orig"):
            self.cellpix = self.cellpix_orig.copy()
            self.outpix = self.outpix_orig.copy()
            del self.outpix_orig, self.outpix_resize
            del self.cellpix_orig, self.cellpix_resize

    def clear_restore(self):
        """ delete restored imgs and reset settings """
        print("GUI_INFO: clearing restored image")
        self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(False)
        if self.ViewDropDown.currentIndex() == self.ViewDropDown.count() - 1:
            self.ViewDropDown.setCurrentIndex(0)
        self.delete_restore()
        self.restore = None
        self.ratio = 1.
        self.set_normalize_params(self.get_normalize_params())

    def brush_choose(self):
        self.brush_size = self.BrushChoose.currentIndex() * 2 + 1
        if self.loaded:
            self.layer.setDrawKernel(kernel_size=self.brush_size)
            self.update_layer()

    def clear_all(self):
        self.prev_selected = 0
        self.selected = 0
        if self.restore and "upsample" in self.restore:
            self.layerz = 0 * np.ones((self.Lyr, self.Lxr, 4), np.uint8)
            self.cellpix = np.zeros((self.NZ, self.Lyr, self.Lxr), np.uint16)
            self.outpix = np.zeros((self.NZ, self.Lyr, self.Lxr), np.uint16)
            self.cellpix_resize = self.cellpix.copy()
            self.outpix_resize = self.outpix.copy()
            self.cellpix_orig = np.zeros((self.NZ, self.Ly0, self.Lx0), np.uint16)
            self.outpix_orig = np.zeros((self.NZ, self.Ly0, self.Lx0), np.uint16)
        else:
            self.layerz = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
            self.cellpix = np.zeros((self.NZ, self.Ly, self.Lx), np.uint16)
            self.outpix = np.zeros((self.NZ, self.Ly, self.Lx), np.uint16)

        self.cellcolors = np.array([255, 255, 255])[np.newaxis, :]
        self.ncells.reset()
        self.toggle_removals()
        self.update_scale()
        self.update_layer()

    def select_cell(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected > 0:
            z = self.currentZ
            self.layerz[self.cellpix[z] == idx] = np.array(
                [255, 255, 255, self.opacity])
            self.update_layer()

    def select_cell_multi(self, idx):
        if idx > 0:
            z = self.currentZ
            self.layerz[self.cellpix[z] == idx] = np.array(
                [255, 255, 255, self.opacity])
            self.update_layer()

    def unselect_cell(self):
        if self.selected > 0:
            idx = self.selected
            if idx < (self.ncells.get() + 1):
                z = self.currentZ
                self.layerz[self.cellpix[z] == idx] = np.append(
                    self.cellcolors[idx], self.opacity)
                if self.outlinesOn:
                    self.layerz[self.outpix[z] == idx] = np.array(self.outcolor).astype(
                        np.uint8)
                    #[0,0,0,self.opacity])
                self.update_layer()
        self.selected = 0

    def unselect_cell_multi(self, idx):
        z = self.currentZ
        self.layerz[self.cellpix[z] == idx] = np.append(self.cellcolors[idx],
                                                        self.opacity)
        if self.outlinesOn:
            self.layerz[self.outpix[z] == idx] = np.array(self.outcolor).astype(
                np.uint8)
            # [0,0,0,self.opacity])
        self.update_layer()

    def remove_cell(self, idx):
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        # because the function remove_single_cell updates the state of the cellpix and outpix arrays
        # by reindexing cells to avoid gaps in the indices, we need to remove the cells in reverse order
        # so that the indices are correct
        idx.sort(reverse=True)
        for i in idx:
            self.remove_single_cell(i)
        self.ncells -= len(idx)  # _save_sets uses ncells
        self.update_layer()

        if self.ncells == 0:
            self.ClearButton.setEnabled(False)
        if self.NZ == 1:
            io._save_sets_with_check(self)


    def remove_single_cell(self, idx):
        # remove from manual array
        self.selected = 0
        if self.NZ > 1:
            zextent = ((self.cellpix == idx).sum(axis=(1, 2)) > 0).nonzero()[0]
        else:
            zextent = [0]
        for z in zextent:
            cp = self.cellpix[z] == idx
            op = self.outpix[z] == idx
            # remove from self.cellpix and self.outpix
            self.cellpix[z, cp] = 0
            self.outpix[z, op] = 0
            if z == self.currentZ:
                # remove from mask layer
                self.layerz[cp] = np.array([0, 0, 0, 0])

        # reduce other pixels by -1
        self.cellpix[self.cellpix > idx] -= 1
        self.outpix[self.outpix > idx] -= 1

        if self.NZ == 1:
            self.removed_cell = [
                self.ismanual[idx - 1], self.cellcolors[idx],
                np.nonzero(cp),
                np.nonzero(op)
            ]
            self.redo.setEnabled(True)
            ar, ac = self.removed_cell[2]
            d = datetime.datetime.now()
            self.track_changes.append(
                [d.strftime("%m/%d/%Y, %H:%M:%S"), "removed mask", [ar, ac]])
        # remove cell from lists
        self.ismanual = np.delete(self.ismanual, idx - 1)
        self.cellcolors = np.delete(self.cellcolors, [idx], axis=0)
        del self.zdraw[idx - 1]
        print("GUI_INFO: removed cell %d" % (idx - 1))

    def remove_region_cells(self):
        if self.removing_cells_list:
            for idx in self.removing_cells_list:
                self.unselect_cell_multi(idx)
            self.removing_cells_list.clear()
        self.disable_buttons_removeROIs()
        self.removing_region = True

        self.clear_multi_selected_cells()

        # make roi region here in center of view, making ROI half the size of the view
        roi_width = self.p0.viewRect().width() / 2
        x_loc = self.p0.viewRect().x() + (roi_width / 2)
        roi_height = self.p0.viewRect().height() / 2
        y_loc = self.p0.viewRect().y() + (roi_height / 2)

        pos = [x_loc, y_loc]
        roi = pg.RectROI(pos, [roi_width, roi_height], pen=pg.mkPen("y", width=2),
                         removable=True)
        roi.sigRemoveRequested.connect(self.remove_roi)
        roi.sigRegionChangeFinished.connect(self.roi_changed)
        self.p0.addItem(roi)
        self.remove_roi_obj = roi
        self.roi_changed(roi)

    def delete_multiple_cells(self):
        self.unselect_cell()
        self.disable_buttons_removeROIs()
        self.DoneDeleteMultipleROIButton.setEnabled(True)
        self.MakeDeletionRegionButton.setEnabled(True)
        self.CancelDeleteMultipleROIButton.setEnabled(True)
        self.deleting_multiple = True

    def done_remove_multiple_cells(self):
        self.deleting_multiple = False
        self.removing_region = False
        self.DoneDeleteMultipleROIButton.setEnabled(False)
        self.MakeDeletionRegionButton.setEnabled(False)
        self.CancelDeleteMultipleROIButton.setEnabled(False)

        if self.removing_cells_list:
            self.removing_cells_list = list(set(self.removing_cells_list))
            display_remove_list = [i - 1 for i in self.removing_cells_list]
            print(f"GUI_INFO: removing cells: {display_remove_list}")
            self.remove_cell(self.removing_cells_list)
            self.removing_cells_list.clear()
            self.unselect_cell()
        self.enable_buttons()

        if self.remove_roi_obj is not None:
            self.remove_roi(self.remove_roi_obj)

    def merge_cells(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected != self.prev_selected:
            for z in range(self.NZ):
                ar0, ac0 = np.nonzero(self.cellpix[z] == self.prev_selected)
                ar1, ac1 = np.nonzero(self.cellpix[z] == self.selected)
                touching = np.logical_and((ar0[:, np.newaxis] - ar1) < 3,
                                          (ac0[:, np.newaxis] - ac1) < 3).sum()
                ar = np.hstack((ar0, ar1))
                ac = np.hstack((ac0, ac1))
                vr0, vc0 = np.nonzero(self.outpix[z] == self.prev_selected)
                vr1, vc1 = np.nonzero(self.outpix[z] == self.selected)
                self.outpix[z, vr0, vc0] = 0
                self.outpix[z, vr1, vc1] = 0
                if touching > 0:
                    mask = np.zeros((np.ptp(ar) + 4, np.ptp(ac) + 4), np.uint8)
                    mask[ar - ar.min() + 2, ac - ac.min() + 2] = 1
                    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
                    pvc, pvr = contours[-2][0].squeeze().T
                    vr, vc = pvr + ar.min() - 2, pvc + ac.min() - 2

                else:
                    vr = np.hstack((vr0, vr1))
                    vc = np.hstack((vc0, vc1))
                color = self.cellcolors[self.prev_selected]
                self.draw_mask(z, ar, ac, vr, vc, color, idx=self.prev_selected)
            self.remove_cell(self.selected)
            print("GUI_INFO: merged two cells")
            self.update_layer()
            io._save_sets_with_check(self)
            self.undo.setEnabled(False)
            self.redo.setEnabled(False)

    def undo_remove_cell(self):
        if len(self.removed_cell) > 0:
            z = 0
            ar, ac = self.removed_cell[2]
            vr, vc = self.removed_cell[3]
            color = self.removed_cell[1]
            self.draw_mask(z, ar, ac, vr, vc, color)
            self.toggle_mask_ops()
            self.cellcolors = np.append(self.cellcolors, color[np.newaxis, :], axis=0)
            self.ncells += 1
            self.ismanual = np.append(self.ismanual, self.removed_cell[0])
            self.zdraw.append([])
            print(">>> added back removed cell")
            self.update_layer()
            io._save_sets_with_check(self)
            self.removed_cell = []
            self.redo.setEnabled(False)

    def remove_stroke(self, delete_points=True, stroke_ind=-1):
        stroke = np.array(self.strokes[stroke_ind])
        cZ = self.currentZ
        inZ = stroke[0, 0] == cZ
        if inZ:
            outpix = self.outpix[cZ, stroke[:, 1], stroke[:, 2]] > 0
            self.layerz[stroke[~outpix, 1], stroke[~outpix, 2]] = np.array([0, 0, 0, 0])
            cellpix = self.cellpix[cZ, stroke[:, 1], stroke[:, 2]]
            ccol = self.cellcolors.copy()
            if self.selected > 0:
                ccol[self.selected] = np.array([255, 255, 255])
            col2mask = ccol[cellpix]
            if self.masksOn:
                col2mask = np.concatenate(
                    (col2mask, self.opacity * (cellpix[:, np.newaxis] > 0)), axis=-1)
            else:
                col2mask = np.concatenate((col2mask, 0 * (cellpix[:, np.newaxis] > 0)),
                                          axis=-1)
            self.layerz[stroke[:, 1], stroke[:, 2], :] = col2mask
            if self.outlinesOn:
                self.layerz[stroke[outpix, 1], stroke[outpix,
                                                      2]] = np.array(self.outcolor)
            if delete_points:
                del self.current_point_set[stroke_ind]
            self.update_layer()

        del self.strokes[stroke_ind]

    def plot_clicked(self, event):
        if event.button()==QtCore.Qt.LeftButton \
                and not event.modifiers() & (QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier)\
                and not self.removing_region:
            if event.double():
                try:
                    self.p0.setYRange(0, self.Ly + self.pr)
                except:
                    self.p0.setYRange(0, self.Ly)
                self.p0.setXRange(0, self.Lx)

    def cancel_remove_multiple(self):
        self.clear_multi_selected_cells()
        self.done_remove_multiple_cells()

    def clear_multi_selected_cells(self):
        # unselect all previously selected cells:
        for idx in self.removing_cells_list:
            self.unselect_cell_multi(idx)
        self.removing_cells_list.clear()

    def add_roi(self, roi):
        self.p0.addItem(roi)
        self.remove_roi_obj = roi

    def remove_roi(self, roi):
        self.clear_multi_selected_cells()
        assert roi == self.remove_roi_obj
        self.remove_roi_obj = None
        self.p0.removeItem(roi)
        self.removing_region = False

    def roi_changed(self, roi):
        # find the overlapping cells and make them selected
        pos = roi.pos()
        size = roi.size()
        x0 = int(pos.x())
        y0 = int(pos.y())
        x1 = int(pos.x() + size.x())
        y1 = int(pos.y() + size.y())
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > self.Lx:
            x1 = self.Lx
        if y1 > self.Ly:
            y1 = self.Ly

        # find cells in that region
        cell_idxs = np.unique(self.cellpix[self.currentZ, y0:y1, x0:x1])
        cell_idxs = np.trim_zeros(cell_idxs)
        # deselect cells not in region by deselecting all and then selecting the ones in the region
        self.clear_multi_selected_cells()

        for idx in cell_idxs:
            self.select_cell_multi(idx)
            self.removing_cells_list.append(idx)

        self.update_layer()

    def mouse_moved(self, pos):
        items = self.win.scene().items(pos)

    def color_choose(self):
        self.color = self.RGBDropDown.currentIndex()
        self.view = 0
        self.ViewDropDown.setCurrentIndex(self.view)
        self.update_plot()

    def update_plot(self):
        self.view = self.ViewDropDown.currentIndex()
        # handle virtual stack with only one plane resident; show the loaded plane (index 0)
        if hasattr(self, "lazy_data") and self.lazy_data is not None and len(self.stack) == 1 and getattr(self, "NZ", 1) > 1:
            stack_index = 0
        else:
            stack_index = self.currentZ if self.currentZ < len(self.stack) else 0
        self.Ly, self.Lx, _ = self.stack[stack_index].shape

        if self.view == 0 or self.view == self.ViewDropDown.count() - 1:
            image = self.stack[
                stack_index] if self.view == 0 else self.stack_filtered[stack_index]
            if self.color == 0:
                self.img.setImage(image, autoLevels=False, lut=None)
                if self.nchan > 1:
                    levels = np.array([
                        self.saturation[0][self.currentZ],
                        self.saturation[1][self.currentZ],
                        self.saturation[2][self.currentZ]
                    ])
                    self.img.setLevels(levels)
                else:
                    self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color > 0 and self.color < 4:
                if self.nchan > 1:
                    image = image[:, :, self.color - 1]
                self.img.setImage(image, autoLevels=False, lut=self.cmap[self.color])
                if self.nchan > 1:
                    self.img.setLevels(self.saturation[self.color - 1][self.currentZ])
                else:
                    self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color == 4:
                if self.nchan > 1:
                    image = image.mean(axis=-1)
                self.img.setImage(image, autoLevels=False, lut=None)
                self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color == 5:
                if self.nchan > 1:
                    image = image.mean(axis=-1)
                self.img.setImage(image, autoLevels=False, lut=self.cmap[0])
                self.img.setLevels(self.saturation[0][self.currentZ])
        else:
            image = np.zeros((self.Ly, self.Lx), np.uint8)
            if len(self.flows) >= self.view - 1 and len(self.flows[self.view - 1]) > 0:
                image = self.flows[self.view - 1][self.currentZ]
            if self.view > 1:
                self.img.setImage(image, autoLevels=False, lut=self.bwr)
            else:
                self.img.setImage(image, autoLevels=False, lut=None)
            self.img.setLevels([0.0, 255.0])

        for r in range(3):
            self.sliders[r].setValue([
                self.saturation[r][self.currentZ][0],
                self.saturation[r][self.currentZ][1]
            ])
        self.win.show()
        self.show()


    def update_layer(self):
        if self.masksOn or self.outlinesOn:
            self.layer.setImage(self.layerz, autoLevels=False)
        self.win.show()
        self.show()


    def add_set(self):
        if len(self.current_point_set) > 0:
            while len(self.strokes) > 0:
                self.remove_stroke(delete_points=False)
            if len(self.current_point_set[0]) > 8:
                color = self.colormap[self.ncells.get(), :3]
                median = self.add_mask(points=self.current_point_set, color=color)
                if median is not None:
                    self.removed_cell = []
                    self.toggle_mask_ops()
                    self.cellcolors = np.append(self.cellcolors, color[np.newaxis, :],
                                                axis=0)
                    self.ncells += 1
                    self.ismanual = np.append(self.ismanual, True)
                    if self.NZ == 1:
                        # only save after each cell if single image
                        io._save_sets_with_check(self)
            else:
                print("GUI_ERROR: cell too small, not drawn")
            self.current_stroke = []
            self.strokes = []
            self.current_point_set = []
            self.update_layer()

    def add_mask(self, points=None, color=(100, 200, 50), dense=True):
        # points is list of strokes
        points_all = np.concatenate(points, axis=0)
        
        # loop over z values
        median = []
        zdraw = np.unique(points_all[:, 0])
        z = 0
        ars, acs, vrs, vcs = np.zeros(0, "int"), np.zeros(0, "int"), np.zeros(
            0, "int"), np.zeros(0, "int")
        for stroke in points:
            stroke = np.concatenate(stroke, axis=0).reshape(-1, 4)
            vr = stroke[:, 1]
            vc = stroke[:, 2]
            # get points inside drawn points
            mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
            pts = np.stack((vc - vc.min() + 2, vr - vr.min() + 2),
                           axis=-1)[:, np.newaxis, :]
            mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
            ar, ac = np.nonzero(mask)
            ar, ac = ar + vr.min() - 2, ac + vc.min() - 2
            # get dense outline
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = contours[-2][0][:,0].T
            vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
            # concatenate all points
            ar, ac = np.hstack((np.vstack((vr, vc)), np.vstack((ar, ac))))
            # if these pixels are overlapping with another cell, reassign them
            ioverlap = self.cellpix[z][ar, ac] > 0
            if (~ioverlap).sum() < 10:
                print("GUI_ERROR: cell < 10 pixels without overlaps, not drawn")
                return None
            elif ioverlap.sum() > 0:
                ar, ac = ar[~ioverlap], ac[~ioverlap]
                # compute outline of new mask
                mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
                mask[ar - vr.min() + 2, ac - vc.min() + 2] = 1
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
                pvc, pvr = contours[-2][0][:,0].T
                vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
            ars = np.concatenate((ars, ar), axis=0)
            acs = np.concatenate((acs, ac), axis=0)
            vrs = np.concatenate((vrs, vr), axis=0)
            vcs = np.concatenate((vcs, vc), axis=0)
            
        self.draw_mask(z, ars, acs, vrs, vcs, color)
        median.append(np.array([np.median(ars), np.median(acs)]))

        self.zdraw.append(zdraw)
        d = datetime.datetime.now()
        self.track_changes.append(
            [d.strftime("%m/%d/%Y, %H:%M:%S"), "added mask", [ar, ac]])
        return median

    def draw_mask(self, z, ar, ac, vr, vc, color, idx=None):
        """ draw single mask using outlines and area """
        if idx is None:
            idx = self.ncells + 1
        self.cellpix[z, vr, vc] = idx
        self.cellpix[z, ar, ac] = idx
        self.outpix[z, vr, vc] = idx
        if self.restore and "upsample" in self.restore:
            if self.resize:
                self.cellpix_resize[z, vr, vc] = idx
                self.cellpix_resize[z, ar, ac] = idx
                self.outpix_resize[z, vr, vc] = idx
                self.cellpix_orig[z, (vr / self.ratio).astype(int),
                                  (vc / self.ratio).astype(int)] = idx
                self.cellpix_orig[z, (ar / self.ratio).astype(int),
                                  (ac / self.ratio).astype(int)] = idx
                self.outpix_orig[z, (vr / self.ratio).astype(int),
                                 (vc / self.ratio).astype(int)] = idx
            else:
                self.cellpix_orig[z, vr, vc] = idx
                self.cellpix_orig[z, ar, ac] = idx
                self.outpix_orig[z, vr, vc] = idx

                # get upsampled mask
                vrr = (vr.copy() * self.ratio).astype(int)
                vcr = (vc.copy() * self.ratio).astype(int)
                mask = np.zeros((np.ptp(vrr) + 4, np.ptp(vcr) + 4), np.uint8)
                pts = np.stack((vcr - vcr.min() + 2, vrr - vrr.min() + 2),
                               axis=-1)[:, np.newaxis, :]
                mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
                arr, acr = np.nonzero(mask)
                arr, acr = arr + vrr.min() - 2, acr + vcr.min() - 2
                # get dense outline
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
                pvc, pvr = contours[-2][0].squeeze().T
                vrr, vcr = pvr + vrr.min() - 2, pvc + vcr.min() - 2
                # concatenate all points
                arr, acr = np.hstack((np.vstack((vrr, vcr)), np.vstack((arr, acr))))
                self.cellpix_resize[z, vrr, vcr] = idx
                self.cellpix_resize[z, arr, acr] = idx
                self.outpix_resize[z, vrr, vcr] = idx

        if z == self.currentZ:
            self.layerz[ar, ac, :3] = color
            if self.masksOn:
                self.layerz[ar, ac, -1] = self.opacity
            if self.outlinesOn:
                self.layerz[vr, vc] = np.array(self.outcolor)

    def compute_scale(self):
        # get diameter from gui
        diameter = self.segmentation_settings.diameter
        if not diameter:
            diameter = 30

        self.pr = int(diameter)
        self.radii_padding = int(self.pr * 1.25)
        self.radii = np.zeros((self.Ly + self.radii_padding, self.Lx, 4), np.uint8)
        yy, xx = disk([self.Ly + self.radii_padding / 2 - 1, self.pr / 2 + 1],
                      self.pr / 2, self.Ly + self.radii_padding, self.Lx)
        # rgb(150,50,150)
        self.radii[yy, xx, 0] = 150
        self.radii[yy, xx, 1] = 50
        self.radii[yy, xx, 2] = 150
        self.radii[yy, xx, 3] = 255
        self.p0.setYRange(0, self.Ly + self.radii_padding)
        self.p0.setXRange(0, self.Lx)

    def update_scale(self):
        self.compute_scale()
        self.scale.setImage(self.radii, autoLevels=False)
        self.scale.setLevels([0.0, 255.0])
        self.win.show()
        self.show()


    def draw_layer(self):
        if self.resize:
            self.Ly, self.Lx = self.Lyr, self.Lxr
        else:
            self.Ly, self.Lx = self.Ly0, self.Lx0

        if self.masksOn or self.outlinesOn:
            if self.restore and "upsample" in self.restore:
                if self.resize:
                    self.cellpix = self.cellpix_resize.copy()
                    self.outpix = self.outpix_resize.copy()
                else:
                    self.cellpix = self.cellpix_orig.copy()
                    self.outpix = self.outpix_orig.copy()

        self.layerz = np.zeros((self.Ly, self.Lx, 4), np.uint8)
        if self.masksOn:
            self.layerz[..., :3] = self.cellcolors[self.cellpix[self.currentZ], :]
            self.layerz[..., 3] = self.opacity * (self.cellpix[self.currentZ]
                                                  > 0).astype(np.uint8)
            if self.selected > 0:
                self.layerz[self.cellpix[self.currentZ] == self.selected] = np.array(
                    [255, 255, 255, self.opacity])
            cZ = self.currentZ
            stroke_z = np.array([s[0][0] for s in self.strokes])
            inZ = np.nonzero(stroke_z == cZ)[0]
            if len(inZ) > 0:
                for i in inZ:
                    stroke = np.array(self.strokes[i])
                    self.layerz[stroke[:, 1], stroke[:,
                                                     2]] = np.array([255, 0, 255, 100])
        else:
            self.layerz[..., 3] = 0

        if self.outlinesOn:
            self.layerz[self.outpix[self.currentZ] > 0] = np.array(
                self.outcolor).astype(np.uint8)


    def set_normalize_params(self, normalize_params):
        from cellpose.models import normalize_default
        if self.restore != "filter":
            keys = list(normalize_params.keys()).copy()
            for key in keys:
                if key != "percentile":
                    normalize_params[key] = normalize_default[key]
        normalize_params = {**normalize_default, **normalize_params}
        out = self.check_filter_params(normalize_params["sharpen_radius"],
                                       normalize_params["smooth_radius"],
                                       normalize_params["tile_norm_blocksize"],
                                       normalize_params["tile_norm_smooth3D"],
                                       normalize_params["norm3D"],
                                       normalize_params["invert"])


    def check_filter_params(self, sharpen, smooth, tile_norm, smooth3D, norm3D, invert):
        tile_norm = 0 if tile_norm < 0 else tile_norm
        sharpen = 0 if sharpen < 0 else sharpen
        smooth = 0 if smooth < 0 else smooth
        smooth3D = 0 if smooth3D < 0 else smooth3D
        norm3D = bool(norm3D)
        invert = bool(invert)
        if tile_norm > self.Ly and tile_norm > self.Lx:
            print(
                "GUI_ERROR: tile size (tile_norm) bigger than both image dimensions, disabling"
            )
            tile_norm = 0
        self.filt_edits[0].setText(str(sharpen))
        self.filt_edits[1].setText(str(smooth))
        self.filt_edits[2].setText(str(tile_norm))
        self.filt_edits[3].setText(str(smooth3D))
        self.norm3D_cb.setChecked(norm3D)
        return sharpen, smooth, tile_norm, smooth3D, norm3D, invert

    def get_normalize_params(self):
        percentile = [
            self.segmentation_settings.low_percentile,
            self.segmentation_settings.high_percentile,
        ]
        normalize_params = {"percentile": percentile}
        norm3D = self.norm3D_cb.isChecked()
        normalize_params["norm3D"] = norm3D
        sharpen = float(self.filt_edits[0].text())
        smooth = float(self.filt_edits[1].text())
        tile_norm = float(self.filt_edits[2].text())
        smooth3D = float(self.filt_edits[3].text())
        invert = False
        out = self.check_filter_params(sharpen, smooth, tile_norm, smooth3D, norm3D,
                                        invert)
        sharpen, smooth, tile_norm, smooth3D, norm3D, invert = out
        normalize_params["sharpen_radius"] = sharpen
        normalize_params["smooth_radius"] = smooth
        normalize_params["tile_norm_blocksize"] = tile_norm
        normalize_params["tile_norm_smooth3D"] = smooth3D
        normalize_params["invert"] = invert

        from cellpose.models import normalize_default
        normalize_params = {**normalize_default, **normalize_params}

        return normalize_params
    
    def compute_saturation_if_checked(self):
        if self.autobtn.isChecked():
            self.compute_saturation()

    def compute_saturation(self, return_img=False):
        # special handling for lazy virtual stacks: only current plane is in memory
        if hasattr(self, "lazy_data") and self.lazy_data is not None and getattr(self, "NZ", 1) > 1 and getattr(self.stack, "shape", [1])[0] == 1:
            # If we already computed global levels for this lazy source, reuse them for all T/Z
            if hasattr(self, "global_sat") and self.global_sat is not None:
                levels = self.global_sat
                nch = max(3, len(levels))
                self.saturation = []
                for c in range(nch):
                    vals = [levels[c] if c < len(levels) else [0, 255.] for _ in range(self.NZ)]
                    self.saturation.append(vals)
                if return_img:
                    return self.stack
                self.update_plot()
                return
            norm = self.get_normalize_params()
            percentile = norm["percentile"]
            invert = norm["invert"]
            img_plane = self.stack[0]  # (Y,X,C)
            levels = []
            for c in range(img_plane.shape[-1]):
                if np.ptp(img_plane[..., c]) > 1e-3:
                    x01 = np.percentile(img_plane[..., c], percentile[0])
                    x99 = np.percentile(img_plane[..., c], percentile[1])
                    if invert:
                        x01i = 255. - x99
                        x99i = 255. - x01
                        x01, x99 = x01i, x99i
                else:
                    x01, x99 = 0., 255.
                levels.append([x01, x99])
            nch = max(3, len(levels))
            self.saturation = []
            for c in range(nch):
                vals = [levels[c] if c < len(levels) else [0, 255.] for _ in range(self.NZ)]
                self.saturation.append(vals)
            # store a single global saturation for the lazy source so it stays static across T/Z
            self.global_sat = levels
            if return_img:
                return self.stack
            self.update_plot()
            return
        # reuse cached saturation for lazy time stacks
        if hasattr(self, "sat_cache") and self.sat_cache is not None and getattr(self, "currentT", None) in self.sat_cache:
            self.saturation = copy.deepcopy(self.sat_cache[self.currentT])
            if return_img:
                return self.stack
            self.update_plot()
            return
        norm = self.get_normalize_params()
        print(norm)
        sharpen, smooth = norm["sharpen_radius"], norm["smooth_radius"]
        percentile = norm["percentile"]
        tile_norm = norm["tile_norm_blocksize"]
        invert = norm["invert"]
        norm3D = norm["norm3D"]
        smooth3D = norm["tile_norm_smooth3D"]
        tile_norm = norm["tile_norm_blocksize"]

        if sharpen > 0 or smooth > 0 or tile_norm > 0:
            img_norm = self.stack.copy()
        else:
            img_norm = self.stack

        if sharpen > 0 or smooth > 0 or tile_norm > 0:
            self.restore = "filter"
            print(
                "GUI_INFO: computing filtered image because sharpen > 0 or tile_norm > 0"
            )
            print(
                "GUI_WARNING: will use memory to create filtered image -- make sure to have RAM for this"
            )
            img_norm = self.stack.copy()
            if sharpen > 0 or smooth > 0:
                img_norm = smooth_sharpen_img(self.stack, sharpen_radius=sharpen,
                                              smooth_radius=smooth)

            if tile_norm > 0:
                img_norm = normalize99_tile(img_norm, blocksize=tile_norm,
                                            lower=percentile[0], upper=percentile[1],
                                            smooth3D=smooth3D, norm3D=norm3D)
            # convert to 0->255
            img_norm_min = img_norm.min()
            img_norm_max = img_norm.max()
            for c in range(img_norm.shape[-1]):
                if np.ptp(img_norm[..., c]) > 1e-3:
                    img_norm[..., c] -= img_norm_min
                    img_norm[..., c] /= (img_norm_max - img_norm_min)
            img_norm *= 255
            self.stack_filtered = img_norm
            self.ViewDropDown.model().item(self.ViewDropDown.count() -
                                           1).setEnabled(True)
            self.ViewDropDown.setCurrentIndex(self.ViewDropDown.count() - 1)
        else:
            img_norm = self.stack if self.restore is None or self.restore == "filter" else self.stack_filtered

        if self.autobtn.isChecked():
            self.saturation = []
            for c in range(img_norm.shape[-1]):
                self.saturation.append([])
                if np.ptp(img_norm[..., c]) > 1e-3:
                    if norm3D:
                        x01 = np.percentile(img_norm[..., c], percentile[0])
                        x99 = np.percentile(img_norm[..., c], percentile[1])
                        if invert:
                            x01i = 255. - x99
                            x99i = 255. - x01
                            x01, x99 = x01i, x99i
                        for n in range(self.NZ):
                            self.saturation[-1].append([x01, x99])
                    else:
                        for z in range(self.NZ):
                            if self.NZ > 1:
                                x01 = np.percentile(img_norm[z, :, :, c], percentile[0])
                                x99 = np.percentile(img_norm[z, :, :, c], percentile[1])
                            else:
                                x01 = np.percentile(img_norm[..., c], percentile[0])
                                x99 = np.percentile(img_norm[..., c], percentile[1])
                            if invert:
                                x01i = 255. - x99
                                x99i = 255. - x01
                                x01, x99 = x01i, x99i
                            self.saturation[-1].append([x01, x99])
                else:
                    for n in range(self.NZ):
                        self.saturation[-1].append([0, 255.])
            # pad saturation to 3 channels for UI safety
            while len(self.saturation) < 3:
                self.saturation.append(self.saturation[-1] if len(self.saturation) > 0 else [[0, 255.] for _ in range(self.NZ)])

            # cache saturation per timepoint for lazy stacks
            if hasattr(self, "sat_cache") and self.sat_cache is not None and getattr(self, "currentT", None) is not None:
                self.sat_cache = {self.currentT: copy.deepcopy(self.saturation)}

        # self.autobtn.setChecked(True)
        self.update_plot()


    def get_model_path(self, custom=False):
        if custom:
            self.current_model = self.ModelChooseC.currentText()
            self.current_model_path = os.fspath(
                models.MODEL_DIR.joinpath(self.current_model))
        else:
            self.current_model = "cpsam"
            self.current_model_path = models.model_path(self.current_model)

    def initialize_model(self, model_name=None, custom=False):
        if model_name is None or custom:
            self.get_model_path(custom=custom)
            if not os.path.exists(self.current_model_path):
                raise ValueError("Model file not found: need to specify model (use dropdown)")

        if model_name is None or not isinstance(model_name, str):
            self.model = models.CellposeModel(gpu=self.useGPU.isChecked(),
                                              pretrained_model=self.current_model_path)
        else:
            self.current_model = model_name
            self.current_model_path = os.fspath(
                models.MODEL_DIR.joinpath(self.current_model))

            self.model = models.CellposeModel(gpu=self.useGPU.isChecked(),
                                             pretrained_model=self.current_model)

    def add_model(self):
        io._add_model(self)
        return

    def remove_model(self):
        io._remove_model(self)
        return

    def new_model(self):
        if self.NZ != 1:
            print("ERROR: cannot train model on 3D data")
            return

        # train model
        image_names = self.get_files()[0]
        self.train_data, self.train_labels, self.train_files, restore, normalize_params = io._get_train_set(
            image_names)
        train_model_choices = []
        for name in list(models.MODEL_NAMES) + list(self.model_strings):
            if isinstance(name, str) and len(name) > 0 and name not in train_model_choices:
                train_model_choices.append(name)
        if len(train_model_choices) == 0:
            train_model_choices = ["cpsam"]
        TW = guiparts.TrainWindow(self, train_model_choices)
        train = TW.exec_()
        if train:
            self.logger.info(
                f"training with {[os.path.split(f)[1] for f in self.train_files]}")
            self.train_model(restore=restore, normalize_params=normalize_params)
        else:
            print("GUI_INFO: training cancelled")

    def train_model(self, restore=None, normalize_params=None):
        from cellpose.models import normalize_default
        if normalize_params is None:
            normalize_params = copy.deepcopy(normalize_default)

        train_model_choices = []
        for name in list(models.MODEL_NAMES) + list(self.model_strings):
            if isinstance(name, str) and len(name) > 0 and name not in train_model_choices:
                train_model_choices.append(name)
        if len(train_model_choices) == 0:
            train_model_choices = ["cpsam"]

        initial_model = self.training_params.get("initial_model", None)
        if not isinstance(initial_model, str) or len(initial_model) == 0:
            try:
                model_index = int(self.training_params.get("model_index", 0))
            except Exception:
                model_index = 0
            if model_index < 0 or model_index >= len(train_model_choices):
                model_index = 0
            initial_model = train_model_choices[model_index]
        if initial_model not in train_model_choices and not os.path.exists(initial_model):
            self.logger.warning(
                f"initial model {initial_model} not found; falling back to cpsam")
            initial_model = "cpsam"

        self.logger.info(f"training new model starting at model {initial_model}")
        self.current_model = initial_model

        self.model = models.CellposeModel(gpu=self.useGPU.isChecked(),
                                          pretrained_model=initial_model)
        save_path = os.path.dirname(self.filename)

        print("GUI_INFO: name of new model: " + self.training_params["model_name"])
        self.new_model_path, train_losses = train.train_seg(
            self.model.net, train_data=self.train_data, train_labels=self.train_labels,
            normalize=normalize_params, min_train_masks=0,
            save_path=save_path, nimg_per_epoch=max(2, len(self.train_data)),
            learning_rate=self.training_params["learning_rate"],
            weight_decay=self.training_params["weight_decay"],
            n_epochs=self.training_params["n_epochs"],
            model_name=self.training_params["model_name"])[:2]
        # save train losses
        np.save(str(self.new_model_path) + "_train_losses.npy", train_losses)
        # run model on next image
        io._add_model(self, self.new_model_path)
        diam_labels = self.model.net.diam_labels.item()  #.copy()
        self.new_model_ind = len(self.model_strings)
        self.autorun = True
        self.clear_all()
        self.restore = restore
        self.set_normalize_params(normalize_params)
        self.get_next_image(load_seg=False)

        self.compute_segmentation(custom=True)
        self.logger.info(
            f"!!! computed masks for {os.path.split(self.filename)[1]} from new model !!!"
        )


    def compute_cprob(self):
        if self.recompute_masks:
            flow_threshold = self.segmentation_settings.flow_threshold
            cellprob_threshold = self.segmentation_settings.cellprob_threshold
            niter = self.segmentation_settings.niter
            min_size = int(self.min_size.text()) if not isinstance(
                self.min_size, int) else self.min_size

            self.logger.info(
                    "computing masks with cell prob=%0.3f, flow error threshold=%0.3f" %
                    (cellprob_threshold, flow_threshold))
            
            try:
                dP = self.flows[2].squeeze()
                cellprob = self.flows[3].squeeze()
            except IndexError:
                self.logger.error("Flows don't exist, try running model again.")
                return
            
            maski = dynamics.compute_masks_and_clean(
                dP=dP,
                cellprob=cellprob,
                niter=niter,
                do_3D=self.load_3D,
                min_size=min_size,
                # max_size_fraction=min_size_fraction, # Leave as default 
                cellprob_threshold=cellprob_threshold, 
                flow_threshold=flow_threshold)
            
            self.masksOn = True
            if not self.OCheckBox.isChecked():
                self.MCheckBox.setChecked(True)
            if maski.ndim < 3:
                maski = maski[np.newaxis, ...]
            self.logger.info("%d cells found" % (len(np.unique(maski)[1:])))
            io._masks_to_gui(self, maski, outlines=None, preserve_labels=False)
            self._current_seg_preserve_labels = False
            self.show()


    def run_custom_segmentation(self):
        """Run custom-model segmentation, optionally across a selected timepoint range."""
        if hasattr(self, "segmentation_settings") and getattr(
            self.segmentation_settings, "segment_all_timepoints", False
        ):
            self.compute_segmentation_all_timepoints(custom=True)
        else:
            seg_T = None
            if getattr(self, "has_time", False) and hasattr(self, "currentT"):
                try:
                    seg_T = int(self.currentT)
                except Exception:
                    seg_T = None
            self.start_segmentation_async(custom=True, time_index=seg_T)


    def _reset_time_stitch_state(self):
        self._seg_all_time_stitch_enabled = False
        self._seg_all_time_stitch_threshold = 0.25
        self._seg_all_time_prev_masks = None
        self._seg_all_time_next_label = 1

    @staticmethod
    def _compact_positive_labels(labels):
        """Map any positive label IDs to compact IDs [1..N], preserving background=0."""
        arr = np.asarray(labels)
        uniq = np.unique(arr)
        uniq = uniq[uniq > 0]
        if uniq.size == 0:
            return np.zeros(arr.shape, dtype=np.int32), uniq

        flat = arr.reshape(-1)
        idx = np.searchsorted(uniq, flat)
        safe_idx = np.clip(idx, 0, uniq.size - 1)
        valid = (flat > 0) & (idx < uniq.size) & (uniq[safe_idx] == flat)
        compact_flat = np.zeros(flat.shape[0], dtype=np.int32)
        compact_flat[valid] = safe_idx[valid] + 1
        return compact_flat.reshape(arr.shape), uniq

    def _stitch_masks_over_time(self, masks, time_index):
        """
        Keep stable ROI IDs across adjacent timepoints by greedy IOU matching.
        Unmatched current ROIs receive new global IDs.
        """
        curr_compact, _ = self._compact_positive_labels(masks)
        n_curr = int(curr_compact.max()) if curr_compact.size else 0
        if n_curr == 0:
            stitched = np.zeros_like(curr_compact, dtype=np.uint16)
            self._seg_all_time_prev_masks = stitched.copy()
            return stitched

        curr_to_global = np.zeros(n_curr + 1, dtype=np.int64)
        prev_masks = getattr(self, "_seg_all_time_prev_masks", None)
        next_label = int(getattr(self, "_seg_all_time_next_label", 1))
        threshold = float(getattr(self, "_seg_all_time_stitch_threshold", 0.25))

        if (
            isinstance(prev_masks, np.ndarray)
            and prev_masks.size > 0
            and prev_masks.shape == curr_compact.shape
        ):
            prev_compact, prev_global_ids = self._compact_positive_labels(prev_masks)
            if prev_global_ids.size > 0:
                iou = metrics._intersection_over_union(curr_compact, prev_compact)[1:, 1:]
                if iou.size > 0:
                    pairs = np.argwhere(iou >= threshold)
                    if pairs.size > 0:
                        scores = iou[pairs[:, 0], pairs[:, 1]]
                        order = np.argsort(scores)[::-1]
                        used_curr = np.zeros(n_curr + 1, dtype=bool)
                        used_prev = np.zeros(prev_global_ids.size + 1, dtype=bool)
                        for oi in order:
                            c_id = int(pairs[oi, 0]) + 1
                            p_id = int(pairs[oi, 1]) + 1
                            if used_curr[c_id] or used_prev[p_id]:
                                continue
                            curr_to_global[c_id] = int(prev_global_ids[p_id - 1])
                            used_curr[c_id] = True
                            used_prev[p_id] = True
        elif isinstance(prev_masks, np.ndarray) and prev_masks.size > 0:
            print(
                "GUI_WARNING: skipping time-stitch for "
                f"T={time_index} (shape changed from {prev_masks.shape} to {curr_compact.shape})"
            )

        unmatched = np.where(curr_to_global[1:] == 0)[0] + 1
        if unmatched.size > 0:
            new_ids = np.arange(next_label, next_label + unmatched.size, dtype=np.int64)
            curr_to_global[unmatched] = new_ids
            next_label = int(new_ids[-1]) + 1
        else:
            next_label = max(next_label, int(curr_to_global.max()) + 1)

        stitched = curr_to_global[curr_compact]
        max_label = int(stitched.max()) if stitched.size else 0
        if max_label < 2**16:
            stitched = stitched.astype(np.uint16, copy=False)
        elif max_label < 2**32:
            stitched = stitched.astype(np.uint32, copy=False)

        self._seg_all_time_prev_masks = stitched.copy()
        self._seg_all_time_next_label = next_label
        return stitched

    def _cache_segmentation_result(self, masks, time_index, preserve_labels=False):
        """Cache per-timepoint segmentation in memory for live browsing and export."""
        if time_index is None:
            return
        try:
            t = int(time_index)
            if not isinstance(self.seg_time_data, dict):
                self.seg_time_data = {}
            masks_arr = np.asarray(masks)
            self.seg_time_data[t] = {
                "masks": masks_arr.copy(),
                "outlines": None,
                "colors": None,
                "time_index": t,
                "axes": "ZYX" if masks_arr.ndim == 3 else "YX",
                "preserve_labels": bool(preserve_labels),
            }
        except Exception as e:
            print(f"ERROR: could not cache segmentation for T={time_index}: {e}")


    def compute_segmentation_all_timepoints(self, custom=False):
        """Run segmentation for a selected timepoint range in a 5D stack.

        For 5D inputs (e.g. .sldy via LazySldy or TCZYX virtual stacks),
        this iterates over the selected time indices and calls `compute_segmentation`
        at each timepoint, reusing the same model for speed.

        Optionally, ROI IDs can be stitched across adjacent timepoints.
        """
        if self._segmentation_running or self._segmentation_all_running:
            print("GUI_INFO: segmentation already running; please wait for it to finish")
            return

        if not getattr(self, "has_time", False) or getattr(self, "NT", 1) <= 1:
            # no time axis -> just segment current view (async)
            self.start_segmentation_async(custom=custom)
            return

        # resolve selected timepoint range (inclusive)
        nt_total = int(self.NT)
        start_t, end_t = 0, nt_total - 1
        try:
            if hasattr(self, "segmentation_settings"):
                start_t, end_t = self.segmentation_settings.get_timepoint_range(
                    nt=nt_total
                )
        except Exception as e:
            print(f"GUI_WARNING: invalid time range settings; using full range ({e})")
            start_t, end_t = 0, nt_total - 1
        if end_t < start_t:
            start_t, end_t = end_t, start_t
        selected_timepoints = list(range(int(start_t), int(end_t) + 1))
        if len(selected_timepoints) == 0:
            print("GUI_WARNING: no timepoints selected for segmentation")
            return

        # initialize async multi-timepoint segmentation state
        self._segmentation_all_running = True
        self._seg_all_queue = selected_timepoints
        self._seg_all_total = len(selected_timepoints)
        self._seg_all_current_t = None
        self._seg_all_original_T = int(getattr(self, "currentT", 0))
        self._seg_all_custom = bool(custom)
        # built-in CPSAM for non-custom runs
        self._seg_all_model_name = None if custom else "cpsam"
        self._seg_all_first = True
        print(
            "GUI_INFO: segmenting timepoints "
            f"T={selected_timepoints[0]}..{selected_timepoints[-1]} "
            f"({len(selected_timepoints)} total)"
        )
        self._reset_time_stitch_state()
        try:
            self._seg_all_time_stitch_enabled = bool(
                getattr(self.segmentation_settings, "stitch_over_time", False)
            )
            self._seg_all_time_stitch_threshold = float(
                getattr(self.segmentation_settings, "time_stitch_threshold", 0.25)
            )
        except Exception:
            self._seg_all_time_stitch_enabled = False
            self._seg_all_time_stitch_threshold = 0.25
        if self._seg_all_time_stitch_enabled:
            print(
                "GUI_INFO: 4D time stitching enabled "
                f"(IOU threshold={self._seg_all_time_stitch_threshold:.3f})"
            )
        self._segmentation_running = False
        # kick off first timepoint
        self._segmentation_all_next()


    def _segmentation_all_next(self):
        """Advance async multi-timepoint segmentation to the next time index."""
        if not self._segmentation_all_running:
            return

        if len(self._seg_all_queue) == 0:
            # finished all timepoints; mark complete and restore the timepoint
            # the user was on when the run started
            print(
                "GUI_INFO: completed timepoint-range segmentation "
                f"({int(getattr(self, '_seg_all_total', 0))} attempted)"
            )
            self._segmentation_all_running = False
            self._segmentation_running = False
            self._seg_all_current_t = None
            self._seg_all_total = 0
            self._reset_time_stitch_state()
            try:
                t_restore = int(getattr(self, "_seg_all_original_T", 0))
                if getattr(self, "has_time", False) and int(getattr(self, "NT", 1)) > 1:
                    self.set_time_index(t_restore)
            except Exception as e:
                print(f"ERROR: could not restore original timepoint after all-time run: {e}")
            return

        # get next timepoint to process (without changing the user's current view)
        t = self._seg_all_queue.pop(0)
        self._seg_all_current_t = int(t)
        seg_idx = int(getattr(self, "_seg_all_total", 0)) - len(self._seg_all_queue)
        seg_total = int(getattr(self, "_seg_all_total", 0))
        print(f"GUI_INFO: starting segmentation T={t} [{seg_idx}/{seg_total}]")

        # load model only on first timepoint; reuse for subsequent ones
        load_model = self._seg_all_first
        self._seg_all_first = False

        if self._seg_all_custom:
            self.start_segmentation_async(custom=True, load_model=load_model, time_index=t)
        else:
            self.start_segmentation_async(
                model_name=self._seg_all_model_name, load_model=load_model, time_index=t
            )


    def _prepare_segmentation_inputs(self, custom=False, model_name=None, load_model=True, time_index=None):
        """Prepare data and parameters for segmentation (shared by sync/async paths)."""
        tic = time.time()
        # Determine target timepoint first so we can avoid clearing the visible layer
        # when segmenting a different timepoint in the background.
        seg_T = time_index
        if seg_T is None and getattr(self, "has_time", False) and hasattr(self, "currentT"):
            try:
                seg_T = int(self.currentT)
            except Exception:
                seg_T = None

        should_clear_view = True
        if seg_T is not None:
            try:
                should_clear_view = int(getattr(self, "currentT", 0)) == int(seg_T)
            except Exception:
                should_clear_view = True
        if should_clear_view:
            self.clear_all()
            self.flows = [[], [], []]

        if load_model:
            self.initialize_model(model_name=model_name, custom=custom)
        self.progress.setValue(10)

        # store segmentation time index on the object so saving is independent
        # from GUI slider changes during the run
        self._current_seg_time_index = seg_T

        do_3D = self.load_3D
        stitch_threshold = (
            float(self.stitch_threshold.text())
            if not isinstance(self.stitch_threshold, float)
            else self.stitch_threshold
        )
        anisotropy = (
            float(self.anisotropy.text())
            if not isinstance(self.anisotropy, float)
            else self.anisotropy
        )
        flow3D_smooth = (
            float(self.flow3D_smooth.text())
            if not isinstance(self.flow3D_smooth, float)
            else self.flow3D_smooth
        )
        min_size = (
            int(self.min_size.text())
            if not isinstance(self.min_size, int)
            else self.min_size
        )
        downscale = getattr(self.segmentation_settings, "downscale_factor", 1.0)

        do_3D = False if stitch_threshold > 0.0 else do_3D

        # Heuristic: for huge 3D volumes from .sldy, prefer 2D+stitch
        # to avoid extremely long 3D runs.
        volume_pixels = int(self.NZ) * int(self.Ly0) * int(self.Lx0)
        if do_3D and volume_pixels > 2_00_000_000:  # tweak threshold as you like
            print(
                f"GUI_INFO: volume {self.NZ}x{self.Ly0}x{self.Lx0} is very large; "
                "switching from 3D to 2D+stitch for speed"
            )
            do_3D = False
            if stitch_threshold <= 0.0:
                stitch_threshold = 0.4

        # choose segmentation data
        # Always pick data by explicit time index (seg_T) to decouple from GUI slider.
        if hasattr(self, "lazy_data") and self.lazy_data is not None:
            t_index_for_data = 0 if seg_T is None else int(seg_T)
            data = self.lazy_data.get_time_stack(t_index_for_data)
        elif getattr(self, "has_time", False) and self.time_stack is not None and seg_T is not None:
            try:
                data = self.time_stack[int(seg_T)]
            except Exception:
                # fallback to current view if direct indexing fails
                data = self.stack_filtered.copy().squeeze() if self.restore == "filter" else self.stack.copy().squeeze()
        else:
            if self.restore == "filter":
                data = self.stack_filtered.copy().squeeze()
            else:
                data = self.stack.copy().squeeze()

        flow_threshold = self.segmentation_settings.flow_threshold
        cellprob_threshold = self.segmentation_settings.cellprob_threshold
        diameter = self.segmentation_settings.diameter
        niter = self.segmentation_settings.niter

        # optional XY downscaling to speed up segmentation
        if downscale is not None and downscale < 1.0:
            try:
                if data.ndim == 4:
                    # Z, Y, X, C
                    data = resize_image(data, rsz=downscale, no_channels=False)
                elif data.ndim == 3:
                    # treat third dim as X; first dim is Z if stack
                    data = resize_image(data, rsz=downscale, no_channels=True)
                elif data.ndim == 2:
                    data = resize_image(data, rsz=downscale, no_channels=False)
            except Exception as e:
                print(
                    f"GUI_WARNING: downscale failed, continuing without resizing ({e})"
                )
                downscale = 1.0

        # adjust diameter to new pixel scale if provided
        if diameter is not None and downscale is not None and downscale < 1.0:
            diameter = diameter * downscale

        normalize_params = self.get_normalize_params()
        print(normalize_params)

        # ensure z_axis is provided whenever images are treated as 3D,
        # i.e. when either do_3D is True or stitch_threshold>0 (2D+stitch)
        # and data has a depth dimension (ndim>=3).
        if (do_3D or stitch_threshold > 0.0) and data.ndim >= 3:
            z_axis = 0
        else:
            z_axis = None

        cfg = {
            "tic": tic,
            "data": data,
            "do_3D": do_3D,
            "stitch_threshold": stitch_threshold,
            "anisotropy": anisotropy,
            "flow3D_smooth": flow3D_smooth,
            "min_size": min_size,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
            "diameter": diameter,
            "niter": niter,
            "downscale": downscale,
            "normalize_params": normalize_params,
            "z_axis": z_axis,
            "time_index": seg_T,
        }
        return cfg


    def compute_segmentation(self, custom=False, model_name=None, load_model=True):
        """Synchronous segmentation (used e.g. by multi-timepoint loop)."""
        if self._segmentation_running:
            print("GUI_INFO: segmentation already running; please wait for it to finish")
            return

        self.progress.setValue(0)
        try:
            cfg = self._prepare_segmentation_inputs(
                custom=custom, model_name=model_name, load_model=load_model, time_index=None
            )
            data = cfg["data"]
            do_3D = cfg["do_3D"]
            stitch_threshold = cfg["stitch_threshold"]
            anisotropy = cfg["anisotropy"]
            flow3D_smooth = cfg["flow3D_smooth"]
            min_size = cfg["min_size"]
            flow_threshold = cfg["flow_threshold"]
            cellprob_threshold = cfg["cellprob_threshold"]
            diameter = cfg["diameter"]
            niter = cfg["niter"]
            downscale = cfg["downscale"]
            normalize_params = cfg["normalize_params"]
            z_axis = cfg["z_axis"]
            tic = cfg["tic"]

            try:
                masks, flows = self.model.eval(
                    data,
                    diameter=diameter,
                    cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold,
                    do_3D=do_3D,
                    niter=niter,
                    normalize=normalize_params,
                    stitch_threshold=stitch_threshold,
                    anisotropy=anisotropy,
                    flow3D_smooth=flow3D_smooth,
                    min_size=min_size,
                    channel_axis=-1,
                    progress=self.progress,
                    z_axis=z_axis,
                )[:2]
            except Exception as e:
                print("NET ERROR: %s" % e)
                self.progress.setValue(0)
                return

            self.progress.setValue(75)

            masks, flows_display, recompute_masks = _process_segmentation_outputs(
                masks,
                flows,
                load_3D=self.load_3D,
                do_3D=do_3D,
                stitch_threshold=stitch_threshold,
                downscale=downscale,
                Ly=self.Ly,
                Lx=self.Lx,
                Ly0=self.Ly0,
                Lx0=self.Lx0,
                Lyr=getattr(self, "Lyr", self.Ly),
                Lxr=getattr(self, "Lxr", self.Lx),
                NZ=self.NZ,
                restore=self.restore,
            )

            self.flows = flows_display
            self.logger.info(
                "%d cells found with model in %0.3f sec"
                % (len(np.unique(masks)[1:]), time.time() - tic)
            )
            self.progress.setValue(80)

            io._masks_to_gui(self, masks, outlines=None, preserve_labels=False)
            self.masksOn = True
            self.MCheckBox.setChecked(True)
            self.progress.setValue(100)
            if (
                self.restore != "filter"
                and self.restore is not None
                and self.autobtn.isChecked()
            ):
                self.compute_saturation()
            self.recompute_masks = recompute_masks
            self._current_seg_preserve_labels = False
        except Exception as e:
            print("ERROR: %s" % e)
        else:
            # autosave segmentation (including aggregated time-lapse seg_all) if enabled
            try:
                io._save_sets_with_check(self)
            except Exception as e:
                print(f"ERROR: could not autosave segmentation: {e}")

    def start_segmentation_async(self, custom=False, model_name=None, load_model=True, time_index=None):
        """Launch segmentation in a background thread so GUI stays responsive."""
        if self._segmentation_running:
            print("GUI_INFO: segmentation already running; please wait for it to finish")
            return

        self.progress.setValue(0)
        try:
            cfg = self._prepare_segmentation_inputs(
                custom=custom, model_name=model_name, load_model=load_model, time_index=time_index
            )
        except Exception as e:
            print(f"ERROR: {e}")
            self.progress.setValue(0)
            return

        data = cfg["data"]
        geom = {
            "load_3D": self.load_3D,
            "NZ": self.NZ,
            "Ly": self.Ly,
            "Lx": self.Lx,
            "Ly0": self.Ly0,
            "Lx0": self.Lx0,
            "Lyr": getattr(self, "Lyr", self.Ly),
            "Lxr": getattr(self, "Lxr", self.Lx),
            "restore": self.restore,
        }

        self._segmentation_running = True
        self._segmentation_thread = QtCore.QThread(self)
        self._segmentation_worker = SegmentationWorker(self.model, data, cfg, geom)
        self._segmentation_worker.moveToThread(self._segmentation_thread)
        self._segmentation_thread.started.connect(self._segmentation_worker.run)
        self._segmentation_worker.progress.connect(self.progress.setValue)
        self._segmentation_worker.error.connect(self._on_segmentation_error)
        self._segmentation_worker.finished.connect(self._on_segmentation_finished)
        self._segmentation_worker.finished.connect(self._segmentation_thread.quit)
        self._segmentation_worker.finished.connect(
            self._segmentation_worker.deleteLater
        )
        self._segmentation_thread.finished.connect(self._segmentation_thread.deleteLater)
        self._segmentation_thread.finished.connect(
            self._on_segmentation_thread_finished
        )
        self._segmentation_thread.start()

    def _on_segmentation_error(self, message):
        t_running = getattr(self, "_seg_all_current_t", None)
        if t_running is None:
            print(f"NET ERROR: {message}")
        else:
            print(f"NET ERROR at T={t_running}: {message}")
        self.progress.setValue(0)
        if self._segmentation_thread is not None:
            try:
                self._segmentation_thread.quit()
            except Exception:
                pass

    def _on_segmentation_finished(self, result):
        try:
            masks = result.get("masks", None)
            flows_display = result.get("flows", None)
            recompute_masks = result.get("recompute_masks", False)
            elapsed = result.get("elapsed", None)
            seg_time_index = result.get("time_index", None)
            preserve_labels = False
            if masks is None or flows_display is None:
                print("ERROR: segmentation worker returned no results")
                self.progress.setValue(0)
                return

            if (
                self._segmentation_all_running
                and seg_time_index is not None
                and getattr(self, "_seg_all_time_stitch_enabled", False)
            ):
                try:
                    masks = self._stitch_masks_over_time(masks, seg_time_index)
                    preserve_labels = True
                except Exception as e:
                    print(
                        "GUI_WARNING: time-stitching failed at "
                        f"T={seg_time_index}; keeping independent labels ({e})"
                    )

            self.flows = flows_display
            n_cells = len(np.unique(masks)[1:])
            if elapsed is not None:
                self.logger.info(
                    "%d cells found with model in %0.3f sec" % (n_cells, elapsed)
                )
            else:
                self.logger.info("%d cells found with model" % n_cells)

            self._cache_segmentation_result(
                masks, seg_time_index, preserve_labels=preserve_labels
            )
            self._current_seg_preserve_labels = bool(preserve_labels)
            if self._segmentation_all_running and seg_time_index is not None:
                try:
                    io._save_timepoint_mask_tiff(self, seg_time_index, masks=masks)
                except Exception as e:
                    print(
                        f"ERROR: could not save incremental timepoint mask TIFF for T={seg_time_index}: {e}"
                    )

            try:
                current_T = int(getattr(self, "currentT", 0))
            except Exception:
                current_T = None

            autosave_enabled = (
                hasattr(self, "disableAutosave")
                and not self.disableAutosave.isChecked()
            )
            should_switch_view = (
                seg_time_index is not None
                and (self._segmentation_all_running or autosave_enabled)
                and current_T != seg_time_index
            )

            if should_switch_view:
                self.set_time_index(seg_time_index)
            elif seg_time_index is None or current_T == seg_time_index:
                io._masks_to_gui(
                    self, masks, outlines=None, preserve_labels=preserve_labels
                )
                self.masksOn = True
                if hasattr(self, "MCheckBox"):
                    self.MCheckBox.setChecked(True)

            if self._segmentation_all_running and seg_time_index is not None:
                done_n = int(getattr(self, "_seg_all_total", 0)) - len(
                    getattr(self, "_seg_all_queue", [])
                )
                total_n = int(getattr(self, "_seg_all_total", 0))
                print(f"GUI_INFO: finished segmentation T={seg_time_index} [{done_n}/{total_n}]")

            self.progress.setValue(100)
            self.recompute_masks = recompute_masks
            if (
                self.restore != "filter"
                and self.restore is not None
                and self.autobtn.isChecked()
            ):
                self.compute_saturation()
        except Exception as e:
            print(f"ERROR: {e}")
            self.progress.setValue(0)
        else:
            # autosave segmentation (including aggregated time-lapse seg) if enabled
            try:
                io._save_sets_with_check(self)
            except Exception as e:
                print(f"ERROR: could not autosave segmentation: {e}")

    def _on_segmentation_thread_finished(self):
        self._segmentation_running = False
        # if running multi-timepoint segmentation, advance to next timepoint
        if self._segmentation_all_running:
            try:
                QtCore.QTimer.singleShot(0, self._segmentation_all_next)
            except Exception as e:
                print(f"ERROR: could not advance multi-timepoint segmentation: {e}")
                self._segmentation_all_running = False
                self._seg_all_current_t = None
                self._seg_all_total = 0
                self._reset_time_stitch_state()
