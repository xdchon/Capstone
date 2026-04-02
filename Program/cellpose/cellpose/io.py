"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import os, warnings, glob, shutil
from natsort import natsorted
import numpy as np
import cv2
import tifffile
import logging, pathlib, sys, copy
from tqdm import tqdm
from pathlib import Path
import re
from .version import version_str
from roifile import ImagejRoi, roiwrite

try:
    import zarr  # noqa: F401
    ZARR_AVAILABLE = True
except Exception:
    ZARR_AVAILABLE = False

try:
    from qtpy import QtGui, QtCore, Qt, QtWidgets
    from qtpy.QtWidgets import QMessageBox
    GUI = True
except:
    GUI = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False

try:
    import nd2
    ND2 = True
except:
    ND2 = False

try:
    import nrrd
    NRRD = True
except:
    NRRD = False

try:
    from google.cloud import storage
    SERVER_UPLOAD = True
except:
    SERVER_UPLOAD = False

io_logger = logging.getLogger(__name__)


class LazySldy:
    """Lazy reader for SlideBook .sldy/.sldyz using bundled SBReadFile toolkit."""

    def __init__(self, path, capture_index=0, position_index=0):
        import importlib.util
        base_dir = pathlib.Path(__file__).resolve().parents[2]
        sb_dir = base_dir.joinpath("SBReadFile22-Python-main")
        if sb_dir.exists() and str(sb_dir) not in sys.path:
            sys.path.insert(0, str(sb_dir))
        spec = importlib.util.find_spec("SBReadFile")
        if spec is None:
            raise ImportError("SBReadFile module not found; ensure SBReadFile22-Python-main is beside cellpose")
        SBReadFile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(SBReadFile)
        self.reader = SBReadFile.SBReadFile()
        ok = self.reader.Open(path, All=True)
        if not ok:
            raise ValueError(f"failed to open SlideBook file: {path}")
        self.capture = capture_index
        self.position = position_index
        self.nt = self.reader.GetNumTimepoints(self.capture)
        self.nz = self.reader.GetNumZPlanes(self.capture)
        self.nc = self.reader.GetNumChannels(self.capture)
        self.ny = self.reader.GetNumYRows(self.capture)
        self.nx = self.reader.GetNumXColumns(self.capture)
        self.channel_index = 0
        if self.nc > 1:
            io_logger.info(
                "SlideBook loader using channel 0 only for 4D TZYX stack; "
                f"{self.nc} channels available"
            )
        self.axes = "TZYX"
        self.shape = (self.nt, self.nz, self.ny, self.nx)
        self.dtype = np.uint16

    def get_plane(self, t_index, z_index):
        return self.reader.ReadImagePlaneBuf(
            self.capture,
            self.position,
            t_index,
            z_index,
            self.channel_index,
            True,
        )

    def get_time_stack(self, t_index):
        """Return full Z stack for a given time index as (Z, Y, X)."""
        planes = []
        for z in range(self.nz):
            planes.append(self.get_plane(t_index, z))
        return np.stack(planes, axis=0)


class LazyTiff5D:
    """Lazy reader for large TCZYX TIFF/BTF using tifffile+zarr virtual stack."""
    def __init__(self, store, axes, shape=None):
        # if aszarr returned a store (ZarrTiffStore), open as zarr array
        if hasattr(store, "__getitem__"):
            self.store = store
        else:
            if not ZARR_AVAILABLE:
                raise ValueError("zarr not available to open aszarr store")
            self.store = zarr.open(store, mode="r")
        self.axes = axes.upper()
        self.shape = shape if shape is not None else getattr(self.store, "shape", None)
        self.dtype = getattr(self.store, "dtype", None)
        if self.shape is None:
            raise ValueError("LazyTiff5D requires shape information for the stack")

    def get_time_stack(self, t_index):
        """Return a full Z stack for a given time index as array (Z, Y, X, C)."""
        axes = self.axes
        if "T" not in axes or "Z" not in axes:
            raise ValueError("LazyTiff5D requires both T and Z axes")

        slicer = []
        for ax in axes:
            if ax == "T":
                slicer.append(t_index)
            else:
                slicer.append(slice(None))
        arr = np.asarray(self.store[tuple(slicer)])

        # now arr has axes without T; build transpose order to Z,Y,X,C
        axes_after = [a for a in axes if a != "T"]
        axis_map = {a: i for i, a in enumerate(axes_after)}
        z_axis = axis_map.get("Z")
        y_axis = axis_map.get("Y")
        x_axis = axis_map.get("X")
        c_axis = axis_map.get("C")

        transpose_order = [z_axis, y_axis, x_axis]
        if c_axis is None:
            arr = arr[..., np.newaxis]
            c_axis = arr.ndim - 1
        transpose_order.append(c_axis)

        arr = np.transpose(arr, transpose_order)

        # ensure 3 channels
        num_channels = arr.shape[-1]
        if num_channels < 3:
            x_out = np.zeros((*arr.shape[:-1], 3), dtype=arr.dtype)
            x_out[..., :num_channels] = arr
            arr = x_out
        elif num_channels > 3:
            arr = arr[..., :3]

        return arr

    def get_plane(self, t_index, z_index):
        """Return single plane (Y, X, C)."""
        axes = self.axes
        slicer = []
        for ax in axes:
            if ax == "T":
                slicer.append(t_index)
            elif ax == "Z":
                slicer.append(z_index)
            else:
                slicer.append(slice(None))
        arr = np.asarray(self.store[tuple(slicer)])
        axes_after = [a for a in axes if a not in ("T", "Z")]
        axis_map = {a: i for i, a in enumerate(axes_after)}
        y_axis = axis_map.get("Y")
        x_axis = axis_map.get("X")
        c_axis = axis_map.get("C")
        transpose_order = [y_axis, x_axis]
        if c_axis is None:
            arr = arr[..., np.newaxis]
            c_axis = arr.ndim - 1
        transpose_order.append(c_axis)
        arr = np.transpose(arr, transpose_order)
        num_channels = arr.shape[-1]
        if num_channels < 3:
            x_out = np.zeros((*arr.shape[:-1], 3), dtype=arr.dtype)
            x_out[..., :num_channels] = arr
            arr = x_out
        elif num_channels > 3:
            arr = arr[..., :3]
        return arr


class LazyTiff5DMemmap:
    """Lazy reader backed by tifffile memmap for TCZYX."""
    def __init__(self, arr, axes):
        self.arr = arr
        self.axes = axes.upper()
        self.shape = arr.shape
        self.dtype = arr.dtype

    def get_time_stack(self, t_index):
        axes = self.axes
        if "T" not in axes or "Z" not in axes:
            raise ValueError("LazyTiff5DMemmap requires both T and Z axes")
        slicer = []
        for ax in axes:
            if ax == "T":
                slicer.append(t_index)
            else:
                slicer.append(slice(None))
        arr = np.asarray(self.arr[tuple(slicer)])

        axes_after = [a for a in axes if a != "T"]
        axis_map = {a: i for i, a in enumerate(axes_after)}
        z_axis = axis_map.get("Z")
        y_axis = axis_map.get("Y")
        x_axis = axis_map.get("X")
        c_axis = axis_map.get("C")

        transpose_order = [z_axis, y_axis, x_axis]
        if c_axis is None:
            arr = arr[..., np.newaxis]
            c_axis = arr.ndim - 1
        transpose_order.append(c_axis)

        arr = np.transpose(arr, transpose_order)

        num_channels = arr.shape[-1]
        if num_channels < 3:
            x_out = np.zeros((*arr.shape[:-1], 3), dtype=arr.dtype)
            x_out[..., :num_channels] = arr
            arr = x_out
        elif num_channels > 3:
            arr = arr[..., :3]

        return arr

    def get_plane(self, t_index, z_index):
        axes = self.axes
        slicer = []
        for ax in axes:
            if ax == "T":
                slicer.append(t_index)
            elif ax == "Z":
                slicer.append(z_index)
            else:
                slicer.append(slice(None))
        arr = np.asarray(self.arr[tuple(slicer)])
        axes_after = [a for a in axes if a not in ("T", "Z")]
        axis_map = {a: i for i, a in enumerate(axes_after)}
        y_axis = axis_map.get("Y")
        x_axis = axis_map.get("X")
        c_axis = axis_map.get("C")
        transpose_order = [y_axis, x_axis]
        if c_axis is None:
            arr = arr[..., np.newaxis]
            c_axis = arr.ndim - 1
        transpose_order.append(c_axis)
        arr = np.transpose(arr, transpose_order)
        num_channels = arr.shape[-1]
        if num_channels < 3:
            x_out = np.zeros((*arr.shape[:-1], 3), dtype=arr.dtype)
            x_out[..., :num_channels] = arr
            arr = x_out
        elif num_channels > 3:
            arr = arr[..., :3]
        return arr

def logger_setup(cp_path=".cellpose", logfile_name="run.log", stdout_file_replacement=None):
    cp_dir = pathlib.Path.home().joinpath(cp_path)
    cp_dir.mkdir(exist_ok=True)
    log_file = cp_dir.joinpath(logfile_name)
    try:
        log_file.unlink()
    except:
        print('creating new log file')
    handlers = [logging.FileHandler(log_file),]
    if stdout_file_replacement is not None:
        handlers.append(logging.FileHandler(stdout_file_replacement))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=handlers,
                    force=True
    )
    logger = logging.getLogger(__name__)
    logger.info(f"WRITING LOG OUTPUT TO {log_file}")
    logger.info(version_str)

    return logger, log_file


from . import utils, plot, transforms

# helper function to check for a path; if it doesn't exist, make it
def check_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def outlines_to_text(base, outlines):
    with open(base + "_cp_outlines.txt", "w") as f:
        for o in outlines:
            xy = list(o.flatten())
            xy_str = ",".join(map(str, xy))
            f.write(xy_str)
            f.write("\n")


def load_dax(filename):
    ### modified from ZhuangLab github:
    ### https://github.com/ZhuangLab/storm-analysis/blob/71ae493cbd17ddb97938d0ae2032d97a0eaa76b2/storm_analysis/sa_library/datareader.py#L156

    inf_filename = os.path.splitext(filename)[0] + ".inf"
    if not os.path.exists(inf_filename):
        io_logger.critical(
            f"ERROR: no inf file found for dax file {filename}, cannot load dax without it"
        )
        return None

    ### get metadata
    image_height, image_width = None, None
    # extract the movie information from the associated inf file
    size_re = re.compile(r"frame dimensions = ([\d]+) x ([\d]+)")
    length_re = re.compile(r"number of frames = ([\d]+)")
    endian_re = re.compile(r" (big|little) endian")

    with open(inf_filename, "r") as inf_file:
        lines = inf_file.read().split("\n")
        for line in lines:
            m = size_re.match(line)
            if m:
                image_height = int(m.group(2))
                image_width = int(m.group(1))
            m = length_re.match(line)
            if m:
                number_frames = int(m.group(1))
            m = endian_re.search(line)
            if m:
                if m.group(1) == "big":
                    bigendian = 1
                else:
                    bigendian = 0
    # set defaults, warn the user that they couldn"t be determined from the inf file.
    if not image_height:
        io_logger.warning("could not determine dax image size, assuming 256x256")
        image_height = 256
        image_width = 256

    ### load image
    img = np.memmap(filename, dtype="uint16",
                    shape=(number_frames, image_height, image_width))
    if bigendian:
        img = img.byteswap()
    img = np.array(img)

    return img


def imread(filename):
    """
    Read in an image file with tif or image file type supported by cv2.

    Args:
        filename (str): The path to the image file.

    Returns:
        numpy.ndarray: The image data as a NumPy array.

    Raises:
        None

    Raises an error if the image file format is not supported.

    Examples:
        >>> img = imread("image.tif")
    """
    # ensure that extension check is not case sensitive
    ext = os.path.splitext(filename)[-1].lower()
    if ext in (".sldy", ".sldyz"):
        # handled in imread_3D; return lazy reader directly
        try:
            return LazySldy(filename)
        except Exception as e:
            io_logger.critical(f"ERROR: could not read SlideBook file, {e}")
            return None
    if ext in (".tif", ".tiff", ".flex", ".btf"):
        with tifffile.TiffFile(filename) as tif:
            ltif = len(tif.pages)
            try:
                full_shape = tif.shaped_metadata[0]["shape"]
            except:
                try:
                    page = tif.series[0][0]
                    full_shape = tif.series[0].shape
                except:
                    ltif = 0
            if ltif < 10:
                img = tif.asarray()
            else:
                page = tif.series[0][0]
                shape, dtype = page.shape, page.dtype
                ltif = int(np.prod(full_shape) / np.prod(shape))
                io_logger.info(f"reading tiff with {ltif} planes")
                est_bytes = np.prod(full_shape) * dtype.itemsize
                use_memmap = est_bytes > 6 * 1024**3
                if use_memmap:
                    io_logger.info(f"large tiff detected (~{est_bytes/1024**3:.2f} GB), using memory map")
                    try:
                        img = tifffile.memmap(filename, series=0)
                    except Exception as e:
                        io_logger.warning(f"memmap load failed ({e}), falling back to sequential read")
                        use_memmap = False
                if not use_memmap:
                    img = np.zeros((ltif, *shape), dtype=dtype)
                    for i, page in enumerate(tqdm(tif.series[0])):
                        img[i] = page.asarray()
                    img = img.reshape(full_shape)
        return img
    elif ext == ".dax":
        img = load_dax(filename)
        return img
    elif ext == ".nd2":
        if not ND2:
            io_logger.critical("ERROR: need to 'pip install nd2' to load in .nd2 file")
            return None
    elif ext == ".nrrd":
        if not NRRD:
            io_logger.critical(
                "ERROR: need to 'pip install pynrrd' to load in .nrrd file")
            return None
        else:
            img, metadata = nrrd.read(filename)
            if img.ndim == 3:
                img = img.transpose(2, 0, 1)
            return img
    elif ext != ".npy":
        try:
            img = cv2.imread(filename, -1)  #cv2.LOAD_IMAGE_ANYDEPTH)
            if img.ndim > 2:
                img = img[..., [2, 1, 0]]
            return img
        except Exception as e:
            io_logger.critical("ERROR: could not read file, %s" % e)
            return None
    else:
        try:
            dat = np.load(filename, allow_pickle=True).item()
            masks = dat["masks"]
            return masks
        except Exception as e:
            io_logger.critical("ERROR: could not read masks from file, %s" % e)
            return None


def imread_2D(img_file):
    """
    Read in a 2D image file and convert it to a 3-channel image. Attempts to do this for multi-channel and grayscale images.
    If the image has more than 3 channels, only the first 3 channels are kept.
    
    Args:
        img_file (str): The path to the image file.

    Returns:
        img_out (numpy.ndarray): The 3-channel image data as a NumPy array.
    """
    img = imread(img_file)
    if hasattr(img, "get_time_stack"):
        # Let the GUI initialize the lazy time-aware source instead of forcing
        # it through 2D array conversion.
        return img
    return transforms.convert_image(img, do_3D=False)


def imread_3D(img_file):
    """
    Read in a 3D image file and convert it to have a channel axis last automatically. Attempts to do this for multi-channel and grayscale images.

    If multichannel image, the channel axis is assumed to be the smallest dimension, and the z axis is the next smallest dimension. 
    Use `cellpose.io.imread()` to load the full image without selecting the z and channel axes. 
    
    Args:
        img_file (str): The path to the image file.

    Returns:
        img_out (numpy.ndarray): The image data as a NumPy array.
    """
    img = imread(img_file)
    if isinstance(img, LazySldy):
        return img

    dimension_lengths = list(img.shape)
    axes = None
    ext = os.path.splitext(img_file)[-1].lower()
    if ext in (".tif", ".tiff", ".btf", ".flex"):
        try:
            with tifffile.TiffFile(img_file) as tif:
                axes = tif.series[0].axes
                try:
                    expected_shape = tif.shaped_metadata[0].get("shape", None)
                except Exception:
                    expected_shape = None
                if expected_shape is None:
                    try:
                        expected_shape = tif.series[0].shape
                    except Exception:
                        expected_shape = None
        except Exception:
            axes = None
            expected_shape = None

    # initialize defaults
    channel_axis = None
    z_axis = None
    time_axis = None

    # use tifffile axes if available
    series_shape = expected_shape

    if series_shape is not None and img.size == np.prod(series_shape) and img.shape != series_shape:
        try:
            img = img.reshape(series_shape)
        except Exception:
            pass

    # if axes includes T and current ndim is lower than axes length, try reshaping to full axes shape
    if axes is not None and expected_shape is not None and len(expected_shape) == len(axes) and img.ndim != len(axes):
        if img.size == np.prod(expected_shape):
            try:
                img = img.reshape(expected_shape)
            except Exception as e:
                io_logger.warning(f"could not reshape to expected axes shape {expected_shape}: {e}")

    if axes is not None and len(axes) == img.ndim:
        axes = axes.upper()
        if "C" in axes:
            channel_axis = axes.index("C")
        if "Z" in axes:
            z_axis = axes.index("Z")
        if "T" in axes:
            time_axis = axes.index("T")

    # fallback guessing
    if z_axis is None or (img.ndim < 5 and channel_axis is None):
        if img.ndim == 3:
            channel_axis = None
            z_axis = np.argmin(dimension_lengths)
        elif img.ndim == 4:
            channel_axis = np.argmin(dimension_lengths)
            dimension_lengths[channel_axis] = max(dimension_lengths)
            z_axis = np.argmin(dimension_lengths)
        elif img.ndim == 5:
            # default TCZYX if not specified
            time_axis = 0 if time_axis is None else time_axis
            channel_axis = 1 if channel_axis is None else channel_axis
            z_axis = 2 if z_axis is None else z_axis
        else:
            raise ValueError(f'image shape error, 3D image must be 3, 4, or 5 dimensional. Number of dimensions: {img.ndim}')

    try:
        # prefer virtual stack for large TCZYX if possible
        est_bytes = img.size * img.dtype.itemsize
        if est_bytes > 6 * 1024**3 and time_axis is not None and z_axis is not None and axes is not None:
            if ZARR_AVAILABLE:
                try:
                    with tifffile.TiffFile(img_file) as tif:
                        store = tif.series[0].aszarr()
                        return LazyTiff5D(store, axes=axes, shape=tif.series[0].shape)
                except Exception as e:
                    io_logger.warning(f"lazy zarr reader failed ({e}), using memmap virtual stack")
            # fallback memmap virtual
            if isinstance(img, np.memmap):
                # reshape memmap view if we know expected shape
                if expected_shape is not None and img.size == np.prod(expected_shape):
                    try:
                        img = img.reshape(expected_shape)
                    except Exception:
                        pass
                return LazyTiff5DMemmap(img, axes=axes)
            else:
                try:
                    mem = tifffile.memmap(img_file, series=0)
                    if expected_shape is not None and mem.size == np.prod(expected_shape):
                        try:
                            mem = mem.reshape(expected_shape)
                        except Exception:
                            pass
                    return LazyTiff5DMemmap(mem, axes=axes)
                except Exception as e:
                    io_logger.warning(f"memmap virtual stack failed ({e}), loading into memory")
        return transforms.convert_image(img, channel_axis=channel_axis, z_axis=z_axis, time_axis=time_axis, do_3D=True)
    except Exception as e:
        io_logger.critical("ERROR: could not read file, %s" % e)
        io_logger.critical("ERROR: Guessed/parsed axes - z_axis: %s, channel_axis: %s, time_axis: %s" % (z_axis, channel_axis, time_axis))
        return None

def remove_model(filename, delete=False):
    """ remove model from .cellpose custom model list """
    filename = os.path.split(filename)[-1]
    from . import models
    model_strings = models.get_user_models()
    if len(model_strings) > 0:
        with open(models.MODEL_LIST_PATH, "w") as textfile:
            for fname in model_strings:
                textfile.write(fname + "\n")
    else:
        # write empty file
        textfile = open(models.MODEL_LIST_PATH, "w")
        textfile.close()
    print(f"{filename} removed from custom model list")
    if delete:
        os.remove(os.fspath(models.MODEL_DIR.joinpath(fname)))
        print("model deleted")


def add_model(filename):
    """ add model to .cellpose models folder to use with GUI or CLI """
    from . import models
    fname = os.path.split(filename)[-1]
    try:
        shutil.copyfile(filename, os.fspath(models.MODEL_DIR.joinpath(fname)))
    except shutil.SameFileError:
        pass
    print(f"{filename} copied to models folder {os.fspath(models.MODEL_DIR)}")
    if fname not in models.get_user_models():
        with open(models.MODEL_LIST_PATH, "a") as textfile:
            textfile.write(fname + "\n")


def imsave(filename, arr):
    """
    Saves an image array to a file.

    Args:
        filename (str): The name of the file to save the image to.
        arr (numpy.ndarray): The image array to be saved.

    Returns:
        None
    """
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".tif" or ext == ".tiff":
        tifffile.imwrite(filename, data=arr, compression="zlib")
    else:
        if len(arr.shape) > 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, arr)


def get_image_files(folder, mask_filter, imf=None, look_one_level_down=False):
    """
    Finds all images in a folder and its subfolders (if specified) with the given file extensions.

    Args:
        folder (str): The path to the folder to search for images.
        mask_filter (str): The filter for mask files.
        imf (str, optional): The additional filter for image files. Defaults to None.
        look_one_level_down (bool, optional): Whether to search for images in subfolders. Defaults to False.

    Returns:
        list: A list of image file paths.

    Raises:
        ValueError: If no files are found in the specified folder.
        ValueError: If no images are found in the specified folder with the supported file extensions.
        ValueError: If no images are found in the specified folder without the mask or flow file endings.
    """
    mask_filters = ["_cp_output", "_flows", "_flows_0", "_flows_1",
                    "_flows_2", "_cellprob", "_masks", mask_filter]
    image_names = []
    if imf is None:
        imf = ""

    folders = []
    if look_one_level_down:
        folders = natsorted(glob.glob(os.path.join(folder, "*/")))
    folders.append(folder)
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".btf", ".flex", ".dax", ".nd2", ".nrrd", ".sldy", ".sldyz"]
    l0 = 0
    al = 0
    for folder in folders:
        all_files = glob.glob(folder + "/*")
        al += len(all_files)
        for ext in exts:
            image_names.extend(glob.glob(folder + f"/*{imf}{ext}"))
            image_names.extend(glob.glob(folder + f"/*{imf}{ext.upper()}"))
        l0 += len(image_names)

    # return error if no files found
    if al == 0:
        raise ValueError("ERROR: no files in --dir folder ")
    elif l0 == 0:
        raise ValueError(
            "ERROR: no images in --dir folder with extensions .png, .jpg, .jpeg, .tif, .tiff, .flex"
        )

    image_names = natsorted(image_names)
    imn = []
    for im in image_names:
        imfile = os.path.splitext(im)[0]
        igood = all([(len(imfile) > len(mask_filter) and
                      imfile[-len(mask_filter):] != mask_filter) or
                     len(imfile) <= len(mask_filter) for mask_filter in mask_filters])
        if len(imf) > 0:
            igood &= imfile[-len(imf):] == imf
        if igood:
            imn.append(im)

    image_names = imn

    # remove duplicates
    image_names = [*set(image_names)]
    image_names = natsorted(image_names)

    if len(image_names) == 0:
        raise ValueError(
            "ERROR: no images in --dir folder without _masks or _flows or _cellprob ending")

    return image_names

def get_label_files(image_names, mask_filter, imf=None):
    """
    Get the label files corresponding to the given image names and mask filter.

    Args:
        image_names (list): List of image names.
        mask_filter (str): Mask filter to be applied.
        imf (str, optional): Image file extension. Defaults to None.

    Returns:
        tuple: A tuple containing the label file names and flow file names (if present).
    """
    nimg = len(image_names)
    label_names0 = [os.path.splitext(image_names[n])[0] for n in range(nimg)]

    if imf is not None and len(imf) > 0:
        label_names = [label_names0[n][:-len(imf)] for n in range(nimg)]
    else:
        label_names = label_names0

    # check for flows
    if os.path.exists(label_names0[0] + "_flows.tif"):
        flow_names = [label_names0[n] + "_flows.tif" for n in range(nimg)]
    else:
        flow_names = [label_names[n] + "_flows.tif" for n in range(nimg)]
    if not all([os.path.exists(flow) for flow in flow_names]):
        io_logger.info(
            "not all flows are present, running flow generation for all images")
        flow_names = None

    # check for masks
    if mask_filter == "_seg.npy":
        label_names = [label_names[n] + mask_filter for n in range(nimg)]
        return label_names, None

    if os.path.exists(label_names[0] + mask_filter + ".tif"):
        label_names = [label_names[n] + mask_filter + ".tif" for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + ".tiff"):
        label_names = [label_names[n] + mask_filter + ".tiff" for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + ".png"):
        label_names = [label_names[n] + mask_filter + ".png" for n in range(nimg)]
    # TODO, allow _seg.npy
    #elif os.path.exists(label_names[0] + "_seg.npy"):
    #    io_logger.info("labels found as _seg.npy files, converting to tif")
    else:
        if not flow_names:
            raise ValueError("labels not provided with correct --mask_filter")
        else:
            label_names = None
    if not all([os.path.exists(label) for label in label_names]):
        if not flow_names:
            raise ValueError(
                "labels not provided for all images in train and/or test set")
        else:
            label_names = None

    return label_names, flow_names


def load_images_labels(tdir, mask_filter="_masks", image_filter=None,
                       look_one_level_down=False):
    """
    Loads images and corresponding labels from a directory.

    Args:
        tdir (str): The directory path.
        mask_filter (str, optional): The filter for mask files. Defaults to "_masks".
        image_filter (str, optional): The filter for image files. Defaults to None.
        look_one_level_down (bool, optional): Whether to look for files one level down. Defaults to False.

    Returns:
        tuple: A tuple containing a list of images, a list of labels, and a list of image names.
    """
    image_names = get_image_files(tdir, mask_filter, image_filter, look_one_level_down)
    nimg = len(image_names)

    # training data
    label_names, flow_names = get_label_files(image_names, mask_filter,
                                              imf=image_filter)

    images = []
    labels = []
    k = 0
    for n in range(nimg):
        if (os.path.isfile(label_names[n]) or
            (flow_names is not None and os.path.isfile(flow_names[0]))):
            image = imread(image_names[n])
            if label_names is not None:
                label = imread(label_names[n])
            if flow_names is not None:
                flow = imread(flow_names[n])
                if flow.shape[0] < 4:
                    label = np.concatenate((label[np.newaxis, :, :], flow), axis=0)
                else:
                    label = flow
            images.append(image)
            labels.append(label)
            k += 1
    io_logger.info(f"{k} / {nimg} images in {tdir} folder have labels")
    return images, labels, image_names

def load_train_test_data(train_dir, test_dir=None, image_filter=None,
                         mask_filter="_masks", look_one_level_down=False):
    """
    Loads training and testing data for a Cellpose model.

    Args:
        train_dir (str): The directory path containing the training data.
        test_dir (str, optional): The directory path containing the testing data. Defaults to None.
        image_filter (str, optional): The filter for selecting image files. Defaults to None.
        mask_filter (str, optional): The filter for selecting mask files. Defaults to "_masks".
        look_one_level_down (bool, optional): Whether to look for data in subdirectories of train_dir and test_dir. Defaults to False.

    Returns:
        images, labels, image_names, test_images, test_labels, test_image_names

    """
    images, labels, image_names = load_images_labels(train_dir, mask_filter,
                                                     image_filter, look_one_level_down)
    # testing data
    test_images, test_labels, test_image_names = None, None, None
    if test_dir is not None:
        test_images, test_labels, test_image_names = load_images_labels(
            test_dir, mask_filter, image_filter, look_one_level_down)

    return images, labels, image_names, test_images, test_labels, test_image_names


def masks_flows_to_seg(images, masks, flows, file_names, 
                       channels=None,
                       imgs_restore=None, restore_type=None, ratio=1.):
    """Save output of model eval to be loaded in GUI.

    Can be list output (run on multiple images) or single output (run on single image).

    Saved to file_names[k]+"_seg.npy".

    Args:
        images (list): Images input into cellpose.
        masks (list): Masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels.
        flows (list): Flows output from Cellpose.eval.
        file_names (list, str): Names of files of images.
        diams (float array): Diameters used to run Cellpose. Defaults to 30. TODO: remove this
        channels (list, int, optional): Channels used to run Cellpose. Defaults to None.

    Returns:
        None
    """

    if channels is None:
        channels = [0, 0]

    if isinstance(masks, list):
        n_items = len(masks)
        if imgs_restore is None:
            imgs_restore = [None] * n_items
        # normalize file_names to a list
        if isinstance(file_names, str) or isinstance(file_names, os.PathLike):
            file_names_list = [file_names] * n_items
            single_base = True
        else:
            file_names_list = list(file_names)
            single_base = len(file_names_list) == 1
        # broadcast single image source (e.g. lazy 5D reader) across items
        if isinstance(images, (list, tuple)):
            images_list = images
        else:
            images_list = [images] * n_items

        for k, (image, mask, flow, file_name, img_restore) in enumerate(
            zip(images_list, masks, flows, file_names_list, imgs_restore)
        ):
            # if a single input name was provided, suffix with time index so each
            # timepoint gets its own *_seg.npy instead of overwriting
            if single_base:
                base, ext = os.path.splitext(os.fspath(file_name))
                file_name_k = base + f"_T{k:04d}" + ext
            else:
                file_name_k = os.fspath(file_name)

            channels_img = channels
            if channels_img is not None and len(channels) > 2:
                channels_img = channels[k]
            masks_flows_to_seg(
                image,
                mask,
                flow,
                file_name_k,
                channels=channels_img,
                imgs_restore=img_restore,
                restore_type=restore_type,
                ratio=ratio,
            )
        return

    if len(channels) == 1:
        channels = channels[0]

    flowi = []
    if flows[0].ndim == 3:
        Ly, Lx = masks.shape[-2:]
        flowi.append(
            cv2.resize(flows[0], (Lx, Ly), interpolation=cv2.INTER_NEAREST)[np.newaxis,
                                                                            ...])
    else:
        flowi.append(flows[0])

    if flows[0].ndim == 3:
        cellprob = (np.clip(transforms.normalize99(flows[2]), 0, 1) * 255).astype(
            np.uint8)
        cellprob = cv2.resize(cellprob, (Lx, Ly), interpolation=cv2.INTER_NEAREST)
        flowi.append(cellprob[np.newaxis, ...])
        flowi.append(np.zeros(flows[0].shape, dtype=np.uint8))
        flowi[-1] = flowi[-1][np.newaxis, ...]
    else:
        flowi.append(
            (np.clip(transforms.normalize99(flows[2]), 0, 1) * 255).astype(np.uint8))
        flowi.append((flows[1][0] / 10 * 127 + 127).astype(np.uint8))
    if len(flows) > 2:
        if len(flows) > 3:
            flowi.append(flows[3])
        else:
            flowi.append([])
        flowi.append(np.concatenate((flows[1], flows[2][np.newaxis, ...]), axis=0))
    outlines = masks * utils.masks_to_outlines(masks)
    base = os.path.splitext(file_names)[0]

    # Derive original image filename and optional time index from file_names
    # e.g. "..._T0003.ext" -> filename_meta="...ext", time_index=3
    time_index = None
    filename_meta = file_names
    try:
        name_str = os.fspath(file_names)
        base_noext, ext = os.path.splitext(name_str)
        m = re.search(r"_T(\d{4})$", base_noext)
        if m is not None:
            time_index = int(m.group(1))
            base_image = base_noext[: -len(m.group(0))]
            filename_meta = base_image + ext
        else:
            filename_meta = name_str
    except Exception:
        filename_meta = file_names

    dat = {
        "outlines":
            outlines.astype(np.uint16) if outlines.max() < 2**16 -
            1 else outlines.astype(np.uint32),
        "masks":
            masks.astype(np.uint16) if outlines.max() < 2**16 -
            1 else masks.astype(np.uint32),
        "chan_choose":
            channels,
        "ismanual":
            np.zeros(masks.max(), bool),
        "filename":
            filename_meta,
        "flows":
            flowi,
        "diameter":
            np.nan
    }
    if restore_type is not None and imgs_restore is not None:
        dat["restore"] = restore_type
        dat["ratio"] = ratio
        dat["img_restore"] = imgs_restore
    if time_index is not None:
        dat["time_index"] = time_index

    np.save(base + "_seg.npy", dat)

def save_to_png(images, masks, flows, file_names):
    """ deprecated (runs io.save_masks with png=True)

        does not work for 3D images

    """
    save_masks(images, masks, flows, file_names, png=True)


def save_rois(masks, file_name, multiprocessing=None, prefix='', pad=False):
    """ save masks to .roi files in .zip archive for ImageJ/Fiji
    When opened in ImageJ, the ROIs will be named [prefix][0000]n where n is 1,2,... corresponding to the masks label

    Args:
        masks (np.ndarray): masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels
        file_name (str): name to save the .zip file to
        multiprocessing (bool, optional): Flag to enable multiprocessing. Defaults to None (disabled).
        prefix (str, optional): prefix to add at the beginning of the ROI labels in ImageJ. Defaults to no prefix
        pad (bool, optional): Whether to pad the numerical part of the label with zeros so that all labels have the same length

    Returns:
        None
    """
    outlines = utils.outlines_list(masks, multiprocessing=multiprocessing)
    
    n_digits = int(np.floor(np.log10(masks.max()))+1) if pad else 0
    fmt = f'{{prefix}}{{id:0{n_digits}d}}'
    rois = []
    for n,outline in zip(np.unique(masks)[1:], outlines):
        if len(outline) > 0:
            rois.append(ImagejRoi.frompoints(outline, name=fmt.format(prefix=prefix, id=n)))

    if len(outlines) != len(rois):
        print(f"empty outlines found, saving {len(rois)} ImageJ ROIs to .zip archive.")

    file_name = os.path.splitext(file_name)[0] + '_rois.zip'
    roiwrite(file_name, rois, mode='w')


def save_masks(images, masks, flows, file_names, png=True, tif=False, channels=[0, 0],
               suffix="_cp_masks", save_flows=False, save_outlines=False, dir_above=False,
               in_folders=False, savedir=None, save_txt=False, save_mpl=False):
    """ Save masks + nicely plotted segmentation image to png and/or tiff.

    Can save masks, flows to different directories, if in_folders is True.

    If png, masks[k] for images[k] are saved to file_names[k]+"_cp_masks.png".

    If tif, masks[k] for images[k] are saved to file_names[k]+"_cp_masks.tif".

    If png and matplotlib installed, full segmentation figure is saved to file_names[k]+"_cp.png".

    Only tif option works for 3D data, and only tif option works for empty masks.

    Args:
        images (list): Images input into cellpose.
        masks (list): Masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels.
        flows (list): Flows output from Cellpose.eval.
        file_names (list, str): Names of files of images.
        png (bool, optional): Save masks to PNG. Defaults to True.
        tif (bool, optional): Save masks to TIF. Defaults to False.
        channels (list, int, optional): Channels used to run Cellpose. Defaults to [0,0].
        suffix (str, optional): Add name to saved masks. Defaults to "_cp_masks".
        save_flows (bool, optional): Save flows output from Cellpose.eval. Defaults to False.
        save_outlines (bool, optional): Save outlines of masks. Defaults to False.
        dir_above (bool, optional): Save masks/flows in directory above. Defaults to False.
        in_folders (bool, optional): Save masks/flows in separate folders. Defaults to False.
        savedir (str, optional): Absolute or relative path where images will be saved. If None, saves to image directory. Defaults to None.
        save_txt (bool, optional): Save masks as list of outlines for ImageJ. Defaults to False.
        save_mpl (bool, optional): If True, saves a matplotlib figure of the original image/segmentation/flows. Does not work for 3D.
                This takes a long time for large images. Defaults to False.

    Returns:
        None
    """

    if isinstance(masks, list):
        for image, mask, flow, file_name in zip(images, masks, flows, file_names):
            save_masks(image, mask, flow, file_name, png=png, tif=tif, suffix=suffix,
                       dir_above=dir_above, save_flows=save_flows,
                       save_outlines=save_outlines, savedir=savedir, save_txt=save_txt,
                       in_folders=in_folders, save_mpl=save_mpl)
        return

    if masks.ndim > 2 and not tif:
        raise ValueError("cannot save 3D outputs as PNG, use tif option instead")

    if masks.max() == 0:
        io_logger.warning("no masks found, will not save PNG or outlines")
        if not tif:
            return
        else:
            png = False
            save_outlines = False
            save_flows = False
            save_txt = False

    if savedir is None:
        if dir_above:
            savedir = Path(file_names).parent.parent#go up a level to save in its own folder
        else:
            savedir = Path(file_names).parent

    savedir = Path(savedir).resolve()
    check_dir(savedir)

    basename = os.path.splitext(os.path.basename(file_names))[0]
    if in_folders:
        maskdir = os.path.join(savedir, "masks")
        outlinedir = os.path.join(savedir, "outlines")
        txtdir = os.path.join(savedir, "txt_outlines")
        flowdir = os.path.join(savedir, "flows")
    else:
        maskdir = savedir
        outlinedir = savedir
        txtdir = savedir
        flowdir = savedir

    check_dir(maskdir)

    exts = []
    if masks.ndim > 2:
        png = False
        tif = True
    if png:
        if masks.max() < 2**16:
            masks = masks.astype(np.uint16)
            exts.append(".png")
        else:
            png = False
            tif = True
            io_logger.warning(
                "found more than 65535 masks in each image, cannot save PNG, saving as TIF"
            )
    if tif:
        exts.append(".tif")

    # save masks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ext in exts:
            imsave(os.path.join(maskdir, basename + suffix + ext), masks)

    if save_mpl and png and MATPLOTLIB and not min(images.shape) > 3:
        # Make and save original/segmentation/flows image

        img = images.copy()
        if img.ndim < 3:
            img = img[:, :, np.newaxis]
        elif img.shape[0] < 8:
            np.transpose(img, (1, 2, 0))

        fig = plt.figure(figsize=(12, 3))
        plot.show_segmentation(fig, img, masks, flows[0])
        fig.savefig(os.path.join(savedir, basename + "_cp_output" + suffix + ".png"),
                    dpi=300)
        plt.close(fig)

    # ImageJ txt outline files
    if masks.ndim < 3 and save_txt:
        check_dir(txtdir)
        outlines = utils.outlines_list(masks)
        outlines_to_text(os.path.join(txtdir, basename), outlines)

    # RGB outline images
    if masks.ndim < 3 and save_outlines:
        check_dir(outlinedir)
        outlines = utils.masks_to_outlines(masks)
        outX, outY = np.nonzero(outlines)
        img0 = transforms.normalize99(images)
        if img0.shape[0] < 4:
            img0 = np.transpose(img0, (1, 2, 0))
        if img0.shape[-1] < 3 or img0.ndim < 3:
            img0 = plot.image_to_rgb(img0, channels=channels)
        else:
            if img0.max() <= 50.0:
                img0 = np.uint8(np.clip(img0 * 255, 0, 1))
        imgout = img0.copy()
        imgout[outX, outY] = np.array([255, 0, 0])  #pure red
        imsave(os.path.join(outlinedir, basename + "_outlines" + suffix + ".png"),
               imgout)

    # save RGB flow picture
    if masks.ndim < 3 and save_flows:
        check_dir(flowdir)
        imsave(os.path.join(flowdir, basename + "_flows" + suffix + ".tif"),
               flows[0]
              )
        #save full flow data
        imsave(os.path.join(flowdir, basename + '_dP' + suffix + '.tif'), flows[1])
