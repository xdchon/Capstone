"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import os, gc, re
import numpy as np
import cv2
import fastremap
import tifffile

from ..io import imread, imread_2D, imread_3D, imsave, outlines_to_text, add_model, remove_model, save_rois
from ..models import normalize_default, MODEL_DIR, MODEL_LIST_PATH, get_user_models
from ..utils import masks_to_outlines, outlines_list

try:
    import qtpy
    from qtpy.QtWidgets import QFileDialog
    GUI = True
except:
    GUI = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False


def _init_model_list(parent):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    parent.model_list_path = MODEL_LIST_PATH
    parent.model_strings = get_user_models()


def _add_model(parent, filename=None, load_model=True):
    if filename is None:
        name = QFileDialog.getOpenFileName(parent, "Add model to GUI")
        filename = name[0]
    add_model(filename)
    fname = os.path.split(filename)[-1]
    parent.ModelChooseC.addItems([fname])
    parent.model_strings.append(fname)

    for ind, model_string in enumerate(parent.model_strings[:-1]):
        if model_string == fname:
            _remove_model(parent, ind=ind + 1, verbose=False)

    parent.ModelChooseC.setCurrentIndex(len(parent.model_strings))
    if load_model:
        parent.model_choose(custom=True)


def _remove_model(parent, ind=None, verbose=True):
    if ind is None:
        ind = parent.ModelChooseC.currentIndex()
    if ind > 0:
        ind -= 1
        parent.ModelChooseC.removeItem(ind + 1)
        del parent.model_strings[ind]
        # remove model from txt path
        modelstr = parent.ModelChooseC.currentText()
        remove_model(modelstr)
        if len(parent.model_strings) > 0:
            parent.ModelChooseC.setCurrentIndex(len(parent.model_strings))
        else:
            parent.ModelChooseC.setCurrentIndex(0)
    else:
        print("ERROR: no model selected to delete")


def _get_train_set(image_names):
    """ get training data and labels for images in current folder image_names"""
    train_data, train_labels, train_files = [], [], []
    restore = None
    normalize_params = normalize_default
    for image_name_full in image_names:
        image_name = os.path.splitext(image_name_full)[0]
        label_name = None
        if os.path.exists(image_name + "_seg.npy"):
            dat = np.load(image_name + "_seg.npy", allow_pickle=True).item()
            masks = dat["masks"].squeeze()
            if masks.ndim == 2:
                fastremap.renumber(masks, in_place=True)
                label_name = image_name + "_seg.npy"
            else:
                print(f"GUI_INFO: _seg.npy found for {image_name} but masks.ndim!=2")
            if "img_restore" in dat:
                data = dat["img_restore"].squeeze()
                restore = dat["restore"]
            else:
                data = imread(image_name_full)
            normalize_params = dat[
                "normalize_params"] if "normalize_params" in dat else normalize_default
        if label_name is not None:
            train_files.append(image_name_full)
            train_data.append(data)
            train_labels.append(masks)
    if restore:
        print(f"GUI_INFO: using {restore} images (dat['img_restore'])")
    return train_data, train_labels, train_files, restore, normalize_params


def _load_image(parent, filename=None, load_seg=True, load_3D=False):
    """ load image with filename; if None, open QFileDialog
    if image is grey change view to default to grey scale 
    """

    if parent.load_3D:
        load_3D = True

    if filename is None:
        name = QFileDialog.getOpenFileName(parent, "Load image")
        filename = name[0]
        if filename == "":
            return
    base, ext = os.path.splitext(filename)
    manual_file = base + "_seg.npy"
    load_mask = False
    if load_seg:
        # For TIFF-like stacks, always prefer *_seg.npy if it exists,
        # regardless of the autoloadMasks setting.
        if os.path.isfile(manual_file) and ext.lower() in (".tif", ".tiff", ".btf", ".flex"):
            if filename is not None:
                image = imread_3D(filename) if load_3D or parent.load_3D else imread_2D(filename)
            else:
                image = None
            # if seg load fails, continue to load image instead of aborting
            if _load_seg(parent, manual_file, image=image, image_file=filename, load_3D=load_3D):
                return

        # Original behavior for other formats (or when *_seg.npy is missing)
        if os.path.isfile(manual_file) and not parent.autoloadMasks.isChecked():
            if filename is not None:
                image = (imread_2D(filename) if not load_3D else 
                         imread_3D(filename))
            else:
                image = None
            # if seg load fails, continue to load image instead of aborting
            if _load_seg(parent, manual_file, image=image, image_file=filename,
                         load_3D=load_3D):
                return
        elif parent.autoloadMasks.isChecked():
            mask_file = base + "_masks" + ext
            mask_file = base + "_masks.tif" if not os.path.isfile(mask_file) else mask_file
            load_mask = True if os.path.isfile(mask_file) else False
    parent.show_loading("Loading image...")
    try:
        print(f"GUI_INFO: loading image: {filename}")
        if not load_3D:
            image = imread_2D(filename)
        else:
            image = imread_3D(filename)
        parent.loaded = True
    except Exception as e:
        print("ERROR: images not compatible")
        print(f"ERROR: {e}")
        parent.loaded = False
        parent.hide_loading()
        return

    if image is None:
        print("ERROR: images not compatible")
        parent.loaded = False
        parent.hide_loading()
        return

    if parent.loaded:
        parent.reset()
        parent.filename = filename
        filename = os.path.split(parent.filename)[-1]
        _initialize_images(parent, image, load_3D=load_3D)
        parent.loaded = True
        parent.enable_buttons()
        if load_mask:
            _load_masks(parent, filename=mask_file)
        else:
            # For time-series data (e.g. .sldy / TCZYX), try to auto-load segmentation
            # for the current timepoint from a single aggregated _seg.npy file if present;
            # otherwise, fall back to per-timepoint *_T####_seg.npy files.
            try:
                if getattr(parent, "has_time", False):
                    base, _ = os.path.splitext(parent.filename)
                    t_index = int(getattr(parent, "currentT", 0))
                    seg_loaded = False

                    # first try aggregated time-lapse seg file
                    seg_all_path = base + "_seg.npy"
                    if os.path.isfile(seg_all_path):
                        try:
                            all_dat = np.load(seg_all_path, allow_pickle=True).item()
                            per_time = all_dat.get("per_time", {})
                            if isinstance(per_time, dict):
                                # Merge disk cache with any in-memory cache to avoid dropping
                                # newly computed, unsaved timepoints.
                                merged_per_time = {}
                                if isinstance(per_time, dict):
                                    merged_per_time.update(per_time)
                                mem_per_time = getattr(parent, "seg_time_data", None)
                                if isinstance(mem_per_time, dict):
                                    merged_per_time.update(mem_per_time)
                                parent.seg_time_data = merged_per_time

                                dat_t = merged_per_time.get(t_index)
                                if dat_t is None and str(t_index) in merged_per_time:
                                    dat_t = merged_per_time[str(t_index)]
                                if isinstance(dat_t, dict):
                                    masks = dat_t.get("masks", None)
                                    outlines = dat_t.get("outlines", None)
                                    colors = dat_t.get("colors", None)
                                    if masks is not None:
                                        _masks_to_gui(parent, masks, outlines=outlines, colors=colors)
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
                                        # bring to TZYX
                                        order = [t_axis, z_axis, y_axis, x_axis]
                                        arr_tzyx = np.transpose(masks_all, order)
                                        if 0 <= t_index < arr_tzyx.shape[0]:
                                            masks_t = arr_tzyx[t_index]
                                            outlines_t = None
                                            if isinstance(outlines_all, np.ndarray) and outlines_all.shape == masks_all.shape:
                                                out_arr = np.transpose(outlines_all, order)
                                                outlines_t = out_arr[t_index]
                                            _masks_to_gui(parent, masks_t, outlines=outlines_t, colors=None)
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
                            if masks is not None:
                                _masks_to_gui(parent, masks, outlines=outlines, colors=colors)
                                seg_loaded = True

                    # final fallback: by-timepoint TIFF folder (e.g. *_cp_masks_by_time)
                    if not seg_loaded:
                        seg_loaded = _load_timepoint_mask_from_by_time(
                            parent, base, t_index
                        )
            except Exception as e:
                print(f"ERROR: could not auto-load seg on initial load: {e}")

    # check if gray and adjust viewer:
    if parent.loaded and not hasattr(image, "get_time_stack") and len(np.unique(image[..., 1:])) == 1:
        parent.color = 4
        parent.RGBDropDown.setCurrentIndex(4) # gray
        parent.update_plot()

    parent.hide_loading()

        
def _initialize_images(parent, image, load_3D=False, time_stack=None, currentT=None, use_lazy=False):
    """ format image for GUI

    assumes image is Z x W x H x C (or T x Z x W x H x C)

    """
    load_3D = parent.load_3D if load_3D is False else load_3D
    already_normalized = False

    def _axis_len(obj, ax_char):
        if hasattr(obj, "axes") and obj.axes is not None and ax_char in obj.axes:
            try:
                return obj.shape[obj.axes.index(ax_char)]
            except Exception:
                return None
        return None

    # handle virtual stack cache (load one T at a time, keep only current T to save RAM)
    keep_current_z = False
    if hasattr(image, "get_time_stack"):  # Lazy readers
        new_lazy_source = (not use_lazy) or not hasattr(parent, "lazy_data") or parent.lazy_data is None or parent.lazy_data is not image
        parent.lazy_data = image
        parent.time_stack = None
        if new_lazy_source or not hasattr(parent, "global_sat"):
            parent.global_sat = None
        nt_lazy = _axis_len(image, "T") or getattr(image, "shape", [1])[0]
        parent.NT = nt_lazy if nt_lazy is not None else 1
        parent.has_time = parent.NT > 1
        if currentT is not None:
            parent.currentT = currentT
        elif not hasattr(parent, "currentT"):
            parent.currentT = 0
        parent.currentT = max(0, min(parent.NT - 1, int(parent.currentT)))
        if new_lazy_source or not hasattr(parent, "time_cache") or parent.time_cache is None:
            parent.time_cache = {}
        if new_lazy_source or not hasattr(parent, "sat_cache") or parent.sat_cache is None:
            parent.sat_cache = {}

        nz_lazy = _axis_len(image, "Z")
        prev_NZ = parent.NZ if hasattr(parent, "NZ") else 1
        parent.NZ = nz_lazy if nz_lazy is not None else prev_NZ
        if not hasattr(parent, "currentZ"):
            parent.currentZ = 0
        if parent.NZ > 0:
            parent.currentZ = max(0, min(parent.NZ - 1, int(parent.currentZ)))

        cache_key = (parent.currentT, parent.currentZ)
        plane = parent.time_cache.get(cache_key) if parent.time_cache is not None else None
        if plane is None:
            plane = image.get_plane(parent.currentT, parent.currentZ)
            plane = plane.astype(np.float32)
            pmin, pmax = plane.min(), plane.max()
            if pmax > pmin + 1e-3:
                plane = (plane - pmin) / (pmax - pmin) * 255.
        # keep only the current plane in cache to minimize RAM
        parent.time_cache = {cache_key: plane}
        image = plane[np.newaxis, ...]
        already_normalized = True
        load_3D = True
        parent.load_3D = True
        keep_current_z = True
    else:
        parent.lazy_data = None
        parent.time_cache = None
        parent.sat_cache = {}
        parent.global_sat = None
        parent.time_stack = image if time_stack is None else time_stack
        parent.has_time = (parent.time_stack is not None and parent.time_stack.ndim == 5 and parent.time_stack.shape[0] > 1)
        parent.NT = parent.time_stack.shape[0] if parent.has_time else 1
        if currentT is not None:
            parent.currentT = currentT
        elif not hasattr(parent, "currentT") or time_stack is None:
            parent.currentT = 0
        if parent.has_time:
            image = parent.time_stack[parent.currentT]
            load_3D = True
            parent.load_3D = True
    parent.stack = image
    print(f"GUI_INFO: image shape: {image.shape}")
    parent.configure_timebar()
    if load_3D:
        if hasattr(parent, "lazy_data") and parent.lazy_data is not None:
            parent.NZ = parent.NZ if hasattr(parent, "NZ") and parent.NZ is not None else len(parent.stack)
        else:
            parent.NZ = len(parent.stack)
        parent.scroll.setMaximum(parent.NZ - 1)
    else:
        parent.NZ = 1
        parent.stack = parent.stack[np.newaxis, ...]

    if not already_normalized:
        img_min = image.min()
        img_max = image.max()
        parent.stack = parent.stack.astype(np.float32)
        parent.stack -= img_min
        if img_max > img_min + 1e-3:
            parent.stack /= (img_max - img_min)
        parent.stack *= 255
        if load_3D:
            print("GUI_INFO: converted to float and normalized values to 0.0->255.0")

    del image
    gc.collect()

    parent.imask = 0
    parent.Ly, parent.Lx = parent.stack.shape[-3:-1]
    parent.Ly0, parent.Lx0 = parent.stack.shape[-3:-1]
    parent.layerz = 255 * np.ones((parent.Ly, parent.Lx, 4), "uint8")
    if hasattr(parent, "stack_filtered"):
        parent.Lyr, parent.Lxr = parent.stack_filtered.shape[-3:-1]
    elif parent.restore and "upsample" in parent.restore:
        parent.Lyr, parent.Lxr = int(parent.Ly * parent.ratio), int(parent.Lx *
                                                                    parent.ratio)
    else:
        parent.Lyr, parent.Lxr = parent.Ly, parent.Lx
    parent.clear_all()

    # reset saturation to match new stack depth, and clear if switching time
    parent.saturation = [[[0, 255] for _ in range(parent.NZ)] for _ in range(3)]
    parent.currentZ = 0

    if not hasattr(parent, "stack_filtered") and parent.restore:
        print("GUI_INFO: no 'img_restore' found, applying current settings")
        parent.compute_restore()

    if parent.autobtn.isChecked():
        # skip saturation computation for lazy virtual stacks (static levels already set)
        if not hasattr(parent, "lazy_data") or parent.lazy_data is None:
            if parent.restore is None or parent.restore != "filter":
                print(
                    "GUI_INFO: normalization checked: computing saturation levels (and optionally filtered image)"
                )
                parent.compute_saturation()
    # elif len(parent.saturation) != parent.NZ:
    #     parent.saturation = []
    #     for r in range(3):
    #         parent.saturation.append([])
    #         for n in range(parent.NZ):
    #             parent.saturation[-1].append([0, 255])
    #         parent.sliders[r].setValue([0, 255])
    parent.compute_scale()
    parent.track_changes = []

    if parent.has_time:
        parent.timeScrollTop.blockSignals(True)
        parent.timeScrollTop.setMaximum(parent.NT - 1)
        parent.timeScrollTop.setValue(parent.currentT)
        parent.timeScrollTop.blockSignals(False)
        if hasattr(parent, "timeScrollBottom") and parent.timeScrollBottom is not None:
            parent.timeScrollBottom.blockSignals(True)
            parent.timeScrollBottom.setMaximum(parent.NT - 1)
            parent.timeScrollBottom.setValue(parent.currentT)
            parent.timeScrollBottom.blockSignals(False)
        parent.timePos.setText(str(parent.currentT))

    if load_3D:
        if not keep_current_z:
            parent.currentZ = min(int(np.floor(parent.NZ / 2)), parent.NZ - 1)
        parent.scroll.setMinimum(0)
        parent.scroll.setMaximum(max(0, parent.NZ - 1))
        parent.scroll.blockSignals(True)
        parent.scroll.setValue(parent.currentZ)
        parent.scroll.blockSignals(False)
        parent.zpos.setText(str(parent.currentZ))
    else:
        parent.currentZ = 0


def _load_seg(parent, filename=None, image=None, image_file=None, load_3D=False):
    """ load *_seg.npy with filename; if None, open QFileDialog """
    if filename is None:
        name = QFileDialog.getOpenFileName(parent, "Load labelled data", filter="*.npy")
        filename = name[0]
    try:
        raw = np.load(filename, allow_pickle=True)
        # Support both plain mask arrays and dict-style seg files (GUI/CLI)
        if isinstance(raw, np.ndarray) and raw.dtype != object and raw.ndim >= 2:
            # plain masks saved via np.save(masks)
            masks_arr = raw.squeeze()
            outlines_arr = masks_to_outlines(masks_arr)
            dat = {"masks": masks_arr, "outlines": outlines_arr}
        else:
            dat_main = raw.item()
            # aggregated time-lapse seg: top-level has "per_time" mapping
            if "per_time" in dat_main and isinstance(dat_main["per_time"], dict):
                per_time = dat_main["per_time"]
                # choose time index: prefer currentT if available
                time_index = None
                if getattr(parent, "has_time", False) and hasattr(parent, "currentT"):
                    try:
                        time_index = int(parent.currentT)
                    except Exception:
                        time_index = None
                # fall back to any stored time_index or first available key
                if time_index is None:
                    try:
                        time_index = int(dat_main.get("time_index", None))
                    except Exception:
                        time_index = None
                dat_t = None
                if time_index is not None:
                    dat_t = per_time.get(time_index)
                    if dat_t is None and str(time_index) in per_time:
                        dat_t = per_time[str(time_index)]
                if dat_t is None and len(per_time) > 0:
                    # fallback: first entry in mapping
                    first_key = next(iter(per_time.keys()))
                    dat_t = per_time[first_key]
                    try:
                        time_index = int(first_key)
                    except Exception:
                        pass
                if isinstance(dat_t, dict):
                    if "time_index" not in dat_t:
                        dat_t["time_index"] = time_index
                    dat = dat_t
                else:
                    # no usable per-time entry; fall back to main dict
                    dat = dat_main
            else:
                # OME-style aggregated masks: axes includes T, Z, Y, X and masks is a numpy array
                axes = dat_main.get("axes", None)
                masks_all = dat_main.get("masks", None)
                outlines_all = dat_main.get("outlines", None)
                is_ome_5d = (
                    isinstance(axes, str)
                    and isinstance(masks_all, np.ndarray)
                    and "T" in axes.upper()
                    and "Z" in axes.upper()
                    and "Y" in axes.upper()
                    and "X" in axes.upper()
                )
                if is_ome_5d:
                    axes_up = axes.upper()
                    t_axis = axes_up.index("T")
                    z_axis = axes_up.index("Z")
                    y_axis = axes_up.index("Y")
                    x_axis = axes_up.index("X")
                    order = [t_axis, z_axis, y_axis, x_axis]
                    try:
                        arr_tzyx = np.transpose(masks_all, order)
                    except Exception:
                        # if transpose fails, fall back to main dict behavior
                        dat = dat_main
                    else:
                        # choose time index: prefer currentT if available
                        if getattr(parent, "has_time", False) and hasattr(parent, "currentT"):
                            try:
                                time_index = int(parent.currentT)
                            except Exception:
                                time_index = 0
                        else:
                            time_index = 0
                        if time_index < 0 or time_index >= arr_tzyx.shape[0]:
                            time_index = 0
                        masks_t = arr_tzyx[time_index]
                        outlines_t = None
                        if isinstance(outlines_all, np.ndarray) and outlines_all.shape == masks_all.shape:
                            try:
                                out_tzyx = np.transpose(outlines_all, order)
                                outlines_t = out_tzyx[time_index]
                            except Exception:
                                outlines_t = None
                        # build a per-time-style dict for GUI consumption
                        dat = dict(dat_main)
                        dat["masks"] = masks_t
                        dat["outlines"] = (
                            outlines_t if outlines_t is not None else masks_to_outlines(masks_t)
                        )
                        dat["time_index"] = time_index
                else:
                    dat = dat_main
                    # ensure 'outlines' key exists for single-time seg files
                    _ = dat["outlines"]
        parent.loaded = True
    except Exception as e:
        # If loading an aggregated seg_all file fails (e.g. truncated / ran out of input),
        # try falling back to a per-timepoint *_T####_seg.npy file for the current time index.
        try:
            if (
                filename is not None
                and isinstance(filename, str)
                and filename.endswith("_seg.npy")
                and getattr(parent, "has_time", False)
                and hasattr(parent, "currentT")
            ):
                # avoid infinite recursion: do not fallback if this already looks like *_T####_seg.npy
                base_noext = filename[:-8]  # strip "_seg.npy"
                stem, _ext = os.path.splitext(base_noext)
                if not re.search(r"_T\d{4}$", stem):
                    try:
                        t_index = int(parent.currentT)
                    except Exception:
                        t_index = 0
                    seg_path = base_noext + f"_T{t_index:04d}_seg.npy"
                    if os.path.isfile(seg_path):
                        print(
                            f"GUI_INFO: seg_all file failed to load ({e}); "
                            f"falling back to per-time file {seg_path}"
                        )
                        return _load_seg(
                            parent,
                            filename=seg_path,
                            image=image,
                            image_file=image_file,
                            load_3D=load_3D,
                        )
        except Exception as ee:
            print(f"ERROR: fallback to per-time seg file failed: {ee}")

        parent.loaded = False
        if isinstance(e, EOFError):
            print("ERROR: seg_all file appears truncated or corrupted (EOF)")
        else:
            print("ERROR: not NPY or incompatible seg file")
        print(f"ERROR: {e}")
        return False

    parent.reset()
    if image is None:
        found_image = False
        if "filename" in dat:
            parent.filename = dat["filename"]
            # direct path saved in npy
            if os.path.isfile(parent.filename):
                found_image = True
            else:
                # try same name in directory of seg file
                imgname = os.path.split(parent.filename)[1]
                root = os.path.split(filename)[0]
                candidate = os.path.join(root, imgname)
                if os.path.isfile(candidate):
                    parent.filename = candidate
                    found_image = True
                else:
                    # handle time-indexed names like *_T0000 before extension
                    base, ext = os.path.splitext(candidate)
                    parts = base.rsplit("_T", 1)
                    if len(parts) == 2 and len(parts[1]) == 4 and parts[1].isdigit():
                        base_orig = parts[0]
                        candidate2 = base_orig + ext
                        if os.path.isfile(candidate2):
                            parent.filename = candidate2
                            found_image = True
        if found_image:
            try:
                print(parent.filename)
                image = (imread_2D(parent.filename) if not load_3D else 
                         imread_3D(parent.filename))
            except:
                parent.loaded = False
                found_image = False
                print("ERROR: cannot find image file, loading from npy")
        if not found_image:
            parent.filename = filename[:-8]
            print(parent.filename)
            if "img" in dat:
                image = dat["img"]
            else:
                print("ERROR: no image file found and no image in npy")
                return
    else:
        parent.filename = image_file

    parent.restore = None
    parent.ratio = 1.

    if "normalize_params" in dat:
        parent.set_normalize_params(dat["normalize_params"])

    currentT = dat.get("time_index", None)
    _initialize_images(parent, image, load_3D=load_3D, currentT=currentT)
    print(parent.stack.shape)

    if "outlines" in dat:
        if isinstance(dat["outlines"], list):
            # old way of saving files
            dat["outlines"] = dat["outlines"][::-1]
            for k, outline in enumerate(dat["outlines"]):
                if "colors" in dat:
                    color = dat["colors"][k]
                else:
                    col_rand = np.random.randint(1000)
                    color = parent.colormap[col_rand, :3]
                median = parent.add_mask(points=outline, color=color)
                if median is not None:
                    parent.cellcolors = np.append(parent.cellcolors,
                                                  color[np.newaxis, :], axis=0)
                    parent.ncells += 1
        else:
            if dat["masks"].min() == -1:
                dat["masks"] += 1
                dat["outlines"] += 1
            parent.ncells.set(dat["masks"].max())
            if "colors" in dat and len(dat["colors"]) == dat["masks"].max():
                colors = dat["colors"]
            else:
                colors = parent.colormap[:parent.ncells.get(), :3]

            _masks_to_gui(parent, dat["masks"], outlines=dat["outlines"], colors=colors)

            parent.draw_layer()

        if "manual_changes" in dat:
            parent.track_changes = dat["manual_changes"]
            print("GUI_INFO: loaded in previous changes")
        if "zdraw" in dat:
            parent.zdraw = dat["zdraw"]
        else:
            parent.zdraw = [None for n in range(parent.ncells.get())]
        parent.loaded = True
    else:
        parent.clear_all()

    parent.ismanual = np.zeros(parent.ncells.get(), bool)
    if "ismanual" in dat:
        if len(dat["ismanual"]) == parent.ncells:
            parent.ismanual = dat["ismanual"]

    if "current_channel" in dat:
        parent.color = (dat["current_channel"] + 2) % 5
        parent.RGBDropDown.setCurrentIndex(parent.color)

    if "flows" in dat:
        parent.flows = dat["flows"]
        try:
            if parent.flows[0].shape[-3] != dat["masks"].shape[-2]:
                Ly, Lx = dat["masks"].shape[-2:]
                for i in range(len(parent.flows)):
                    parent.flows[i] = cv2.resize(
                        parent.flows[i].squeeze(), (Lx, Ly),
                        interpolation=cv2.INTER_NEAREST)[np.newaxis, ...]
            if parent.NZ == 1:
                parent.recompute_masks = True
            else:
                parent.recompute_masks = False

        except:
            try:
                if len(parent.flows[0]) > 0:
                    parent.flows = parent.flows[0]
            except:
                parent.flows = [[], [], [], [], [[]]]
            parent.recompute_masks = False

    parent.enable_buttons()
    parent.update_layer()
    del dat
    gc.collect()
    return True


def _load_masks(parent, filename=None):
    """ load zeros-based masks (0=no cell, 1=cell 1, ...) """
    if filename is None:
        name = QFileDialog.getOpenFileName(parent, "Load masks (PNG or TIFF)")
        filename = name[0]
    print(f"GUI_INFO: loading masks: {filename}")
    masks = imread(filename)
    outlines = None
    if masks.ndim > 3:
        # Z x nchannels x Ly x Lx
        if masks.shape[-1] > 5:
            parent.flows = list(np.transpose(masks[:, :, :, 2:], (3, 0, 1, 2)))
            outlines = masks[..., 1]
            masks = masks[..., 0]
        else:
            parent.flows = list(np.transpose(masks[:, :, :, 1:], (3, 0, 1, 2)))
            masks = masks[..., 0]
    elif masks.ndim == 3:
        if masks.shape[-1] < 5:
            masks = masks[np.newaxis, :, :, 0]
    elif masks.ndim < 3:
        masks = masks[np.newaxis, :, :]
    # masks should be Z x Ly x Lx
    if masks.shape[0] != parent.NZ:
        print("ERROR: masks are not same depth (number of planes) as image stack")
        return

    _masks_to_gui(parent, masks, outlines)
    if parent.ncells > 0:
        parent.draw_layer()
        parent.toggle_mask_ops()
    del masks
    gc.collect()
    parent.update_layer()
    parent.update_plot()


def _masks_to_gui(parent, masks, outlines=None, colors=None):
    """ masks loaded into GUI """
    # get unique values
    shape = masks.shape
    if len(fastremap.unique(masks)) != masks.max() + 1:
        print("GUI_INFO: renumbering masks")
        fastremap.renumber(masks, in_place=True)
        outlines = None
        masks = masks.reshape(shape)
    if masks.ndim == 2:
        outlines = None
    masks = masks.astype(np.uint16) if masks.max() < 2**16 - 1 else masks.astype(
        np.uint32)

    # ensure masks match image stack layout: Z x Ly x Lx
    if masks.ndim == 3:
        Ly_img, Lx_img = parent.Ly0, parent.Lx0
        dims = list(masks.shape)
        # if axes are mis-ordered (e.g. Ly x Lx x Z), permute so that Y,X match image
        if dims[1] != Ly_img or dims[2] != Lx_img:
            y_candidates = [i for i, s in enumerate(dims) if s == Ly_img]
            x_candidates = [i for i, s in enumerate(dims) if s == Lx_img]
            if len(y_candidates) == 1 and len(x_candidates) == 1:
                z_axis = [i for i in range(3) if i not in (y_candidates[0], x_candidates[0])]
                if z_axis:
                    order = (z_axis[0], y_candidates[0], x_candidates[0])
                    if order != (0, 1, 2):
                        masks = np.transpose(masks, order)
        # special case: single-Z saved as Ly x Lx x 1
        if masks.shape[-1] == 1 and masks.shape[0] == Ly_img and masks.shape[1] == Lx_img:
            masks = masks[..., 0][np.newaxis, ...]

    if parent.restore and "upsample" in parent.restore:
        parent.cellpix_resize = masks.copy()
        parent.cellpix = parent.cellpix_resize.copy()
        parent.cellpix_orig = cv2.resize(
            masks.squeeze(), (parent.Lx0, parent.Ly0),
            interpolation=cv2.INTER_NEAREST)[np.newaxis, :, :]
        parent.resize = True
    else:
        parent.cellpix = masks
    if parent.cellpix.ndim == 2:
        parent.cellpix = parent.cellpix[np.newaxis, :, :]
        if parent.restore and "upsample" in parent.restore:
            if parent.cellpix_resize.ndim == 2:
                parent.cellpix_resize = parent.cellpix_resize[np.newaxis, :, :]
            if parent.cellpix_orig.ndim == 2:
                parent.cellpix_orig = parent.cellpix_orig[np.newaxis, :, :]

    print(f"GUI_INFO: {masks.max()} masks found")

    # get outlines
    if outlines is None:  # parent.outlinesOn
        parent.outpix = np.zeros_like(parent.cellpix)
        if parent.restore and "upsample" in parent.restore:
            parent.outpix_orig = np.zeros_like(parent.cellpix_orig)
        nz_masks = parent.cellpix.shape[0]
        for z in range(nz_masks):
            outlines = masks_to_outlines(parent.cellpix[z])
            parent.outpix[z] = outlines * parent.cellpix[z]
            if parent.restore and "upsample" in parent.restore:
                outlines = masks_to_outlines(parent.cellpix_orig[z])
                parent.outpix_orig[z] = outlines * parent.cellpix_orig[z]
            if z % 50 == 0 and parent.NZ > 1:
                print("GUI_INFO: plane %d outlines processed" % z)
        if parent.restore and "upsample" in parent.restore:
            parent.outpix_resize = parent.outpix.copy()
    else:
        parent.outpix = outlines
        if parent.restore and "upsample" in parent.restore:
            parent.outpix_resize = parent.outpix.copy()
            parent.outpix_orig = np.zeros_like(parent.cellpix_orig)
            for z in range(parent.NZ):
                outlines = masks_to_outlines(parent.cellpix_orig[z])
                parent.outpix_orig[z] = outlines * parent.cellpix_orig[z]
                if z % 50 == 0 and parent.NZ > 1:
                    print("GUI_INFO: plane %d outlines processed" % z)

    if parent.outpix.ndim == 2:
        parent.outpix = parent.outpix[np.newaxis, :, :]
        if parent.restore and "upsample" in parent.restore:
            if parent.outpix_resize.ndim == 2:
                parent.outpix_resize = parent.outpix_resize[np.newaxis, :, :]
            if parent.outpix_orig.ndim == 2:
                parent.outpix_orig = parent.outpix_orig[np.newaxis, :, :]

    parent.ncells.set(parent.cellpix.max())
    colors = parent.colormap[:parent.ncells.get(), :3] if colors is None else colors
    print("GUI_INFO: creating cellcolors and drawing masks")
    parent.cellcolors = np.concatenate((np.array([[255, 255, 255]]), colors),
                                       axis=0).astype(np.uint8)
    # ensure mask overlay is visible when segmentation is present
    if parent.ncells > 0:
        if hasattr(parent, "MCheckBox"):
            parent.MCheckBox.setChecked(True)
        parent.masksOn = True
        parent.draw_layer()
        parent.toggle_mask_ops()
    parent.ismanual = np.zeros(parent.ncells.get(), bool)
    parent.zdraw = list(-1 * np.ones(parent.ncells.get(), np.int16))

    if hasattr(parent, "stack_filtered"):
        parent.ViewDropDown.setCurrentIndex(parent.ViewDropDown.count() - 1)
        print("set denoised/filtered view")
    else:
        parent.ViewDropDown.setCurrentIndex(0)


def _infer_time_index_from_filename(path):
    """Infer time index from common mask filename patterns."""
    name = os.path.basename(path)
    patterns = [
        r"timepoint_(\d+)",
        r"_T(\d+)(?:_|\.|$)",
        r"\bT(\d+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None


def _as_zyx_mask(arr):
    """Normalize mask array to ZYX layout."""
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"expected 2D or 3D mask array, got shape {arr.shape}")


def _default_ome_output_from_mask_folder(folder):
    folder = os.path.normpath(folder)
    parent_dir = os.path.dirname(folder)
    name = os.path.basename(folder)
    if name.endswith("_cp_masks_by_time"):
        stem = name[:-len("_by_time")]
    elif name.endswith("_by_time"):
        stem = name[:-len("_by_time")]
    else:
        stem = name + "_combined_masks"
    return os.path.join(parent_dir, stem + ".ome.tif")


def _collect_timepoint_mask_files(folder):
    """Collect timepoint mask TIFF files from a folder as {time_index: path}."""
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and f.lower().endswith((".tif", ".tiff"))
    ]
    files = sorted(files)
    if len(files) == 0:
        return {}

    time_map = {}
    next_seq = 0
    for p in files:
        t = _infer_time_index_from_filename(p)
        if t is None or t in time_map:
            while next_seq in time_map:
                next_seq += 1
            t = next_seq
            next_seq += 1
        else:
            t = int(t)
        time_map[t] = p
    return time_map


def _get_timepoint_mask_map_for_base(parent, base):
    """
    Return (folder, {time_index: tif_path}) for by-time mask folders near `base`.
    Uses a lightweight folder-mtime cache on the GUI parent.
    """
    candidates = [base + "_cp_masks_by_time", base + "_masks_by_time"]
    folder = next((d for d in candidates if os.path.isdir(d)), None)
    if folder is None:
        return None, {}

    try:
        folder_mtime = os.path.getmtime(folder)
    except Exception:
        folder_mtime = None

    cache_base = getattr(parent, "_time_mask_base", None)
    cache_folder = getattr(parent, "_time_mask_folder", None)
    cache_mtime = getattr(parent, "_time_mask_folder_mtime", None)
    cache_map = getattr(parent, "_time_mask_file_map", None)

    if (
        cache_base == base
        and cache_folder == folder
        and isinstance(cache_map, dict)
        and cache_mtime == folder_mtime
    ):
        return folder, cache_map

    time_map = _collect_timepoint_mask_files(folder)
    parent._time_mask_base = base
    parent._time_mask_folder = folder
    parent._time_mask_folder_mtime = folder_mtime
    parent._time_mask_file_map = time_map
    try:
        parent.available_mask_timepoints = sorted(time_map.keys())
    except Exception:
        pass
    if len(time_map) > 0:
        print(
            f"GUI_INFO: found {len(time_map)} timepoint mask TIFFs in {folder}"
        )
    return folder, time_map


def _load_timepoint_mask_from_by_time(parent, base, t_index):
    """
    Load masks for a specific timepoint from *_cp_masks_by_time folder if available.
    Returns True if loaded, else False.
    """
    folder, time_map = _get_timepoint_mask_map_for_base(parent, base)
    if folder is None or len(time_map) == 0:
        return False
    try:
        t = int(t_index)
    except Exception:
        return False
    if t not in time_map:
        return False

    mask_path = time_map[t]
    try:
        masks = _as_zyx_mask(tifffile.imread(mask_path))
        _masks_to_gui(parent, masks, outlines=None, colors=None)
        if not isinstance(getattr(parent, "seg_time_data", None), dict):
            parent.seg_time_data = {}
        parent.seg_time_data[t] = {
            "masks": masks.copy(),
            "outlines": None,
            "colors": None,
            "time_index": t,
            "axes": "ZYX",
        }
        print(
            f"GUI_INFO: loaded timepoint mask T={t} from by-time folder file {os.path.basename(mask_path)}"
        )
        return True
    except Exception as e:
        print(f"ERROR: could not load by-time mask for T={t} from {mask_path}: {e}")
        return False


def combine_timepoint_masks_folder_to_ome(
    input_folder, output_path=None, strict_shape=False, compression="zlib"
):
    """
    Combine per-timepoint labeled mask TIFFs into one labeled OME-TIFF (TZYX).

    This function is disk-backed: it assembles a temporary memmap volume, then
    writes one OME-TIFF, so peak RAM stays low even for large time-lapse stacks.
    """
    folder = os.path.abspath(os.path.expanduser(str(input_folder)))
    if not os.path.isdir(folder):
        raise ValueError(f"input folder does not exist: {folder}")

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and f.lower().endswith((".tif", ".tiff"))
    ]
    files = sorted(files)
    if len(files) == 0:
        raise ValueError(f"no TIFF files found in folder: {folder}")

    indexed = []
    for p in files:
        indexed.append((_infer_time_index_from_filename(p), p))

    has_explicit_t = any(t is not None for t, _ in indexed)
    if has_explicit_t:
        used = set()
        normalized = []
        next_t = 0
        for t, p in indexed:
            if t is None:
                while next_t in used:
                    next_t += 1
                t_use = next_t
                used.add(t_use)
                next_t += 1
            else:
                t_use = int(t)
                used.add(t_use)
            normalized.append((t_use, p))
    else:
        normalized = [(i, p) for i, (_t, p) in enumerate(indexed)]
    normalized.sort(key=lambda x: x[0])

    nt = max(t for t, _ in normalized) + 1
    target_shape = None
    max_label = 0
    for t, p in normalized:
        arr = _as_zyx_mask(tifffile.imread(p))
        if target_shape is None:
            target_shape = tuple(arr.shape)
        elif strict_shape and tuple(arr.shape) != target_shape:
            raise ValueError(
                f"shape mismatch for T={t}: expected {target_shape}, got {arr.shape} in {p}"
            )
        if arr.size > 0:
            try:
                max_label = max(max_label, int(arr.max()))
            except Exception:
                pass

    if target_shape is None:
        raise ValueError("no readable mask TIFFs found")

    out_dtype = np.uint16 if max_label < 2**16 else np.uint32
    if output_path is None or str(output_path).strip() == "":
        output_path = _default_ome_output_from_mask_folder(folder)
    output_path = os.path.abspath(os.path.expanduser(str(output_path)))
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_stack_path = output_path + ".stack.tmp"
    try:
        stack = np.memmap(
            tmp_stack_path,
            mode="w+",
            dtype=out_dtype,
            shape=(nt, target_shape[0], target_shape[1], target_shape[2]),
        )
        stack[:] = 0

        for t, p in normalized:
            arr = _as_zyx_mask(tifffile.imread(p))
            if tuple(arr.shape) != target_shape:
                arr_fit = np.zeros(target_shape, dtype=out_dtype)
                z_min = min(target_shape[0], arr.shape[0])
                y_min = min(target_shape[1], arr.shape[1])
                x_min = min(target_shape[2], arr.shape[2])
                arr_fit[:z_min, :y_min, :x_min] = arr[:z_min, :y_min, :x_min].astype(
                    out_dtype, copy=False
                )
                arr = arr_fit
            else:
                arr = arr.astype(out_dtype, copy=False)
            stack[t, :, :, :] = arr
            if t % 10 == 0:
                print(f"GUI_INFO: packed T={t}")

        stack.flush()
        del stack

        stack_ro = np.memmap(
            tmp_stack_path,
            mode="r",
            dtype=out_dtype,
            shape=(nt, target_shape[0], target_shape[1], target_shape[2]),
        )
        total_bytes = (
            int(nt)
            * int(target_shape[0])
            * int(target_shape[1])
            * int(target_shape[2])
            * np.dtype(out_dtype).itemsize
        )
        bigtiff = total_bytes >= 3_500_000_000
        tifffile.imwrite(
            output_path,
            stack_ro,
            compression=compression,
            metadata={"axes": "TZYX"},
            bigtiff=bigtiff,
        )
        del stack_ro
    finally:
        try:
            if os.path.isfile(tmp_stack_path):
                os.remove(tmp_stack_path)
        except Exception:
            pass

    return output_path, nt, target_shape, out_dtype


def _build_ome_from_timepoint_masks(parent):
    """GUI action: pick a folder of per-timepoint mask TIFFs and build one OME-TIFF."""
    if not GUI:
        print("ERROR: GUI not available for folder selection")
        return
    start_dir = os.path.dirname(parent.filename) if hasattr(parent, "filename") else os.getcwd()
    folder = QFileDialog.getExistingDirectory(
        parent, "Select folder containing timepoint mask TIFFs", start_dir
    )
    if not folder:
        return

    default_out = _default_ome_output_from_mask_folder(folder)
    out_name = QFileDialog.getSaveFileName(
        parent,
        "Save combined OME-TIFF",
        default_out,
        "OME-TIFF (*.ome.tif *.ome.tiff);;TIFF (*.tif *.tiff)",
    )
    output_path = out_name[0]
    if output_path == "":
        return

    try:
        out_path, nt, shape, out_dtype = combine_timepoint_masks_folder_to_ome(
            folder, output_path=output_path, strict_shape=False, compression="zlib"
        )
        print(
            f"GUI_INFO: wrote combined labeled OME-TIFF {out_path} "
            f"shape=(T={nt}, Z={shape[0]}, Y={shape[1]}, X={shape[2]}) dtype={np.dtype(out_dtype).name}"
        )
    except Exception as e:
        print(f"ERROR: failed to build OME-TIFF from folder {folder}: {e}")


def _save_png(parent):
    """save masks to png/tiff and, for time-lapse data, optionally one labeled OME-TIFF."""
    filename = parent.filename
    base = os.path.splitext(filename)[0]

    def _label_dtype(arr):
        try:
            max_label = int(np.asarray(arr).max()) if np.asarray(arr).size else 0
        except Exception:
            max_label = 0
        return np.uint16 if max_label < 2**16 else np.uint32

    if parent.NZ == 1:
        if parent.cellpix[0].max() > 65534:
            print("GUI_INFO: saving 2D masks to tif (too many masks for PNG)")
            labels_2d = np.asarray(parent.cellpix[0], dtype=_label_dtype(parent.cellpix[0]))
            tifffile.imwrite(
                base + "_cp_masks.tif",
                labels_2d,
                compression="zlib",
                metadata={"axes": "YX"},
            )
            print(
                f"GUI_INFO: saved labeled mask TIFF dtype={labels_2d.dtype} max={int(labels_2d.max())}"
            )
        else:
            print("GUI_INFO: saving 2D masks to png")
            imsave(base + "_cp_masks.png", parent.cellpix[0].astype(np.uint16))
    else:
        print("GUI_INFO: saving 3D masks to tiff")
        labels_zyx = np.asarray(parent.cellpix, dtype=_label_dtype(parent.cellpix))
        tifffile.imwrite(
            base + "_cp_masks.tif",
            labels_zyx,
            compression="zlib",
            metadata={"axes": "ZYX"},
        )
        print(
            f"GUI_INFO: saved labeled mask TIFF dtype={labels_zyx.dtype} max={int(labels_zyx.max())}"
        )

    # For time-lapse segmentation cached in memory, also export one labeled OME-TIFF (TZYX).
    per_time = getattr(parent, "seg_time_data", None)
    if not (
        getattr(parent, "has_time", False)
        and isinstance(per_time, dict)
        and len(per_time) > 0
    ):
        return

    def _time_index_from_entry(key, value):
        if isinstance(value, dict):
            t_val = value.get("time_index", None)
            if t_val is not None:
                try:
                    return int(t_val)
                except Exception:
                    pass
        try:
            return int(key)
        except Exception:
            return None

    nz = int(getattr(parent, "NZ", 1))
    Ly = int(getattr(parent, "Ly0", getattr(parent, "Ly", 0)))
    Lx = int(getattr(parent, "Lx0", getattr(parent, "Lx", 0)))
    if nz <= 0 or Ly <= 0 or Lx <= 0:
        print("ERROR: invalid image dimensions, cannot save time-lapse OME-TIFF labels")
        return

    entries = []
    max_label = 0
    for key, dat_t in per_time.items():
        if not isinstance(dat_t, dict):
            continue
        masks_t = dat_t.get("masks", None)
        if masks_t is None:
            continue
        t = _time_index_from_entry(key, dat_t)
        if t is None or t < 0:
            continue
        arr = np.asarray(masks_t)
        if arr.ndim not in (2, 3):
            continue
        if arr.size > 0:
            try:
                max_label = max(max_label, int(arr.max()))
            except Exception:
                pass
        entries.append((t, arr))

    if len(entries) == 0:
        print("GUI_INFO: no cached per-timepoint masks available for OME-TIFF export")
        return

    nt_from_parent = 0
    try:
        nt_from_parent = int(getattr(parent, "NT", 0))
    except Exception:
        nt_from_parent = 0
    nt = max(nt_from_parent, max(t for t, _ in entries) + 1)
    label_dtype = np.uint16 if max_label < 2**16 else np.uint32

    def _copy_to_zyx(dst, src):
        src = np.asarray(src)
        if src.ndim == 2:
            y_min = min(dst.shape[1], src.shape[0])
            x_min = min(dst.shape[2], src.shape[1])
            dst[0, :y_min, :x_min] = src[:y_min, :x_min].astype(label_dtype, copy=False)
        elif src.ndim == 3:
            z_min = min(dst.shape[0], src.shape[0])
            y_min = min(dst.shape[1], src.shape[1])
            x_min = min(dst.shape[2], src.shape[2])
            dst[:z_min, :y_min, :x_min] = src[:z_min, :y_min, :x_min].astype(
                label_dtype, copy=False
            )

    # Always keep per-timepoint TIFFs for time-lapse masks (useful for .sldy workflows
    # and as a robust fallback when *_seg.npy is unavailable).
    fallback_dir = base + "_cp_masks_by_time"
    os.makedirs(fallback_dir, exist_ok=True)
    for t, arr in entries:
        vol = np.zeros((nz, Ly, Lx), dtype=label_dtype)
        _copy_to_zyx(vol, arr)
        tifffile.imwrite(
            os.path.join(fallback_dir, f"masks_T{t:04d}_ZYX.tif"),
            vol,
            compression="zlib",
            metadata={"axes": "ZYX"},
        )
    print(
        f"GUI_INFO: saved {len(entries)} per-timepoint labeled TIFF stacks in {fallback_dir}"
    )

    total_bytes = nt * nz * Ly * Lx * np.dtype(label_dtype).itemsize
    if total_bytes > 1_500_000_000:
        print(
            "GUI_WARNING: time-lapse OME-TIFF would be too large; "
            "kept per-timepoint labeled TIFFs instead"
        )
        return

    labels_tzyx = np.zeros((nt, nz, Ly, Lx), dtype=label_dtype)
    for t, arr in entries:
        if 0 <= t < nt:
            _copy_to_zyx(labels_tzyx[t], arr)

    ome_path = base + "_cp_masks.ome.tif"
    tifffile.imwrite(
        ome_path,
        labels_tzyx,
        compression="zlib",
        metadata={"axes": "TZYX"},
    )
    print(f"GUI_INFO: saved time-lapse labeled masks to {ome_path}")


def _save_flows(parent):
    """ save flows and cellprob to tiff """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    print("GUI_INFO: saving flows and cellprob to tiff")
    if len(parent.flows) > 0:
        imsave(base + "_cp_cellprob.tif", parent.flows[1])
        for i in range(3):
            imsave(base + f"_cp_flows_{i}.tif", parent.flows[0][..., i])
        if len(parent.flows) > 2:
            imsave(base + "_cp_flows.tif", parent.flows[2])
        print("GUI_INFO: saved flows and cellprob")
    else:
        print("ERROR: no flows or cellprob found")


def _save_rois(parent):
    """ save masks as rois in .zip file for ImageJ """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ == 1:
        print(
            f"GUI_INFO: saving {parent.cellpix[0].max()} ImageJ ROIs to .zip archive.")
        save_rois(parent.cellpix[0], parent.filename)
    else:
        # For 3D stacks, write one ROI zip per Z plane.
        nz = int(parent.cellpix.shape[0]) if hasattr(parent, "cellpix") else int(parent.NZ)
        n_saved = 0
        for z in range(nz):
            masks_z = np.asarray(parent.cellpix[z])
            if masks_z.size == 0 or int(masks_z.max()) <= 0:
                continue
            out_name = f"{base}_Z{z:04d}.tif"
            try:
                save_rois(masks_z, out_name, prefix=f"Z{z:04d}_")
                n_saved += 1
            except Exception as e:
                print(f"ERROR: could not save ROI zip for Z={z}: {e}")
        if n_saved == 0:
            print("GUI_INFO: no non-empty masks found; no ROI zip files were created")
        else:
            print(
                f"GUI_INFO: saved ImageJ ROI zip files for {n_saved} Z-planes "
                f"(files named like {os.path.basename(base)}_Z0000_rois.zip)"
            )


def _save_outlines(parent):
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ == 1:
        print(
            "GUI_INFO: saving 2D outlines to text file, see docs for info to load into ImageJ"
        )
        outlines = outlines_list(parent.cellpix[0])
        outlines_to_text(base, outlines)
    else:
        # For 3D stacks, write one outlines text file per Z plane.
        out_dir = base + "_cp_outlines_by_z"
        os.makedirs(out_dir, exist_ok=True)
        nz = int(parent.cellpix.shape[0]) if hasattr(parent, "cellpix") else int(parent.NZ)
        n_saved = 0
        for z in range(nz):
            masks_z = np.asarray(parent.cellpix[z])
            if masks_z.size == 0 or int(masks_z.max()) <= 0:
                continue
            outlines = outlines_list(masks_z)
            if outlines is None or len(outlines) == 0:
                continue
            out_base = os.path.join(
                out_dir, f"{os.path.basename(base)}_Z{z:04d}"
            )
            outlines_to_text(out_base, outlines)
            n_saved += 1
        if n_saved == 0:
            print("GUI_INFO: no non-empty outlines found; no outline text files were created")
        else:
            print(
                f"GUI_INFO: saved outlines text files for {n_saved} Z-planes in {out_dir}"
            )


def _save_sets_with_check(parent):
    """ Save masks and update *_seg.npy file. Use this function when saving should be optional
     based on the disableAutosave checkbox. Otherwise, use _save_sets """
    if not parent.disableAutosave.isChecked():
        _save_sets(parent)


def _save_sets(parent):
    """ save masks to *_seg.npy. This function should be used when saving
    is forced, e.g. when clicking the save button. Otherwise, use _save_sets_with_check
    """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    # determine time index from current segmentation run, not GUI slider
    time_index = getattr(parent, "_current_seg_time_index", None)
    has_time = getattr(parent, "has_time", False) and time_index is not None
    # always compute a per-time save path when time_index is known
    if has_time:
        per_time_save = f"{base}_T{time_index:04d}_seg.npy"
    else:
        per_time_save = base + "_seg.npy"
    flow_threshold = parent.segmentation_settings.flow_threshold
    cellprob_threshold = parent.segmentation_settings.cellprob_threshold

    if parent.NZ > 1:
        dat = {
            "outlines":
                parent.outpix,
            "colors":
                parent.cellcolors[1:],
            "masks":
                parent.cellpix,
            "current_channel": (parent.color - 2) % 5,
            "filename":
                parent.filename,
            "flows":
                parent.flows,
            "zdraw":
                parent.zdraw,
            "model_path":
                parent.current_model_path
                if hasattr(parent, "current_model_path") else 0,
            "flow_threshold":
                flow_threshold,
            "cellprob_threshold":
                cellprob_threshold,
            "normalize_params":
                parent.get_normalize_params(),
            "restore":
                parent.restore,
            "ratio":
                parent.ratio,
            "diameter":
                parent.segmentation_settings.diameter,
            # spatial axes for volumetric masks: Z (planes), Y, X
            "axes":
                "ZYX",
        }
        if time_index is not None:
            dat["time_index"] = time_index
        if parent.restore is not None:
            dat["img_restore"] = parent.stack_filtered
    else:
        dat = {
            "outlines":
                parent.outpix.squeeze() if parent.restore is None or
                not "upsample" in parent.restore else parent.outpix_resize.squeeze(),
            "colors":
                parent.cellcolors[1:],
            "masks":
                parent.cellpix.squeeze() if parent.restore is None or
                not "upsample" in parent.restore else parent.cellpix_resize.squeeze(),
            "filename":
                parent.filename,
            "flows":
                parent.flows,
            "ismanual":
                parent.ismanual,
            "manual_changes":
                parent.track_changes,
            "model_path":
                parent.current_model_path
                if hasattr(parent, "current_model_path") else 0,
            "flow_threshold":
                flow_threshold,
            "cellprob_threshold":
                cellprob_threshold,
            "normalize_params":
                parent.get_normalize_params(),
            "restore":
                parent.restore,
            "ratio":
                parent.ratio,
            "diameter":
                parent.segmentation_settings.diameter,
            # spatial axes for 2D masks: Y, X
            "axes":
                "YX",
        }
        if time_index is not None:
            dat["time_index"] = time_index
        if parent.restore is not None:
            dat["img_restore"] = parent.stack_filtered
    # always write a per-time *_T####_seg.npy when time_index is known; otherwise write base_seg.npy
    try:
        seg_path = per_time_save
        tmp_path = seg_path + ".tmp"
        # Use a file handle so numpy does not append a second ".npy" suffix.
        with open(tmp_path, "wb") as fh:
            np.save(fh, dat)
        os.replace(tmp_path, seg_path)
        print("GUI_INFO: %d ROIs saved to %s" % (parent.ncells.get(), seg_path))
    except Exception as e:
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        print(f"ERROR: {e}")

    # additionally, for time-lapse / 5D data, maintain a single aggregated _seg.npy
    # that is aware of timepoints and stores per-timepoint segmentation dictionaries.
    # For OME-style consumers, this aggregated file also exposes masks/outlines
    # as a TZYX array (Time, Z, Y, X) with 'axes' == "TZYX".
    if has_time and time_index is not None:
        seg_all_path = base + "_seg.npy"
        try:
            all_dat = {}
            if os.path.isfile(seg_all_path):
                try:
                    all_dat = np.load(seg_all_path, allow_pickle=True).item()
                except Exception as e:
                    print(
                        f"ERROR: could not load existing seg_all file {seg_all_path}: {e}"
                    )
                    all_dat = {}
            per_time = all_dat.get("per_time", {})
            # store this timepoint's segmentation; keys can be ints
            per_time[time_index] = dat
            all_dat["per_time"] = per_time
            all_dat["filename"] = parent.filename

            # determine number of timepoints and spatial shape
            NT_existing = all_dat.get("NT", None)
            NT_parent = getattr(parent, "NT", None)
            try:
                NT_from_parent = int(NT_parent) if NT_parent is not None else None
            except Exception:
                NT_from_parent = None
            # minimal NT that can hold current time_index
            NT_min = time_index + 1
            NT_candidates = [NT_existing, NT_from_parent, NT_min]
            NT = None
            for val in NT_candidates:
                try:
                    if val is not None:
                        v_int = int(val)
                        if v_int > 0:
                            NT = v_int if NT is None else max(NT, v_int)
                except Exception:
                    continue
            if NT is None:
                NT = NT_min
            all_dat["NT"] = NT

            # spatial dimensions
            nz = (
                int(parent.NZ)
                if hasattr(parent, "NZ") and parent.NZ is not None
                else 1
            )
            Ly = int(parent.Ly0) if hasattr(parent, "Ly0") else int(parent.Ly)
            Lx = int(parent.Lx0) if hasattr(parent, "Lx0") else int(parent.Lx)

            # keep per-time data in memory regardless of save outcome
            try:
                parent.seg_time_data = per_time
            except Exception:
                pass

            # initialize or validate OME-style TZYX arrays, but skip if too large to be practical
            masks_ome = all_dat.get("masks", None)
            outlines_ome = all_dat.get("outlines", None)
            include_ome = True

            def _allocate_ome(arr, dtype):
                try:
                    if (
                        isinstance(arr, np.ndarray)
                        and arr.shape == (NT, nz, Ly, Lx)
                        and arr.dtype == dtype
                    ):
                        return arr
                except Exception:
                    pass
                return np.zeros((NT, nz, Ly, Lx), dtype=dtype)

            # convert current timepoint masks/outlines to ZYX volume matching parent.NZ, Ly0, Lx0
            masks_t = dat["masks"]
            if masks_t.ndim == 2:
                # single plane -> first Z slice, rest left empty
                masks_z = np.zeros((nz, Ly, Lx), dtype=masks_t.dtype)
                masks_z[0, :, :] = masks_t
            elif masks_t.ndim == 3:
                # assume Z, Y, X
                if (
                    masks_t.shape[0] == nz
                    and masks_t.shape[1] == Ly
                    and masks_t.shape[2] == Lx
                ):
                    masks_z = masks_t
                else:
                    # best-effort resize / crop-pad mismatch
                    masks_z = np.zeros((nz, Ly, Lx), dtype=masks_t.dtype)
                    z_min = min(nz, masks_t.shape[0])
                    y_min = min(Ly, masks_t.shape[1])
                    x_min = min(Lx, masks_t.shape[2])
                    masks_z[:z_min, :y_min, :x_min] = masks_t[:z_min, :y_min, :x_min]
            else:
                # unexpected ndim; fall back to zeros and warn
                print(
                    f"GUI_WARNING: unexpected masks ndim={masks_t.ndim} for OME export; skipping volume copy"
                )
                masks_z = np.zeros((nz, Ly, Lx), dtype=masks_t.dtype)

            outlines_t = dat.get("outlines", None)
            if isinstance(outlines_t, np.ndarray):
                if outlines_t.ndim == 2:
                    outlines_z = np.zeros((nz, Ly, Lx), dtype=outlines_t.dtype)
                    outlines_z[0, :, :] = outlines_t
                elif outlines_t.ndim == 3:
                    if (
                        outlines_t.shape[0] == nz
                        and outlines_t.shape[1] == Ly
                        and outlines_t.shape[2] == Lx
                    ):
                        outlines_z = outlines_t
                    else:
                        outlines_z = np.zeros((nz, Ly, Lx), dtype=outlines_t.dtype)
                        z_min = min(nz, outlines_t.shape[0])
                        y_min = min(Ly, outlines_t.shape[1])
                        x_min = min(Lx, outlines_t.shape[2])
                        outlines_z[:z_min, :y_min, :x_min] = outlines_t[
                            :z_min, :y_min, :x_min
                        ]
                else:
                    print(
                        f"GUI_WARNING: unexpected outlines ndim={outlines_t.ndim} for OME export; skipping volume copy"
                    )
                    outlines_z = None
            else:
                outlines_z = None

            # size guard for OME export: skip if too large (prevents MemoryError / huge files)
            try:
                total_bytes = masks_z.size * masks_z.dtype.itemsize * max(NT, 1)
                # cap around ~1.5GB per OME array to avoid runaway memory/disk usage
                if total_bytes > 1_500_000_000:
                    include_ome = False
                    print(
                        "GUI_WARNING: skipping OME TZYX masks/outlines in seg_all file "
                        f"(estimated size {total_bytes/1e9:.2f} GB too large)"
                    )
            except Exception:
                pass

            if include_ome:
                try:
                    masks_ome = _allocate_ome(masks_ome, masks_z.dtype)
                    masks_ome[time_index, :, :, :] = masks_z
                    all_dat["masks"] = masks_ome

                    if outlines_z is not None:
                        outlines_ome = _allocate_ome(outlines_ome, outlines_z.dtype)
                        outlines_ome[time_index, :, :, :] = outlines_z
                        all_dat["outlines"] = outlines_ome

                    # mark axes for OME consumers
                    all_dat["axes"] = "TZYX"
                except MemoryError as me:
                    include_ome = False
                    print(
                        f"GUI_WARNING: skipping OME TZYX export due to memory error: {me}"
                    )
                except Exception as ee:
                    include_ome = False
                    print(
                        f"GUI_WARNING: skipping OME TZYX export due to error: {ee}"
                    )
            if not include_ome:
                # drop any stale OME arrays to keep seg_all compact when skipping
                all_dat.pop("masks", None)
                all_dat.pop("outlines", None)
                all_dat.pop("axes", None)

            try:
                tmp_all_path = seg_all_path + ".tmp"
                # Use a file handle so numpy does not append a second ".npy" suffix.
                with open(tmp_all_path, "wb") as fh:
                    np.save(fh, all_dat)
                os.replace(tmp_all_path, seg_all_path)
                print(
                    f"GUI_INFO: updated timepoint {time_index} in aggregated seg file {seg_all_path}"
                )
            except Exception as e:
                try:
                    if os.path.isfile(tmp_all_path):
                        os.remove(tmp_all_path)
                except Exception:
                    pass
                print(
                    f"ERROR: could not save aggregated seg_all file {seg_all_path}: {type(e).__name__}: {e}"
                )
            # keep in-memory cache so time navigation doesn't reread from disk
            try:
                parent.seg_time_data = per_time
            except Exception:
                pass
        except Exception as e:
            # Never let aggregated save failures prevent per-time *_seg.npy from working.
            print(
                f"ERROR: failed to update aggregated seg_all file {seg_all_path}: {e}"
            )
    del dat
