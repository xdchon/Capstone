"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.
"""

import os, time
from pathlib import Path
import numpy as np
from tqdm import trange
import torch
from scipy.ndimage import gaussian_filter
import gc
import cv2

import logging

models_logger = logging.getLogger(__name__)

from . import transforms, dynamics, utils, plot
from .vit_sam import Transformer
from .core import assign_device, run_net, run_3D

_CPSAM_MODEL_URL = "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam"
_MODEL_DIR_ENV = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
_MODEL_DIR_DEFAULT = Path.home().joinpath(".cellpose", "models")
MODEL_DIR = Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else _MODEL_DIR_DEFAULT

MODEL_NAMES = ["cpsam"]

MODEL_LIST_PATH = os.fspath(MODEL_DIR.joinpath("gui_models.txt"))

normalize_default = {
    "lowhigh": None,
    "percentile": None,
    "normalize": True,
    "norm3D": True,
    "sharpen_radius": 0,
    "smooth_radius": 0,
    "tile_norm_blocksize": 0,
    "tile_norm_smooth3D": 1,
    "invert": False
}


def model_path(model_type, model_index=0):
    return cache_CPSAM_model_path()


def cache_CPSAM_model_path():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cached_file = os.fspath(MODEL_DIR.joinpath('cpsam'))
    if not os.path.exists(cached_file):
        models_logger.info('Downloading: "{}" to {}\n'.format(_CPSAM_MODEL_URL, cached_file))
        utils.download_url_to_file(_CPSAM_MODEL_URL, cached_file, progress=True)
    return cached_file


def get_user_models():
    model_strings = []
    if os.path.exists(MODEL_LIST_PATH):
        with open(MODEL_LIST_PATH, "r") as textfile:
            lines = [line.rstrip() for line in textfile]
            if len(lines) > 0:
                model_strings.extend(lines)
    return model_strings


class CellposeModel():
    """
    Class representing a Cellpose model.

    Attributes:
        diam_mean (float): Mean "diameter" value for the model.
        builtin (bool): Whether the model is a built-in model or not.
        device (torch device): Device used for model running / training.
        nclasses (int): Number of classes in the model.
        nbase (list): List of base values for the model.
        net (CPnet): Cellpose network.
        pretrained_model (str): Path to pretrained cellpose model.
        pretrained_model_ortho (str): Path or model_name for pretrained cellpose model for ortho views in 3D.
        backbone (str): Type of network ("default" is the standard res-unet, "transformer" for the segformer).

    Methods:
        __init__(self, gpu=False, pretrained_model=False, model_type=None, diam_mean=30., device=None):
            Initialize the CellposeModel.
        
        eval(self, x, batch_size=8, resample=True, channels=None, channel_axis=None, z_axis=None, normalize=True, invert=False, rescale=None, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False, anisotropy=None, stitch_threshold=0.0, min_size=15, niter=None, augment=False, tile_overlap=0.1, bsize=256, interp=True, compute_masks=True, progress=None):
            Segment list of images x, or 4D array - Z x C x Y x X.

    """

    def __init__(self, gpu=False, pretrained_model="cpsam", model_type=None,
                 diam_mean=None, device=None, nchan=None, use_bfloat16=True):
        """
        Initialize the CellposeModel.

        Parameters:
            gpu (bool, optional): Whether or not to save model to GPU, will check if GPU available.
            pretrained_model (str or list of strings, optional): Full path to pretrained cellpose model(s), if None or False, no model loaded.
            model_type (str, optional): Any model that is available in the GUI, use name in GUI e.g. "livecell" (can be user-trained or model zoo).
            diam_mean (float, optional): Mean "diameter", 30. is built-in value for "cyto" model; 17. is built-in value for "nuclei" model; if saved in custom model file (cellpose>=2.0) then it will be loaded automatically and overwrite this value.
            device (torch device, optional): Device used for model running / training (torch.device("cuda") or torch.device("cpu")), overrides gpu input, recommended if you want to use a specific GPU (e.g. torch.device("cuda:1")).
            use_bfloat16 (bool, optional): Use 16bit float precision instead of 32bit for model weights. Default to 16bit (True).
        """
        if diam_mean is not None:
            models_logger.warning(
                "diam_mean argument are not used in v4.0.1+. Ignoring this argument..."
            )
        if model_type is not None:
            models_logger.warning(
                "model_type argument is not used in v4.0.1+. Ignoring this argument..."
            )
        if nchan is not None:
            models_logger.warning("nchan argument is deprecated in v4.0.1+. Ignoring this argument")

        ### assign model device
        self.device = assign_device(gpu=gpu)[0] if device is None else device
        if torch.cuda.is_available():
            device_gpu = self.device.type == "cuda"
        elif torch.backends.mps.is_available():
            device_gpu = self.device.type == "mps"
        else:
            device_gpu = False
        self.gpu = device_gpu

        if pretrained_model is None:
            raise ValueError("Must specify a pretrained model, training from scratch is not implemented")
        
        ### create neural network
        if pretrained_model and not os.path.exists(pretrained_model):
            # check if pretrained model is in the models directory
            model_strings = get_user_models()
            all_models = MODEL_NAMES.copy()
            all_models.extend(model_strings)
            if pretrained_model in all_models:
                pretrained_model = os.path.join(MODEL_DIR, pretrained_model)
            else:
                pretrained_model = os.path.join(MODEL_DIR, "cpsam")
                models_logger.warning(
                    f"pretrained model {pretrained_model} not found, using default model"
                )

        self.pretrained_model = pretrained_model
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.net = Transformer(dtype=dtype).to(self.device)

        if os.path.exists(self.pretrained_model):
            models_logger.info(f">>>> loading model {self.pretrained_model}")
            self.net.load_model(self.pretrained_model, device=self.device)
        else:
            if os.path.split(self.pretrained_model)[-1] != 'cpsam':
                raise FileNotFoundError('model file not recognized')
            cache_CPSAM_model_path()
            self.net.load_model(self.pretrained_model, device=self.device)
        
        
    def eval(self, x, batch_size=8, resample=True, channels=None, channel_axis=None,
             z_axis=None, normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, 
             min_size=15, max_size_fraction=0.4, niter=None, 
             augment=False, tile_overlap=0.1, bsize=256, 
             compute_masks=True, progress=None):
        """ segment list of images x, or 4D array - Z x 3 x Y x X

        Args:
            x (list, np.ndarry): can be list of 2D/3D/4D images, or array of 2D/3D/4D images. Images must have 3 channels.
            batch_size (int, optional): number of 256x256 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage). Defaults to 64.
            resample (bool, optional): run dynamics at original image size (will be slower but create more accurate boundaries). 
            channel_axis (int, optional): channel axis in element of list x, or of np.ndarray x. 
                if None, channels dimension is attempted to be automatically determined. Defaults to None.
            z_axis  (int, optional): z axis in element of list x, or of np.ndarray x. 
                if None, z dimension is attempted to be automatically determined. Defaults to None.
            normalize (bool, optional): if True, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel; 
                can also pass dictionary of parameters (all keys are optional, default values shown): 
                    - "lowhigh"=None : pass in normalization values for 0.0 and 1.0 as list [low, high] (if not None, all following parameters ignored)
                    - "sharpen"=0 ; sharpen image with high pass filter, recommended to be 1/4-1/8 diameter of cells in pixels
                    - "normalize"=True ; run normalization (if False, all following parameters ignored)
                    - "percentile"=None : pass in percentiles to use as list [perc_low, perc_high]
                    - "tile_norm_blocksize"=0 ; compute normalization in tiles across image to brighten dark areas, to turn on set to window size in pixels (e.g. 100)
                    - "norm3D"=True ; compute normalization across entire z-stack rather than plane-by-plane in stitching mode.
                Defaults to True.
            invert (bool, optional): invert image pixel intensity before running network. Defaults to False.
            rescale (float, optional): resize factor for each image, if None, set to 1.0;
                (only used if diameter is None). Defaults to None.
            diameter (float or list of float, optional): diameters are used to rescale the image to 30 pix cell diameter.
            flow_threshold (float, optional): flow error threshold (all cells with errors below threshold are kept) (not used for 3D). Defaults to 0.4.
            cellprob_threshold (float, optional): all pixels with value above threshold kept for masks, decrease to find more and larger masks. Defaults to 0.0.
            do_3D (bool, optional): set to True to run 3D segmentation on 3D/4D image input. Defaults to False.
            flow3D_smooth (int, optional): if do_3D and flow3D_smooth>0, smooth flows with gaussian filter of this stddev. Defaults to 0.
            anisotropy (float, optional): for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y). Defaults to None.
            stitch_threshold (float, optional): if stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume segmentation. Defaults to 0.0.
            min_size (int, optional): all ROIs below this size, in pixels, will be discarded. Defaults to 15.
            max_size_fraction (float, optional): max_size_fraction (float, optional): Masks larger than max_size_fraction of
                total image size are removed. Default is 0.4.
            niter (int, optional): number of iterations for dynamics computation. if None, it is set proportional to the diameter. Defaults to None.
            augment (bool, optional): tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
            tile_overlap (float, optional): fraction of overlap of tiles when computing flows. Defaults to 0.1.
            bsize (int, optional): block size for tiles, recommended to keep at 256, like in training. Defaults to 256.
            interp (bool, optional): interpolate during 2D dynamics (not available in 3D) . Defaults to True.
            compute_masks (bool, optional): Whether or not to compute dynamics and return masks. Returns empty array if False. Defaults to True.
            progress (QProgressBar, optional): pyqt progress bar. Defaults to None.

        Returns:
            A tuple containing (masks, flows, styles, diams): 
            masks (list of 2D arrays or single 3D array): Labelled image, where 0=no masks; 1,2,...=mask labels;
            flows (list of lists 2D arrays or list of 3D arrays): 
                flows[k][0] = XY flow in HSV 0-255; 
                flows[k][1] = XY flows at each pixel; 
                flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics); 
                flows[k][3] = final pixel locations after Euler integration; 
            styles (list of 1D arrays of length 256 or single 1D array): Style vector containing only zeros. Retained for compaibility with CP3. 
            
        """

        if rescale is not None:
            models_logger.warning("rescaling deprecated in v4.0.1+") 
        if channels is not None:
            models_logger.warning("channels deprecated in v4.0.1+. If data contain more than 3 channels, only the first 3 channels will be used")

        # Lazy 5D time-lapse sources (e.g. SlideBook .sldy or TCZYX virtual stacks)
        # expose a get_time_stack(t_index) method to pull one Z-stack at a time.
        if hasattr(x, "get_time_stack"):
            # determine number of timepoints
            n_time = None
            if hasattr(x, "nt"):
                try:
                    n_time = int(getattr(x, "nt"))
                except Exception:
                    n_time = None
            if n_time is None and hasattr(x, "axes") and hasattr(x, "shape"):
                try:
                    axes_str = getattr(x, "axes", None)
                    if axes_str is not None:
                        axes_up = str(axes_str).upper()
                        if "T" in axes_up:
                            t_axis = axes_up.index("T")
                            n_time = int(getattr(x, "shape")[t_axis])
                except Exception:
                    n_time = None
            if n_time is None and hasattr(x, "shape"):
                try:
                    n_time = int(getattr(x, "shape")[0])
                except Exception:
                    n_time = 1
            if n_time is None or n_time < 1:
                n_time = 1

            self.timing = []
            masks_all, flows_all, styles_all = [], [], []
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(n_time, file=tqdm_out,
                              mininterval=30) if n_time > 1 else range(n_time)
            for t in iterator:
                tic = time.time()
                stack_t = x.get_time_stack(t)
                diam_t = diameter[t] if isinstance(diameter, (list, np.ndarray)) else diameter
                mask_t, flow_t, style_t = self.eval(
                    stack_t,
                    batch_size=batch_size,
                    resample=resample,
                    channels=channels,
                    channel_axis=channel_axis,
                    z_axis=z_axis,
                    normalize=normalize,
                    invert=invert,
                    rescale=rescale,
                    diameter=diam_t,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    do_3D=do_3D,
                    flow3D_smooth=flow3D_smooth,
                    anisotropy=anisotropy,
                    stitch_threshold=stitch_threshold,
                    min_size=min_size,
                    max_size_fraction=max_size_fraction,
                    niter=niter,
                    augment=augment,
                    tile_overlap=tile_overlap,
                    bsize=bsize,
                    interp=interp,
                    compute_masks=compute_masks,
                    progress=progress,
                )
                masks_all.append(mask_t)
                flows_all.append(flow_t)
                styles_all.append(style_t)
                self.timing.append(time.time() - tic)
            return masks_all, flows_all, styles_all

        if isinstance(x, list) or x.squeeze().ndim == 5:
            self.timing = []
            masks, styles, flows = [], [], []
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            nimg = len(x)
            iterator = trange(nimg, file=tqdm_out,
                              mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                tic = time.time()
                maski, flowi, stylei = self.eval(
                    x[i], 
                    batch_size=batch_size,
                    channel_axis=channel_axis, 
                    z_axis=z_axis,
                    normalize=normalize, 
                    invert=invert,
                    diameter=diameter[i] if isinstance(diameter, list) or
                        isinstance(diameter, np.ndarray) else diameter, 
                    do_3D=do_3D,
                    anisotropy=anisotropy, 
                    augment=augment, 
                    tile_overlap=tile_overlap, 
                    bsize=bsize, 
                    resample=resample,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold, 
                    compute_masks=compute_masks,
                    min_size=min_size, 
                    max_size_fraction=max_size_fraction, 
                    stitch_threshold=stitch_threshold, 
                    flow3D_smooth=flow3D_smooth,
                    progress=progress, 
                    niter=niter)
                masks.append(maski)
                flows.append(flowi)
                styles.append(stylei)
                self.timing.append(time.time() - tic)
            return masks, flows, styles

        ############# actual eval code ############
        # reshape image
        x = transforms.convert_image(x, channel_axis=channel_axis,
                                        z_axis=z_axis, 
                                        time_axis=None,
                                        do_3D=(do_3D or stitch_threshold > 0))

        def _eval_no_time(x_in):
            # Add batch dimension if not present
            if x_in.ndim < 4:
                x_local = x_in[np.newaxis, ...]
            else:
                x_local = x_in
            nimg_local = x_local.shape[0]

            # pad channels to 3 here (after possible 5D split) to avoid huge allocations earlier
            num_channels_local = x_local.shape[-1]
            if num_channels_local < 3:
                x_out = np.zeros((*x_local.shape[:-1], 3), dtype=x_local.dtype)
                x_out[..., :num_channels_local] = x_local
                x_local = x_out
                del x_out
            elif num_channels_local > 3:
                x_local = x_local[..., :3]
            
            image_scaling = None
            Ly_0 = x_local.shape[1]
            Lx_0 = x_local.shape[2]
            Lz_0 = None
            if do_3D or stitch_threshold > 0:
                Lz_0 = x_local.shape[0]
            if diameter is not None:
                image_scaling = 30. / diameter
                # avoid upsampling very large stacks
                if (do_3D or stitch_threshold > 0) and x_local.size > 256**3 and image_scaling > 1.0:
                    models_logger.info(
                        f"skipping upsampling (image_scaling={image_scaling:.2f}) "
                        "because volume is very large"
                    )
                    image_scaling = 1.0
                x_local = transforms.resize_image(
                    x_local,
                    Ly=int(x_local.shape[1] * image_scaling),
                    Lx=int(x_local.shape[2] * image_scaling),
                )

            # normalize image
            normalize_params = normalize_default
            if isinstance(normalize, dict):
                normalize_params = {**normalize_params, **normalize}
            elif not isinstance(normalize, bool):
                raise ValueError("normalize parameter must be a bool or a dict")
            else:
                normalize_params["normalize"] = normalize
                normalize_params["invert"] = invert

            # pre-normalize if 3D stack for stitching or do_3D
            do_normalization = True if normalize_params["normalize"] else False
            if nimg_local > 1 and do_normalization and (stitch_threshold or do_3D):
                normalize_params["norm3D"] = True if do_3D else normalize_params["norm3D"]
                x_local = transforms.normalize_img(x_local, **normalize_params)
                do_normalization = False # do not normalize again
            else:
                if normalize_params["norm3D"] and nimg_local > 1 and do_normalization:
                    models_logger.warning(
                        "normalize_params['norm3D'] is True but do_3D is False and stitch_threshold=0, so setting to False"
                    )
                    normalize_params["norm3D"] = False
            if do_normalization:
                x_local = transforms.normalize_img(x_local, **normalize_params)

            # ajust the anisotropy when diameter is specified and images are resized:
            anisotropy_local = anisotropy
            if isinstance(anisotropy_local, (float, int)) and image_scaling:
                anisotropy_local = image_scaling * anisotropy_local

            dP, cellprob, styles = self._run_net(
                x_local, 
                augment=augment, 
                batch_size=batch_size, 
                tile_overlap=tile_overlap, 
                bsize=bsize,
                do_3D=do_3D, 
                anisotropy=anisotropy_local)

            if do_3D:    
                if flow3D_smooth > 0:
                    models_logger.info(f"smoothing flows with sigma={flow3D_smooth}")
                    dP = gaussian_filter(dP, (0, flow3D_smooth, flow3D_smooth, flow3D_smooth))
                torch.cuda.empty_cache()
                gc.collect()

                if resample:
                    # resize XY then YZ and then put channels first
                    dP = transforms.resize_image(dP.transpose(1, 2, 3, 0), Ly=Ly_0, Lx=Lx_0, no_channels=False)
                    dP = transforms.resize_image(dP.transpose(1, 0, 2, 3), Lx=Lx_0, Ly=Lz_0, no_channels=False)
                    dP = dP.transpose(3, 1, 0, 2)

                    # resize cellprob:
                    cellprob = transforms.resize_image(cellprob, Ly=Ly_0, Lx=Lx_0, no_channels=True)
                    cellprob = transforms.resize_image(cellprob.transpose(1, 0, 2), Lx=Lx_0, Ly=Lz_0, no_channels=True)
                    cellprob = cellprob.transpose(1, 0, 2)


            # 2d case:
            if resample and not do_3D:
                # 2D images have N = 1 in batch dimension:
                dP = transforms.resize_image(dP.transpose(1, 2, 3, 0), Ly=Ly_0, Lx=Lx_0, no_channels=False).transpose(3, 0, 1, 2)
                cellprob = transforms.resize_image(cellprob, Ly=Ly_0, Lx=Lx_0, no_channels=True)

            if compute_masks:
                # use user niter if specified, otherwise scale niter (200) with diameter
                niter_scale = 1 if image_scaling is None else image_scaling
                niter_local = int(200 / niter_scale) if (niter is None or niter == 0) else niter
                masks = self._compute_masks(
                    (Lz_0 or nimg_local, Ly_0, Lx_0),
                    dP,
                    cellprob,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    min_size=min_size,
                    max_size_fraction=max_size_fraction,
                    niter=niter_local,
                    stitch_threshold=stitch_threshold,
                    do_3D=do_3D,
                )
            else:
                masks = np.zeros(0) #pass back zeros if not compute_masks
            
            masks, dP, cellprob = masks.squeeze(), dP.squeeze(), cellprob.squeeze()

            return masks, [plot.dx_to_circ(dP), dP, cellprob], styles

        if x.ndim == 5:
            masks_all, flows_all, styles_all = [], [], []
            for t in range(x.shape[0]):
                m_t, f_t, s_t = _eval_no_time(x[t])
                masks_all.append(m_t)
                flows_all.append(f_t)
                styles_all.append(s_t)
            return masks_all, flows_all, styles_all

        return _eval_no_time(x)
    

    def _run_net(self, x, 
                augment=False, 
                batch_size=8, tile_overlap=0.1,
                bsize=256, anisotropy=1.0, do_3D=False):
        """ run network on image x """
        tic = time.time()
        shape = x.shape
        nimg = shape[0]


        if do_3D:
            Lz, Ly, Lx = shape[:-1]
            if anisotropy is not None and anisotropy != 1.0:
                models_logger.info(f"resizing 3D image with anisotropy={anisotropy}")
                x = transforms.resize_image(x.transpose(1,0,2,3),
                                        Ly=int(Lz*anisotropy), 
                                        Lx=int(Lx)).transpose(1,0,2,3)
            yf, styles = run_3D(self.net, x,
                                batch_size=batch_size, augment=augment,  
                                tile_overlap=tile_overlap, 
                                bsize=bsize
                                )
            cellprob = yf[..., -1]
            dP = yf[..., :-1].transpose((3, 0, 1, 2))
        else:
            yf, styles = run_net(self.net, x, bsize=bsize, augment=augment,
                                batch_size=batch_size,  
                                tile_overlap=tile_overlap, 
                                )
            cellprob = yf[..., -1]
            dP = yf[..., -3:-1].transpose((3, 0, 1, 2))
            if yf.shape[-1] > 3:
                styles = yf[..., :-3]
        
        styles = styles.squeeze()

        net_time = time.time() - tic
        if nimg > 1:
            models_logger.info("network run in %2.2fs" % (net_time))

        return dP, cellprob, styles
    
    def _compute_masks(self, shape, dP, cellprob, flow_threshold=0.4, cellprob_threshold=0.0,
                       min_size=15, max_size_fraction=0.4, niter=None,
                       do_3D=False, stitch_threshold=0.0):
        """ compute masks from flows and cell probability """
        changed_device_from = None
        if self.device.type == "mps" and do_3D:
            models_logger.warning("MPS does not support 3D post-processing, switching to CPU")
            self.device = torch.device("cpu")
            changed_device_from = "mps"
        Lz, Ly, Lx = shape[:3]
        tic = time.time()
        if do_3D:
            masks = dynamics.resize_and_compute_masks(
                dP, cellprob, niter=niter, cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold, do_3D=do_3D,
                min_size=min_size, max_size_fraction=max_size_fraction, 
                resize=shape[:3] if (np.array(dP.shape[-3:])!=np.array(shape[:3])).sum() 
                        else None,
                device=self.device)
        else:
            nimg = shape[0]
            Ly0, Lx0 = cellprob[0].shape 
            resize = None if Ly0==Ly and Lx0==Lx else [Ly, Lx]
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(nimg, file=tqdm_out,
                            mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                # turn off min_size for 3D stitching
                min_size0 = min_size if stitch_threshold == 0 or nimg == 1 else -1
                outputs = dynamics.resize_and_compute_masks(
                    dP[:, i], cellprob[i],
                    niter=niter, cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold, resize=resize,
                    min_size=min_size0, max_size_fraction=max_size_fraction,
                    device=self.device)
                if i==0 and nimg > 1:
                    masks = np.zeros((nimg, shape[1], shape[2]), outputs.dtype)
                if nimg > 1:
                    masks[i] = outputs
                else:
                    masks = outputs

            if stitch_threshold > 0 and nimg > 1:
                models_logger.info(
                    f"stitching {nimg} planes using stitch_threshold={stitch_threshold:0.3f} to make 3D masks"
                )
                masks = utils.stitch3D(masks, stitch_threshold=stitch_threshold)
                masks = utils.fill_holes_and_remove_small_masks(
                    masks, min_size=min_size)
            elif nimg > 1:
                models_logger.warning(
                    "3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only"
                )

        flow_time = time.time() - tic
        if shape[0] > 1:
            models_logger.info("masks created in %2.2fs" % (flow_time))
        
        if changed_device_from is not None:
            models_logger.info("switching back to device %s" % self.device)
            self.device = torch.device(changed_device_from)
        return masks
