"""Ultrack-based tracking utilities for Cellpose-generated label masks.

This module assumes segmentation is already done. It only performs tracking
over time and exports new label masks whose values correspond to Ultrack
`track_id` assignments.
"""

from __future__ import annotations

import copy
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import tifffile
from scipy import ndimage


LabelSeriesInput = np.ndarray | Sequence[np.ndarray] | Mapping[int, np.ndarray]


@dataclass(slots=True)
class UltrackTrackingConfig:
    """High-level Ultrack settings for tracking existing instance labels."""

    working_dir: str | Path | None = None
    database: str = "sqlite"
    sigma: float | Sequence[float] | None = None
    scale: Sequence[float] | None = None
    vector_field: Any = None
    overwrite: str | bool = "all"
    n_workers: int | None = None
    min_area: int | None = None
    min_area_factor: float | None = None
    max_area: int | None = None
    min_frontier: float = 0.1
    threshold: float | None = None
    anisotropy_penalization: float | None = None
    max_distance: float | None = None
    max_neighbors: int | None = None
    distance_weight: float | None = None
    appear_weight: float | None = None
    disappear_weight: float | None = None
    division_weight: float | None = None
    image_border_size: Sequence[int] | None = None
    window_size: int | None = None
    overlap_size: int | None = None
    solution_gap: float | None = None
    time_limit: int | None = None
    solver_name: str | None = None
    n_threads: int | None = None
    link_function: str | None = None
    power: float | None = None
    bias: float | None = None
    include_parents: bool = True
    include_node_ids: bool = True
    segment_kwargs: Mapping[str, Any] = field(default_factory=dict)
    link_kwargs: Mapping[str, Any] = field(default_factory=dict)
    solve_kwargs: Mapping[str, Any] = field(default_factory=dict)
    export_store_or_path: str | Path | None = None
    export_chunks: Sequence[int] | None = None
    export_overwrite: bool = True


@dataclass(slots=True)
class UltrackTrackingResult:
    """Tracked outputs returned by :func:`track_labels_with_ultrack`."""

    tracked_masks: np.ndarray
    tracks_df: Any
    lineage_graph: Mapping[int, Any]
    time_indices: tuple[int, ...]
    main_config: Any
    working_dir: Path

    @property
    def n_timepoints(self) -> int:
        return len(self.time_indices)

    @property
    def ndim_per_timepoint(self) -> int:
        return max(0, int(self.tracked_masks.ndim) - 1)

    def as_time_dict(self) -> dict[int, np.ndarray]:
        """Return tracked masks as ``{time_index: labels}``."""
        return {
            int(t): np.asarray(self.tracked_masks[i]).copy()
            for i, t in enumerate(self.time_indices)
        }


def _import_ultrack_api() -> SimpleNamespace:
    """Import Ultrack lazily so Cellpose can still import without it installed."""

    try:
        import ultrack
    except ImportError as exc:
        raise ImportError(
            "Ultrack is required for tracking existing masks. "
            "Install it in this environment with `pip install ultrack`."
        ) from exc

    try:
        from ultrack import MainConfig, Tracker
    except ImportError as exc:
        raise ImportError(
            "Ultrack is installed but `MainConfig`/`Tracker` could not be imported."
        ) from exc

    try:
        from ultrack.config import (
            DataConfig,
            LinkingConfig,
            SegmentationConfig,
            TrackingConfig,
        )
    except ImportError:
        DataConfig = getattr(ultrack, "DataConfig", None)
        SegmentationConfig = getattr(ultrack, "SegmentationConfig", None)
        LinkingConfig = getattr(ultrack, "LinkingConfig", None)
        TrackingConfig = getattr(ultrack, "TrackingConfig", None)

    tracks_to_zarr = getattr(ultrack, "tracks_to_zarr", None)
    if tracks_to_zarr is None:
        try:
            from ultrack.core.export import tracks_to_zarr
        except ImportError:
            tracks_to_zarr = None

    return SimpleNamespace(
        MainConfig=MainConfig,
        Tracker=Tracker,
        DataConfig=DataConfig,
        SegmentationConfig=SegmentationConfig,
        LinkingConfig=LinkingConfig,
        TrackingConfig=TrackingConfig,
        tracks_to_zarr=tracks_to_zarr,
    )


def _clone_main_config(config: Any) -> Any:
    copy_method = getattr(config, "copy", None)
    if callable(copy_method):
        try:
            return copy_method(deep=True)
        except TypeError:
            return copy_method()
    return copy.deepcopy(config)


def _ensure_subconfig(main_config: Any, attr_name: str, cls: Any) -> Any:
    subconfig = getattr(main_config, attr_name, None)
    if subconfig is None:
        if cls is None:
            raise RuntimeError(
                f"Ultrack did not expose `{attr_name}` config class; "
                "cannot construct a complete MainConfig."
            )
        subconfig = cls()
        setattr(main_config, attr_name, subconfig)
    return subconfig


def _normalize_overwrite(overwrite: str | bool) -> str:
    if isinstance(overwrite, bool):
        return "all" if overwrite else "none"

    value = str(overwrite).strip().lower()
    valid = {"all", "links", "solutions", "none"}
    if value not in valid:
        raise ValueError(
            f"Invalid Ultrack overwrite mode `{overwrite}`. Expected one of {sorted(valid)}."
        )
    return value


def _choose_label_dtype(mask: np.ndarray) -> np.dtype:
    max_label = int(np.max(mask)) if mask.size else 0
    if max_label < 2**16:
        return np.uint16
    if max_label < 2**32:
        return np.uint32
    return np.uint64


def _coerce_integer_labels(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim not in (2, 3, 4):
        raise ValueError(
            "Expected labels with shape (Y, X), (Z, Y, X), (T, Y, X), or (T, Z, Y, X)."
        )

    if not np.issubdtype(arr.dtype, np.integer):
        rounded = np.rint(arr)
        if not np.allclose(arr, rounded):
            raise TypeError("Tracking labels must contain integer-valued IDs.")
        arr = rounded.astype(np.int64, copy=False)

    if np.any(arr < 0):
        raise ValueError("Tracking labels must be non-negative, with background=0.")

    if arr.dtype.itemsize < 4:
        arr = arr.astype(np.int32, copy=False)
    return arr


def _stack_label_sequence(frames: Sequence[np.ndarray]) -> np.ndarray:
    if len(frames) == 0:
        raise ValueError("At least one timepoint is required for Ultrack tracking.")

    normalized = [_coerce_integer_labels(np.asarray(frame)) for frame in frames]
    expected_shape = normalized[0].shape
    expected_ndim = normalized[0].ndim
    if expected_ndim not in (2, 3):
        raise ValueError(
            "Each timepoint must be a 2D (Y, X) or 3D (Z, Y, X) label image."
        )

    for i, frame in enumerate(normalized[1:], start=1):
        if frame.ndim != expected_ndim:
            raise ValueError(
                "All timepoints must have the same dimensionality; "
                f"timepoint 0 has ndim={expected_ndim}, timepoint {i} has ndim={frame.ndim}."
            )
        if frame.shape != expected_shape:
            raise ValueError(
                "All timepoints must share the same spatial shape; "
                f"timepoint 0 has shape={expected_shape}, timepoint {i} has shape={frame.shape}."
            )

    return np.stack(normalized, axis=0)


def _make_frame_labels_trackable(frame: np.ndarray) -> np.ndarray:
    """Split disconnected components that share the same label ID.

    Cellpose slice-wise masks can reuse the same numeric label on different Z
    planes. Ultrack expects each connected component within a timepoint to have
    its own unique ID, so this step makes labels safe to track without forcing
    any Z stitching.
    """

    arr = _coerce_integer_labels(np.asarray(frame))
    if arr.ndim not in (2, 3):
        raise ValueError(
            "Each timepoint must be a 2D (Y, X) or 3D (Z, Y, X) label image."
        )

    positive_labels = np.unique(arr)
    positive_labels = positive_labels[positive_labels > 0]
    if positive_labels.size == 0:
        return np.zeros(arr.shape, dtype=np.uint16)

    structure = ndimage.generate_binary_structure(arr.ndim, 1)
    relabeled = np.zeros(arr.shape, dtype=np.uint64)
    next_label = 1

    for label_value in positive_labels:
        components, n_components = ndimage.label(arr == label_value, structure=structure)
        if n_components == 0:
            continue
        for component_id in range(1, n_components + 1):
            relabeled[components == component_id] = next_label
            next_label += 1

    return relabeled.astype(_choose_label_dtype(relabeled), copy=False)


def make_labels_trackable(labels: LabelSeriesInput) -> tuple[np.ndarray, tuple[int, ...]]:
    """Normalize a time series of labels and make every component ID unique."""

    label_array, time_indices = _normalize_label_series(labels)
    prepared = np.stack(
        [_make_frame_labels_trackable(label_array[t]) for t in range(label_array.shape[0])],
        axis=0,
    )
    return prepared, time_indices


def _normalize_label_series(labels: LabelSeriesInput) -> tuple[np.ndarray, tuple[int, ...]]:
    if isinstance(labels, np.ndarray):
        arr = _coerce_integer_labels(labels)
        if arr.ndim not in (3, 4):
            raise ValueError(
                "Time-lapse labels must have shape (T, Y, X) or (T, Z, Y, X) when passed as an ndarray."
            )
        time_indices = tuple(range(arr.shape[0]))
        return arr, time_indices

    if isinstance(labels, Mapping):
        if len(labels) == 0:
            raise ValueError("Tracking labels mapping is empty.")
        ordered = sorted((int(t), np.asarray(frame)) for t, frame in labels.items())
        time_indices = tuple(t for t, _ in ordered)
        arr = _stack_label_sequence([frame for _, frame in ordered])
        return arr, time_indices

    if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        arr = _stack_label_sequence([np.asarray(frame) for frame in labels])
        time_indices = tuple(range(arr.shape[0]))
        return arr, time_indices

    raise TypeError(
        "labels must be a numpy array, a sequence of per-timepoint arrays, "
        "or a mapping of time index to per-timepoint arrays."
    )


def build_ultrack_config(
    tracking_config: UltrackTrackingConfig | None = None,
    *,
    main_config: Any | None = None,
    working_dir: str | Path | None = None,
) -> Any:
    """Build and populate an Ultrack ``MainConfig`` for labels-only tracking."""

    api = _import_ultrack_api()
    settings = tracking_config or UltrackTrackingConfig()
    config = api.MainConfig() if main_config is None else _clone_main_config(main_config)

    data_cfg = _ensure_subconfig(config, "data_config", api.DataConfig)
    seg_cfg = _ensure_subconfig(config, "segmentation_config", api.SegmentationConfig)
    link_cfg = _ensure_subconfig(config, "linking_config", api.LinkingConfig)
    track_cfg = _ensure_subconfig(config, "tracking_config", api.TrackingConfig)

    resolved_working_dir = working_dir if working_dir is not None else settings.working_dir
    if resolved_working_dir is not None:
        working_dir_path = Path(resolved_working_dir).expanduser().resolve()
        working_dir_path.mkdir(parents=True, exist_ok=True)
        data_cfg.working_dir = working_dir_path

    if settings.database:
        data_cfg.database = str(settings.database)

    if settings.n_workers is not None:
        n_workers = int(settings.n_workers)
        data_cfg.n_workers = n_workers
        seg_cfg.n_workers = n_workers
        link_cfg.n_workers = n_workers

    if settings.min_area is not None:
        seg_cfg.min_area = int(settings.min_area)
    if settings.min_area_factor is not None:
        seg_cfg.min_area_factor = float(settings.min_area_factor)
    if settings.max_area is not None:
        seg_cfg.max_area = int(settings.max_area)
    seg_cfg.min_frontier = float(settings.min_frontier)
    if settings.threshold is not None:
        seg_cfg.threshold = float(settings.threshold)
    if settings.anisotropy_penalization is not None:
        seg_cfg.anisotropy_penalization = float(settings.anisotropy_penalization)

    if settings.max_distance is not None:
        link_cfg.max_distance = float(settings.max_distance)
    if settings.max_neighbors is not None:
        link_cfg.max_neighbors = int(settings.max_neighbors)
    if settings.distance_weight is not None:
        link_cfg.distance_weight = float(settings.distance_weight)

    if settings.appear_weight is not None:
        track_cfg.appear_weight = float(settings.appear_weight)
    if settings.disappear_weight is not None:
        track_cfg.disappear_weight = float(settings.disappear_weight)
    if settings.division_weight is not None:
        track_cfg.division_weight = float(settings.division_weight)
    if settings.image_border_size is not None:
        track_cfg.image_border_size = tuple(int(v) for v in settings.image_border_size)
    if settings.window_size is not None:
        track_cfg.window_size = int(settings.window_size)
    if settings.overlap_size is not None:
        track_cfg.overlap_size = int(settings.overlap_size)
    if settings.solution_gap is not None:
        track_cfg.solution_gap = float(settings.solution_gap)
    if settings.time_limit is not None:
        track_cfg.time_limit = int(settings.time_limit)
    if settings.solver_name is not None:
        track_cfg.solver_name = str(settings.solver_name)
    if settings.n_threads is not None:
        track_cfg.n_threads = int(settings.n_threads)
    if settings.link_function is not None:
        track_cfg.link_function = str(settings.link_function)
    if settings.power is not None:
        track_cfg.power = float(settings.power)
    if settings.bias is not None:
        track_cfg.bias = float(settings.bias)

    return config


def _export_tracked_masks(
    api: SimpleNamespace,
    tracker: Any,
    main_config: Any,
    tracks_df: Any,
    settings: UltrackTrackingConfig,
) -> np.ndarray:
    kwargs = {
        "tracks_df": tracks_df,
        "store_or_path": settings.export_store_or_path,
        "chunks": None if settings.export_chunks is None else tuple(settings.export_chunks),
        "overwrite": bool(settings.export_overwrite),
    }

    if callable(api.tracks_to_zarr):
        tracked = api.tracks_to_zarr(main_config, **kwargs)
    else:
        to_zarr = getattr(tracker, "to_zarr", None)
        if not callable(to_zarr):
            raise RuntimeError(
                "Ultrack did not expose `tracks_to_zarr` or `Tracker.to_zarr`; "
                "cannot export tracked masks."
            )
        try:
            tracked = to_zarr(**kwargs)
        except TypeError:
            try:
                tracked = to_zarr(tracks_df)
            except TypeError:
                tracked = to_zarr()

    arr = np.asarray(tracked).copy()
    return arr.astype(_choose_label_dtype(arr), copy=False)


def track_labels_with_ultrack(
    labels: LabelSeriesInput,
    tracking_config: UltrackTrackingConfig | None = None,
    *,
    main_config: Any | None = None,
) -> UltrackTrackingResult:
    """Track existing label masks over time using Ultrack.

    Parameters
    ----------
    labels
        Existing instance labels with time as the first axis. Accepted formats:
        ``(T, Y, X)``, ``(T, Z, Y, X)``, ``Sequence[(Y, X)/(Z, Y, X)]``, or
        ``{time_index: labels}``.
    tracking_config
        High-level Ultrack settings. The default is tuned for labels-only
        tracking from existing Cellpose masks.
    main_config
        Optional existing Ultrack ``MainConfig``. When provided, it is cloned
        and then overwritten by values from ``tracking_config``.
    """

    settings = tracking_config or UltrackTrackingConfig()
    labels_array, time_indices = make_labels_trackable(labels)
    api = _import_ultrack_api()

    temp_dir_cm = None
    working_dir_value = settings.working_dir
    if working_dir_value is None:
        temp_dir_cm = tempfile.TemporaryDirectory(prefix="cellpose_ultrack_")
        working_dir_value = temp_dir_cm.name

    try:
        built_config = build_ultrack_config(
            settings,
            main_config=main_config,
            working_dir=working_dir_value,
        )
        tracker = api.Tracker(built_config)
        tracker.track(
            labels=labels_array,
            sigma=settings.sigma,
            scale=settings.scale,
            vector_field=settings.vector_field,
            overwrite=_normalize_overwrite(settings.overwrite),
            segment_kwargs=dict(settings.segment_kwargs),
            link_kwargs=dict(settings.link_kwargs),
            solve_kwargs=dict(settings.solve_kwargs),
        )
        tracks_df, lineage_graph = tracker.to_tracks_layer(
            include_parents=bool(settings.include_parents),
            include_node_ids=bool(settings.include_node_ids),
        )
        tracked_masks = _export_tracked_masks(
            api,
            tracker,
            built_config,
            tracks_df,
            settings,
        )
        working_dir = Path(getattr(built_config.data_config, "working_dir")).resolve()
    finally:
        if temp_dir_cm is not None:
            temp_dir_cm.cleanup()

    return UltrackTrackingResult(
        tracked_masks=tracked_masks,
        tracks_df=tracks_df,
        lineage_graph=lineage_graph,
        time_indices=time_indices,
        main_config=built_config,
        working_dir=working_dir,
    )


def track_masks_with_ultrack(
    labels: LabelSeriesInput,
    tracking_config: UltrackTrackingConfig | None = None,
    *,
    main_config: Any | None = None,
) -> UltrackTrackingResult:
    """Alias for :func:`track_labels_with_ultrack`."""

    return track_labels_with_ultrack(
        labels,
        tracking_config=tracking_config,
        main_config=main_config,
    )


def save_tracked_masks_by_time(
    tracked_masks: np.ndarray | UltrackTrackingResult,
    output_dir: str | Path,
    *,
    time_indices: Sequence[int] | None = None,
    compression: str = "zlib",
) -> dict[int, Path]:
    """Save tracked masks as one TIFF per timepoint."""

    if isinstance(tracked_masks, UltrackTrackingResult):
        result = tracked_masks
        array = np.asarray(result.tracked_masks)
        indices = result.time_indices
    else:
        array = _coerce_integer_labels(np.asarray(tracked_masks))
        if array.ndim not in (3, 4):
            raise ValueError(
                "tracked_masks must have shape (T, Y, X) or (T, Z, Y, X)."
            )
        if time_indices is None:
            indices = tuple(range(array.shape[0]))
        else:
            indices = tuple(int(t) for t in time_indices)

    if len(indices) != int(array.shape[0]):
        raise ValueError(
            "time_indices length must match the first axis of tracked_masks."
        )

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, Path] = {}

    for i, t in enumerate(indices):
        frame = np.asarray(array[i])
        dtype = _choose_label_dtype(frame)
        axes = "ZYX" if frame.ndim == 3 else "YX"
        suffix = "ZYX" if frame.ndim == 3 else "YX"
        out_path = out_dir / f"masks_T{int(t):04d}_{suffix}.tif"
        tifffile.imwrite(
            out_path,
            frame.astype(dtype, copy=False),
            compression=compression,
            metadata={"axes": axes},
        )
        written[int(t)] = out_path

    return written


__all__ = [
    "UltrackTrackingConfig",
    "UltrackTrackingResult",
    "build_ultrack_config",
    "make_labels_trackable",
    "track_labels_with_ultrack",
    "track_masks_with_ultrack",
    "save_tracked_masks_by_time",
]
