from pathlib import Path

import numpy as np

from cellpose.ultrack_tracking import (
    UltrackTrackingConfig,
    build_ultrack_config,
    make_labels_trackable,
    save_tracked_masks_by_time,
    track_labels_with_ultrack,
)


def _make_fake_api(record):
    class FakeDataConfig:
        def __init__(self):
            self.n_workers = 1
            self.working_dir = Path(".")
            self.database = "sqlite"

    class FakeSegmentationConfig:
        def __init__(self):
            self.n_workers = 1
            self.min_area = 100
            self.min_area_factor = 4.0
            self.max_area = 1_000_000
            self.min_frontier = 0.0
            self.threshold = 0.5
            self.anisotropy_penalization = 0.0

    class FakeLinkingConfig:
        def __init__(self):
            self.n_workers = 1
            self.max_distance = 15.0
            self.max_neighbors = 5
            self.distance_weight = 0.0

    class FakeTrackingConfig:
        def __init__(self):
            self.appear_weight = -0.001
            self.disappear_weight = -0.001
            self.division_weight = -0.001
            self.image_border_size = None
            self.window_size = None
            self.overlap_size = 1
            self.solution_gap = 0.001
            self.time_limit = 36000
            self.solver_name = ""
            self.n_threads = -1
            self.link_function = "power"
            self.power = 4.0
            self.bias = -0.0

    class FakeMainConfig:
        def __init__(self):
            self.data_config = FakeDataConfig()
            self.segmentation_config = FakeSegmentationConfig()
            self.linking_config = FakeLinkingConfig()
            self.tracking_config = FakeTrackingConfig()

        def copy(self, deep=True):
            cloned = FakeMainConfig()
            cloned.data_config.n_workers = self.data_config.n_workers
            cloned.data_config.working_dir = self.data_config.working_dir
            cloned.data_config.database = self.data_config.database
            cloned.segmentation_config.min_frontier = (
                self.segmentation_config.min_frontier
            )
            cloned.linking_config.max_distance = self.linking_config.max_distance
            cloned.tracking_config.window_size = self.tracking_config.window_size
            return cloned

    class FakeTracker:
        def __init__(self, config):
            record["tracker_config"] = config
            self.config = config

        def track(self, **kwargs):
            record["track_kwargs"] = kwargs

        def to_tracks_layer(self, **kwargs):
            record["to_tracks_layer_kwargs"] = kwargs
            tracks_df = {"track_id": [11, 12], "node_id": [101, 102]}
            lineage_graph = {12: [11]}
            return tracks_df, lineage_graph

    def fake_tracks_to_zarr(main_config, **kwargs):
        record["tracks_to_zarr_main_config"] = main_config
        record["tracks_to_zarr_kwargs"] = kwargs
        labels = record["track_kwargs"]["labels"]
        tracked = np.where(labels > 0, labels + 100, 0)
        return tracked

    class FakeNamespace:
        MainConfig = FakeMainConfig
        Tracker = FakeTracker
        DataConfig = FakeDataConfig
        SegmentationConfig = FakeSegmentationConfig
        LinkingConfig = FakeLinkingConfig
        TrackingConfig = FakeTrackingConfig
        tracks_to_zarr = staticmethod(fake_tracks_to_zarr)

    return FakeNamespace()


def test_build_ultrack_config_applies_overrides(monkeypatch, tmp_path):
    import cellpose.ultrack_tracking as ut

    record = {}
    monkeypatch.setattr(ut, "_import_ultrack_api", lambda: _make_fake_api(record))

    config = build_ultrack_config(
        UltrackTrackingConfig(
            working_dir=tmp_path,
            n_workers=3,
            min_frontier=0.25,
            max_distance=42.0,
            window_size=5,
        )
    )

    assert config.data_config.working_dir == tmp_path.resolve()
    assert config.data_config.n_workers == 3
    assert config.segmentation_config.n_workers == 3
    assert config.linking_config.n_workers == 3
    assert config.segmentation_config.min_frontier == 0.25
    assert config.linking_config.max_distance == 42.0
    assert config.tracking_config.window_size == 5


def test_track_labels_with_ultrack_accepts_mapping(monkeypatch, tmp_path):
    import cellpose.ultrack_tracking as ut

    record = {}
    monkeypatch.setattr(ut, "_import_ultrack_api", lambda: _make_fake_api(record))

    labels = {
        5: np.array([[0, 1], [2, 0]], dtype=np.uint16),
        7: np.array([[0, 3], [0, 4]], dtype=np.uint16),
    }
    result = track_labels_with_ultrack(
        labels,
        UltrackTrackingConfig(
            working_dir=tmp_path,
            overwrite=True,
            sigma=1.5,
            max_distance=21.0,
        ),
    )

    assert result.time_indices == (5, 7)
    assert result.tracked_masks.shape == (2, 2, 2)
    assert np.array_equal(
        record["track_kwargs"]["labels"],
        np.stack([labels[5], labels[7]], axis=0),
    )
    assert record["track_kwargs"]["overwrite"] == "all"
    assert record["track_kwargs"]["sigma"] == 1.5
    assert record["to_tracks_layer_kwargs"] == {
        "include_parents": True,
        "include_node_ids": True,
    }
    assert record["tracks_to_zarr_kwargs"]["tracks_df"] == {
        "track_id": [11, 12],
        "node_id": [101, 102],
    }
    assert np.array_equal(
        result.tracked_masks,
        np.where(np.stack([labels[5], labels[7]], axis=0) > 0,
                 np.stack([labels[5], labels[7]], axis=0) + 100, 0),
    )


def test_make_labels_trackable_splits_reused_ids_across_z():
    labels = np.array(
        [
            [
                [[0, 1], [0, 0]],
                [[0, 0], [1, 0]],
            ]
        ],
        dtype=np.uint16,
    )

    prepared, time_indices = make_labels_trackable(labels)

    assert time_indices == (0,)
    positive = sorted(int(v) for v in np.unique(prepared) if v > 0)
    assert positive == [1, 2]


def test_save_tracked_masks_by_time_writes_expected_files(tmp_path):
    tracked = np.array(
        [
            [[0, 101], [102, 0]],
            [[0, 103], [0, 104]],
        ],
        dtype=np.uint32,
    )

    written = save_tracked_masks_by_time(
        tracked,
        tmp_path,
        time_indices=[3, 4],
        compression="zlib",
    )

    assert sorted(written) == [3, 4]
    assert written[3].name == "masks_T0003_YX.tif"
    assert written[4].name == "masks_T0004_YX.tif"
    assert written[3].is_file()
    assert written[4].is_file()
