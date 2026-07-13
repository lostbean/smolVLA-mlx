"""Unit tests for the held-out episode split (parity_gate.split) -- TDD
directive step (1): "the held-out split logic, tested against the real
dataset's real episode count/structure".

`test_split_matches_real_dataset_episode_count` is the "tested against the
real dataset's real episode/structure" half: it reads the real, already
locally-cached `lerobot/svla_so101_pickplace` snapshot's own
`meta/info.json` for its real `total_episodes` (50, at the time this test
was written) and checks the split behaves sensibly against that real
number -- skipped (not failed) if the real dataset isn't cached locally,
since this repo's fast test gate must not require a network fetch.
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from finetune_job.parity_gate.split import (
    EpisodeSplit,
    build_train_only_dataset,
    split_episodes,
)

REAL_DATASET_ROOT = Path(
    os.environ.get(
        "FINETUNE_DATASET_ROOT",
        "~/.cache/huggingface/lerobot/hub/datasets--lerobot--svla_so101_pickplace/"
        "snapshots/f641879e22172be7e8161d5e6c1503c2d2feb657",
    )
).expanduser()


def test_split_partitions_every_episode_exactly_once():
    split = split_episodes(50, n_holdout=6)

    assert len(split.holdout_episodes) == 6
    assert len(split.train_episodes) == 44
    assert set(split.train_episodes) | set(split.holdout_episodes) == set(range(50))
    assert set(split.train_episodes) & set(split.holdout_episodes) == set()


def test_split_holdout_spans_the_full_index_range_not_just_the_tail():
    split = split_episodes(50, n_holdout=6)

    # A held-out set concentrated at the trailing end would all be >= 44;
    # spanning the full range means the minimum stays well below that.
    assert min(split.holdout_episodes) < 25
    assert max(split.holdout_episodes) >= 25


def test_split_is_deterministic():
    a = split_episodes(50, n_holdout=6)
    b = split_episodes(50, n_holdout=6)
    assert a == b


def test_split_rejects_n_holdout_covering_the_whole_dataset():
    with pytest.raises(ValueError):
        split_episodes(10, n_holdout=10)


def test_split_rejects_zero_or_negative_n_holdout():
    with pytest.raises(ValueError):
        split_episodes(10, n_holdout=0)
    with pytest.raises(ValueError):
        split_episodes(10, n_holdout=-1)


def test_split_rejects_non_positive_total_episodes():
    with pytest.raises(ValueError):
        split_episodes(0, n_holdout=1)


@pytest.mark.skipif(
    not REAL_DATASET_ROOT.is_dir(),
    reason=f"real dataset not cached locally at {REAL_DATASET_ROOT}",
)
def test_split_matches_real_dataset_episode_count():
    info = json.loads((REAL_DATASET_ROOT / "meta" / "info.json").read_text())
    total_episodes = info["total_episodes"]

    split = split_episodes(total_episodes, n_holdout=6)

    assert split.total_episodes == total_episodes
    assert len(split.train_episodes) + len(split.holdout_episodes) == total_episodes


@pytest.mark.skipif(
    not REAL_DATASET_ROOT.is_dir(),
    reason=f"real dataset not cached locally at {REAL_DATASET_ROOT}",
)
def test_build_train_only_dataset_contains_only_train_episode_rows(tmp_path):
    """Real-data structural check: materializing the train-only copy
    against the real dataset produces a real, smaller parquet containing
    exactly the training subset's frames -- never a held-out episode's
    row, never a fabricated one. `episode_index` values are RENUMBERED
    contiguously from 0 (see `build_train_only_dataset`'s own docstring
    for why -- LeRobot's own dataloader indexes `meta/episodes`
    positionally, not by the original episode_index value, so a real
    training run against a sparsely-numbered copy crashes), so this
    checks frame COUNTS and ORIGINAL-episode disjointness rather than
    comparing raw episode_index values directly."""
    info = json.loads((REAL_DATASET_ROOT / "meta" / "info.json").read_text())
    total_episodes = info["total_episodes"]
    split = split_episodes(total_episodes, n_holdout=6)

    dest = tmp_path / "train_only"
    build_train_only_dataset(REAL_DATASET_ROOT, dest, split)

    out_info = json.loads((dest / "meta" / "info.json").read_text())
    assert out_info["total_episodes"] == len(split.train_episodes)

    data_df = pd.read_parquet(dest / "data" / "chunk-000" / "file-000.parquet")
    # Renumbered contiguously: exactly 0..len(train_episodes)-1.
    assert set(data_df["episode_index"].unique()) == set(
        range(len(split.train_episodes))
    )
    # The renumbered global `index` column is also contiguous from 0 --
    # required by LeRobot's own dataloader (see build_train_only_dataset's
    # docstring on why the original global offsets no longer apply).
    assert sorted(data_df["index"].tolist()) == list(range(len(data_df)))

    episodes_df = pd.read_parquet(
        dest / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    assert set(episodes_df["episode_index"].unique()) == set(
        range(len(split.train_episodes))
    )
    # dataset_from_index/dataset_to_index are recomputed to match the
    # renumbered data table's real row ranges.
    for _, row in episodes_df.iterrows():
        ep_rows = data_df[data_df["episode_index"] == row["episode_index"]]
        assert row["dataset_from_index"] == ep_rows["index"].min()
        assert row["dataset_to_index"] == ep_rows["index"].max() + 1

    # Every real source row for a training episode survives the filter
    # (a real subset, not a truncated/corrupted one) -- same COUNT as the
    # source's own training-episode frame count.
    source_data_df = pd.read_parquet(
        REAL_DATASET_ROOT / "data" / "chunk-000" / "file-000.parquet"
    )
    expected_frame_count = source_data_df[
        source_data_df["episode_index"].isin(split.train_episodes)
    ].shape[0]
    assert data_df.shape[0] == expected_frame_count

    # Video files are copied unchanged -- a held-out episode's frames are
    # still physically decodable from them, but no metadata row in
    # dest_root points at a held-out episode's timestamp range, so no
    # code path in either trainer can reach them (see split.py's module
    # docstring).
    assert (dest / "videos").is_dir()


def test_episode_split_is_a_frozen_value():
    split = split_episodes(20, n_holdout=4)
    assert isinstance(split, EpisodeSplit)
    with pytest.raises(Exception):
        split.train_episodes = ()  # type: ignore[misc]
