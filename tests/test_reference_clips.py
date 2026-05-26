"""Tests for vnl_playground.tasks.reference_clips."""

def test_extract_clip_names_falls_back_when_regex_collapses_to_duplicates():
    """If the regex collapses all snips_order entries to the same prefix
    (e.g. 'clip_0000' -> 'clip' for every entry), return the raw names
    verbatim rather than 1730 copies of 'clip'."""
    from vnl_playground.tasks.reference_clips import ReferenceClips

    rc = ReferenceClips.__new__(ReferenceClips)  # bypass __init__
    config = {"model": {"snips_order": ["clip_0000", "clip_0001", "clip_0002"]}}
    names = rc._extract_clip_names(config)
    # With the patched fallback, expect unique names
    assert names is not None
    assert list(names) == ["clip_0000", "clip_0001", "clip_0002"]


import os
import pytest
import numpy as np

FLY_LEGACY_PATH = "/home/talmolab/Desktop/SalkResearch/track-mjx/data/fly/fly_reference_clip.legacy.h5"


@pytest.mark.skipif(not os.path.exists(FLY_LEGACY_PATH), reason="fly legacy H5 not present")
def test_fly_legacy_loads_via_unified_loader():
    from vnl_playground.tasks.reference_clips import ReferenceClips
    clips = ReferenceClips(FLY_LEGACY_PATH, n_frames_per_clip=600)
    assert clips.qpos.shape == (1730, 600, 43)
    assert clips.qvel.shape == (1730, 600, 42)
    assert clips.clip_names is not None
    assert len(clips.clip_names) == 1730
    # Synthetic clip names preserved verbatim by the fallback
    assert clips.clip_names[0] == "clip_0000"
    # Body lookup works
    walker_xpos = clips.body_xpos("walker")
    assert walker_xpos.shape == (1730, 600, 3)
