"""Tests for vnl_playground.tasks.reference_clips."""

import os

import numpy as np
import pytest

from vnl_playground.tasks.reference_clips import ReferenceClips

FLY_LEGACY_PATH = "/home/talmolab/Desktop/SalkResearch/track-mjx/data/fly/fly_reference_clip.h5"


def _extract(snips_order):
    """Call _extract_clip_names directly without going through __init__."""
    rc = ReferenceClips.__new__(ReferenceClips)
    return rc._extract_clip_names({"model": {"snips_order": list(snips_order)}})


def test_extract_clip_names_falls_back_when_regex_collapses_to_one_label():
    """Synthetic clip_0000…clip_NNNN all extract to 'clip' — fallback returns
    raw names so clip identity is preserved."""
    names = _extract(["clip_0000", "clip_0001", "clip_0002"])
    assert list(names) == ["clip_0000", "clip_0001", "clip_0002"]


def test_extract_clip_names_keeps_extracted_form_for_multi_behavior_clips():
    """Rodent-style snips_order with multiple behaviors and duplicates per
    behavior (e.g. Walk_001, Walk_002, Run_001) must keep the extracted
    labels (['Walk', 'Walk', 'Run']) — fallback must NOT trigger just
    because set-size < list-length."""
    names = _extract(["Walk_001.p", "Walk_002.p", "Run_001.p"])
    assert list(names) == ["Walk", "Walk", "Run"]


def test_extract_clip_names_single_entry_keeps_extracted_form():
    """Single-entry snips_order: extraction works, fallback guard is
    len > 1 so the raw-name fallback never fires."""
    names = _extract(["Walk_001.p"])
    assert list(names) == ["Walk"]


def test_extract_clip_names_returns_none_when_snips_order_missing():
    rc = ReferenceClips.__new__(ReferenceClips)
    assert rc._extract_clip_names({}) is None
    assert rc._extract_clip_names({"model": {}}) is None


@pytest.mark.skipif(
    not os.path.exists(FLY_LEGACY_PATH), reason="fly legacy H5 not present"
)
def test_fly_legacy_loads_via_unified_loader():
    clips = ReferenceClips(FLY_LEGACY_PATH, n_frames_per_clip=600)
    assert clips.qpos.shape == (1730, 600, 43)
    assert clips.qvel.shape == (1730, 600, 42)
    assert clips.clip_names is not None
    assert len(clips.clip_names) == 1730
    # Synthetic clip names preserved verbatim by the fallback
    assert clips.clip_names[0] == "clip_0000"
    # Body lookup works for a uniquely-positioned body that appears in any
    # supported fly MJCF variant.
    head_xpos = clips.body_xpos("head")
    assert head_xpos.shape == (1730, 600, 3)
