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
