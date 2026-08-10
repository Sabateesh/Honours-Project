from pathlib import Path
from comas.data import _session_id, collect_samples

def test_session_id_parses_prefix(tmp_path: Path):
    assert _session_id(tmp_path / "session1_001.png") == "session1"
    assert _session_id(tmp_path / "abc_xyz_001.png") == "abc"
    assert _session_id(tmp_path / "noprefix.png") == "noprefix"

def test_collect_handles_missing_folders(tmp_path: Path):
    (tmp_path / "no_copilot").mkdir()
    samples = collect_samples(tmp_path)
    assert samples == []