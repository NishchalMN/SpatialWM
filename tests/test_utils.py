"""
Tests for spatialwm.utils — all GREEN now (glue is fully implemented).
"""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_returns_known_device_string(self):
        """get_device() returns one of the three valid device names."""
        from spatialwm.utils.device import get_device

        result = get_device()
        assert result in {"mps", "cuda", "cpu"}, f"Unknown device: {result!r}"

    def test_explicit_cpu(self):
        """Explicit pref='cpu' is returned verbatim."""
        from spatialwm.utils.device import get_device

        assert get_device("cpu") == "cpu"

    def test_explicit_mps(self):
        """Explicit pref='mps' is returned verbatim (caller's responsibility)."""
        from spatialwm.utils.device import get_device

        assert get_device("mps") == "mps"

    def test_explicit_cuda(self):
        """Explicit pref='cuda' is returned verbatim."""
        from spatialwm.utils.device import get_device

        assert get_device("cuda") == "cuda"

    def test_auto_prefers_mps_over_cpu(self):
        """On an Apple-Silicon machine, 'auto' should prefer mps."""
        import torch

        from spatialwm.utils.device import get_device

        result = get_device("auto")
        if torch.backends.mps.is_available():
            assert result == "mps"
        elif torch.cuda.is_available():
            assert result == "cuda"
        else:
            assert result == "cpu"


# ---------------------------------------------------------------------------
# load_config + require_keys
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path):
        """load_config parses a well-formed YAML file into a dict."""
        from spatialwm.utils.config import load_config

        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("name: test\nseed: 42\n")
        cfg = load_config(str(cfg_file))
        assert cfg["name"] == "test"
        assert cfg["seed"] == 42

    def test_nested_keys_preserved(self, tmp_path):
        """Nested YAML structures come back as nested dicts."""
        from spatialwm.utils.config import load_config

        cfg_file = tmp_path / "nested.yaml"
        cfg_file.write_text("outer:\n  inner: 7\n")
        cfg = load_config(str(cfg_file))
        assert cfg["outer"]["inner"] == 7

    def test_require_keys_passes_for_present_keys(self, tmp_path):
        """require_keys does not raise when all keys are present."""
        from spatialwm.utils.config import load_config, require_keys

        cfg_file = tmp_path / "full.yaml"
        cfg_file.write_text("a: 1\nb: 2\nc: 3\n")
        cfg = load_config(str(cfg_file))
        # Should not raise
        require_keys(cfg, ["a", "b", "c"])

    def test_require_keys_raises_for_missing(self, tmp_path):
        """require_keys raises KeyError listing missing keys."""
        from spatialwm.utils.config import load_config, require_keys

        cfg_file = tmp_path / "partial.yaml"
        cfg_file.write_text("a: 1\n")
        cfg = load_config(str(cfg_file))
        with pytest.raises(KeyError, match="missing"):
            require_keys(cfg, ["a", "z_missing"])

    def test_require_keys_empty_list_always_passes(self, tmp_path):
        """require_keys([]) never raises."""
        from spatialwm.utils.config import load_config, require_keys

        cfg_file = tmp_path / "empty_req.yaml"
        cfg_file.write_text("{}\n")
        cfg = load_config(str(cfg_file))
        require_keys(cfg, [])  # must not raise


# ---------------------------------------------------------------------------
# append_result
# ---------------------------------------------------------------------------

class TestAppendResult:
    def test_creates_file_and_header_on_first_write(self, tmp_path):
        """First call creates the CSV file with correct header and one data row."""
        from spatialwm.utils.results import append_result

        csv_path = str(tmp_path / "summary.csv")
        append_result({"run": "r1", "metric": 1.5}, csv_path)

        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["run"] == "r1"
        assert float(rows[0]["metric"]) == pytest.approx(1.5)

    def test_appends_second_row_with_union_header(self, tmp_path):
        """Second call with a new key expands the header; both rows are present."""
        from spatialwm.utils.results import append_result

        csv_path = str(tmp_path / "summary.csv")
        append_result({"run": "r1", "metric_a": 1.0}, csv_path)
        append_result({"run": "r2", "metric_b": 2.0}, csv_path)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        # Header is the union
        assert set(fieldnames) == {"run", "metric_a", "metric_b"}
        assert len(rows) == 2

        runs = {r["run"] for r in rows}
        assert "r1" in runs
        assert "r2" in runs

    def test_creates_parent_directories(self, tmp_path):
        """append_result creates intermediate directories if they don't exist."""
        from spatialwm.utils.results import append_result

        csv_path = str(tmp_path / "deep" / "nested" / "results.csv")
        append_result({"x": 1}, csv_path)
        assert os.path.exists(csv_path)

    def test_missing_key_in_later_row_leaves_blank(self, tmp_path):
        """A row that doesn't have all header keys gets blank cells, not an error."""
        from spatialwm.utils.results import append_result

        csv_path = str(tmp_path / "sparse.csv")
        append_result({"a": 1, "b": 2}, csv_path)
        append_result({"a": 3}, csv_path)           # no 'b'

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        # Second row's 'b' should be empty string (blank cell)
        assert rows[1]["b"] == "" or rows[1]["b"] is None


# ---------------------------------------------------------------------------
# seed_all
# ---------------------------------------------------------------------------

class TestSeedAll:
    def test_same_seed_yields_same_numpy_draws(self):
        """Calling seed_all with the same seed produces identical numpy sequences."""
        from spatialwm.utils.seed import seed_all

        seed_all(123)
        a = np.random.rand(10)

        seed_all(123)
        b = np.random.rand(10)

        np.testing.assert_array_equal(a, b)

    def test_different_seeds_yield_different_draws(self):
        """Different seeds produce different sequences (collision extremely unlikely)."""
        from spatialwm.utils.seed import seed_all

        seed_all(0)
        a = np.random.rand(10)

        seed_all(1)
        b = np.random.rand(10)

        assert not np.array_equal(a, b)

    def test_seed_all_does_not_raise(self):
        """seed_all runs without error (covers random + numpy + torch branches)."""
        from spatialwm.utils.seed import seed_all

        seed_all(999)  # must not raise
