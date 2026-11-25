import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest
from evorag import resources
from evorag.config import Settings
from evorag.errors import ResourceUnavailable
from evorag.resources import GIB, ResourceManager


def test_busy_and_release_after_exception() -> None:
    manager = ResourceManager(Settings(), lambda: 3 * GIB)

    def attempt() -> None:
        with manager.generation_slot():
            pass

    with pytest.raises(RuntimeError, match="job failed"), manager.generation_slot():
        with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(ResourceUnavailable):
            pool.submit(attempt).result(timeout=2)
        raise RuntimeError("job failed")
    attempt()


def test_ram_rejection_does_not_leak_slot() -> None:
    readings = iter((None, 3 * GIB - 1, 3 * GIB))
    manager = ResourceManager(Settings(), lambda: next(readings))
    for _ in range(2):
        with pytest.raises(ResourceUnavailable, match="RAM"), manager.generation_slot():
            pytest.fail("admission should fail")
    with manager.generation_slot():
        pass


def test_storage_budget(tmp_path: Path) -> None:
    manager = ResourceManager(Settings())
    with pytest.raises(ValueError):
        manager.check_storage(used_bytes=-1, additional_bytes=0, path=tmp_path)
    with pytest.raises(ResourceUnavailable, match="budget"):
        manager.check_storage(used_bytes=20 * GIB, additional_bytes=1, path=tmp_path)
    with patch.object(resources, "available_disk", side_effect=[None, 0, 1]):
        for _ in range(2):
            with pytest.raises(ResourceUnavailable, match="disk"):
                manager.check_storage(used_bytes=0, additional_bytes=1, path=tmp_path)
        manager.check_storage(used_bytes=0, additional_bytes=1, path=tmp_path)


def test_missing_telemetry(tmp_path: Path) -> None:
    with (
        patch("psutil.virtual_memory", side_effect=PermissionError),
        patch("shutil.disk_usage", side_effect=OSError),
        patch("shutil.which", return_value=None),
    ):
        assert resources.resource_report(tmp_path) == resources.ResourceReport(None, None, None)
    with patch("shutil.which", return_value="nvidia-smi"):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 2)):
            assert resources.nvidia_gpus() is None
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, "GPU A\n")):
            assert resources.nvidia_gpus() == ("GPU A",)
