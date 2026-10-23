"""Scaffold tests for PoolForge environment and baseline requirements."""

import os
import sys
import pytest


def test_python_version():
    """Verify supported Python runtime (>= 3.10)."""
    assert sys.version_info >= (3, 10), "PoolForge requires Python 3.10 or newer."


def test_core_dependencies_importable():
    """Verify all pinned dependencies import cleanly."""
    import torch
    import numpy as np
    import fastapi
    import uvicorn
    import pydantic
    import websockets
    import psutil
    import cryptography

    assert torch.__version__ is not None
    assert np.__version__ is not None
    assert fastapi.__version__ is not None
    assert uvicorn.__version__ is not None
    assert pydantic.__version__ is not None
    assert websockets.__version__ is not None
    assert psutil.__version__ is not None
    assert cryptography.__version__ is not None


def test_common_package():
    """Verify common package is accessible."""
    import common
    assert common.__version__ == "0.1.0"


def test_agent_package():
    """Verify agent package is accessible."""
    import agent
    assert agent.__version__ == "0.1.0"


def test_coordinator_package():
    """Verify coordinator package is accessible."""
    import coordinator
    assert coordinator.__version__ == "0.1.0"


def test_cross_platform_pathing(tmp_path):
    """Verify file paths work across Windows and Linux without unix-only assumptions."""
    sample_dir = tmp_path / "poolforge_test"
    sample_dir.mkdir()
    sample_file = sample_dir / "test.txt"
    sample_file.write_text("poolforge cross platform", encoding="utf-8")
    assert sample_file.read_text(encoding="utf-8") == "poolforge cross platform"
    assert os.path.exists(os.path.normpath(str(sample_file)))
