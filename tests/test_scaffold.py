"""Phase 0: the environment is usable and the packages import."""

import sys


def test_python_version():
    assert sys.version_info >= (3, 10)


def test_packages_import():
    import agent
    import common
    import coordinator

    assert common.__version__ == agent.__version__ == coordinator.__version__


def test_coordinator_side_code_does_not_need_torch():
    """The coordinator plans splits with pure-Python spec code, so its Docker image can skip torch."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    files = list((root / "coordinator").glob("*.py")) + [root / "common" / f for f in ("specs.py", "model_zoo.py", "schemas.py", "apps.py")]
    for f in files:
        if not f.exists():
            continue
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                assert not any(n.split(".")[0] == "torch" for n in names), f"{f.name} imports torch"
                assert not any(n in ("common.models", "common.pipeline", "common.protocol") for n in names), f.name
