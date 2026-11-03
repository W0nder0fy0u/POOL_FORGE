"""Build the PoolForge desktop app: one small file friends download and double-click.

    pip install nuitka==4.3.4 patchelf==0.17.2.4 zstandard   # patchelf: Linux only
    python deploy/build_app.py --server https://my-pool.onrender.com
    -> dist-app/PoolForge.exe (Windows) or dist-app/PoolForge (Linux), about 20 MB

Must run with Python 3.12 (the app installs Python 3.12 on friends' laptops and the
compiled agent only loads in the same version). Build on Windows for the .exe and
on Linux for Linux; .github/workflows/release.yml does both.

What ships: the agent and common packages compiled to native code by Nuitka
(no .py source, no docstrings, no .pyi stubs), the sample apps, and the launcher. The coordinator
is never included. PyTorch is not bundled; the app installs it on first run.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist-app"
PAYLOAD = OUT / "payload"  # mapped to launcher/payload inside the app
BUILD_INFO = ROOT / "launcher" / "_build.py"


def nuitka(python: str, *args: str) -> None:
    # tools pip-installed next to this Python (patchelf) must be on PATH for Nuitka to find them
    env = {**os.environ, "PATH": os.pathsep.join([str(Path(python).parent), os.environ.get("PATH", "")])}
    subprocess.run([python, "-m", "nuitka", "--assume-yes-for-downloads", "--remove-output", *args],
                   cwd=ROOT, check=True, env=env)


def python_minor(python: str) -> str:
    return subprocess.run([python, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
                          capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="", help="pool server used when a friend types only the 6-character code")
    p.add_argument("--version", default=None, help="app version (default: the one in common/__init__.py)")
    p.add_argument("--agent-python", default=sys.executable,
                   help="Python 3.12 that compiles the agent modules; must match the runtime the app downloads "
                        "(default: this Python). The launcher window is built with this Python, so run build_app "
                        "with any Python whose Tkinter works.")
    args = p.parse_args()

    sys.path.insert(0, str(ROOT))
    from common import __version__
    from launcher.setup_env import PYTHON_VERSION

    want = PYTHON_VERSION.rsplit(".", 1)[0]
    if python_minor(args.agent_python) != want:
        sys.exit(f"--agent-python must be Python {want} (the version the app downloads), not {python_minor(args.agent_python)}.")
    version = args.version or __version__

    shutil.rmtree(PAYLOAD, ignore_errors=True)
    PAYLOAD.mkdir(parents=True)
    for package in ("agent", "common"):  # compiled against the downloaded runtime's ABI
        nuitka(args.agent_python, "--mode=module", package, f"--include-package={package}", "--no-pyi-file",
               "--python-flag=no_docstrings", f"--output-dir={PAYLOAD}")
    # hosting.py looks for the samples next to the package folders
    shutil.copytree(ROOT / "deploy" / "sample_apps", PAYLOAD / "deploy" / "sample_apps",
                    ignore=shutil.ignore_patterns("__pycache__"))
    leaked = [f for f in PAYLOAD.rglob("*") if f.suffix in (".py", ".pyi", ".c") and "sample_apps" not in f.parts]
    if leaked:
        sys.exit(f"Source files ended up in the payload: {leaked}")

    BUILD_INFO.write_text(f"# Written by deploy/build_app.py\nVERSION = {version!r}\nDEFAULT_SERVER = {args.server!r}\n")
    name = "PoolForge.exe" if sys.platform == "win32" else "PoolForge"
    try:
        nuitka(sys.executable, "--onefile", "--enable-plugin=tk-inter", "--include-package=launcher",
               "--nofollow-import-to=agent,common,coordinator,torch",
               # raw-dir, not data-dir: data-dir silently drops code files, i.e. the compiled .pyd/.so agent
               f"--include-raw-dir={PAYLOAD}=launcher/payload",
               "--windows-console-mode=disable", "--product-name=PoolForge", "--company-name=PoolForge",
               f"--file-version={version}", f"--product-version={version}",
               f"--output-dir={OUT}", f"--output-filename={name}", "launcher/app.py")
    finally:
        BUILD_INFO.unlink(missing_ok=True)
        shutil.rmtree(PAYLOAD, ignore_errors=True)
    print(f"Built {OUT / name} ({(OUT / name).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
