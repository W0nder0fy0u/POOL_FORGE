"""First-run setup for the PoolForge desktop app (no GUI code here, so it is testable).

The app download is small. The first time it runs it fills a private folder
(%LOCALAPPDATA%\\PoolForge on Windows, ~/.local/share/poolforge on Linux):

    python/   a portable Python (python-build-standalone), so friends need no Python
    lib/      the agent, compiled to native modules (no .py source is shipped)
    work/     checkpoints and datasets
    state.json / config.json

PyTorch comes straight from download.pytorch.org: the CUDA build when `nvidia-smi`
finds an NVIDIA GPU, the CPU build otherwise. Later runs skip every step that is
already done, so they start in seconds.
"""

import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlparse

try:  # written by deploy/build_app.py; absent when running from source
    from launcher import _build
except ImportError:  # pragma: no cover - depends on how the app was built
    _build = None

VERSION = getattr(_build, "VERSION", "dev")
DEFAULT_SERVER = getattr(_build, "DEFAULT_SERVER", "")

# Must match the Python that deploy/build_app.py compiles the agent with (3.12).
PYTHON_RELEASE = "20260623"
PYTHON_VERSION = "3.12.13"
PYTHON_TARGETS = {"windows": "x86_64-pc-windows-msvc", "linux": "x86_64-unknown-linux-gnu"}
PYTHON_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/"
              "{release}/cpython-{version}%2B{release}-{target}-install_only_stripped.tar.gz")

TORCH = ["torch==2.12.1", "torchvision==0.27.1"]
TORCH_INDEX = "https://download.pytorch.org/whl/{variant}"
# Same pins as requirements.txt (agent side only). flask lets the node host the Flask sample.
DEPS = ["numpy==2.5.4", "pydantic==2.13.4", "websockets==16.0", "psutil==7.2.2", "cryptography==50.0.2", "flask==3.1.3"]

ROOM_CODE = re.compile(r"^[A-Z0-9]{6}$")
REPO_ROOT = Path(__file__).resolve().parent.parent
PAYLOAD_DIR = Path(__file__).resolve().parent / "payload"  # bundled into the app by deploy/build_app.py

Log = Callable[[str], None]
Progress = Callable[[Optional[float]], None]  # 0..1, or None for "busy, unknown how long"


class SetupError(Exception):
    """Something the user can act on (no internet, unsupported OS, ...)."""


def os_name() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    raise SetupError(f"PoolForge's app supports Windows and Linux, not {sys.platform}.")


def data_dir() -> Path:
    if os.environ.get("POOLFORGE_HOME"):
        return Path(os.environ["POOLFORGE_HOME"])
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "PoolForge"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "poolforge"


def parse_link(text: str, default_server: str = "") -> Tuple[str, str]:
    """'https://host/join/X7K2AB' -> ('https://host', 'X7K2AB'). A bare code uses default_server."""
    text = text.strip().rstrip("/")
    if ROOM_CODE.match(text.upper()):
        if not default_server:
            raise ValueError("Paste the whole link (it starts with https://), not just the code.")
        return default_server.rstrip("/"), text.upper()
    url = urlparse(text if "://" in text else f"https://{text}")
    match = re.search(r"^(.*)/join/([A-Za-z0-9]{6})$", url.path)
    if url.scheme not in ("http", "https") or not url.netloc or not match:
        raise ValueError("That doesn't look like a PoolForge link. It should end in /join/ and a 6-character code.")
    return f"{url.scheme}://{url.netloc}{match.group(1)}", match.group(2).upper()


# ------------------------------------------------------------------ small files
def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_config() -> dict:
    return _read_json(data_dir() / "config.json")


def save_config(**changes) -> dict:
    cfg = {**load_config(), **changes}
    _write_json(data_dir() / "config.json", cfg)
    return cfg


def node_id() -> str:
    """A fixed name per laptop, so a node that restarts can resume its part of a job."""
    cfg = load_config()
    if not cfg.get("node_id"):
        host = re.sub(r"[^A-Za-z0-9_.-]", "-", socket.gethostname())[:20].strip("-.") or "laptop"
        cfg = save_config(node_id=f"{host}-{secrets.token_hex(2)}")
    return cfg["node_id"]


# ------------------------------------------------------------------ steps
def python_exe() -> Path:
    root = data_dir() / "python"
    return root / "python.exe" if sys.platform == "win32" else root / "bin" / "python3"


def has_nvidia_gpu() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=10, **_no_window())
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and "GPU" in out.stdout


def _download(url: str, dest: Path, progress: Progress) -> None:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, open(dest, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(1 << 16):
                out.write(chunk)
                done += len(chunk)
                progress(done / total if total else None)
    except OSError as exc:
        raise SetupError(f"Download failed ({exc}). Check your internet connection and try again.") from exc


def ensure_python(log: Log, progress: Progress) -> None:
    state = _read_json(data_dir() / "state.json")
    if python_exe().exists() and state.get("python") == PYTHON_VERSION:
        return
    log(f"Downloading Python {PYTHON_VERSION} (about 35 MB, only once)...")
    target = PYTHON_TARGETS[os_name()]
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(root / "python", ignore_errors=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        archive = Path(tmp) / "python.tar.gz"
        _download(PYTHON_URL.format(release=PYTHON_RELEASE, version=PYTHON_VERSION, target=target), archive, progress)
        log("Unpacking Python...")
        progress(None)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")  # the archive holds one top-level python/ folder
        os.replace(Path(tmp) / "python", root / "python")
    _write_json(root / "state.json", {**state, "python": PYTHON_VERSION, "deps": None})


def _pip(args: List[str], log: Log, progress: Progress) -> None:
    """Run pip in the private Python and turn its `--progress-bar raw` lines into progress."""
    cmd = [str(python_exe()), "-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location",
           "--progress-bar", "raw", *args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                            **_no_window())
    tail: List[str] = []
    for line in proc.stdout:
        line = line.rstrip()
        m = re.match(r"Progress (\d+) of (\d+)", line)
        if m:
            progress(int(m.group(1)) / max(int(m.group(2)), 1))
            continue
        if line:
            tail = (tail + [line])[-15:]
            if line.startswith(("Collecting", "Downloading", "Installing", "Successfully")):
                log(line)
                if line.startswith("Installing"):
                    progress(None)
    if proc.wait() != 0:
        log("\n".join(tail))
        raise SetupError("Installing PoolForge's parts failed (details above). Check your internet connection and try again.")


def ensure_deps(log: Log, progress: Progress) -> str:
    """Install PyTorch (GPU or CPU build) and the agent's other libraries. Returns the variant."""
    # POOLFORGE_TORCH=cpu skips the big GPU download even when there is an NVIDIA GPU
    variant = os.environ.get("POOLFORGE_TORCH") or ("cu126" if has_nvidia_gpu() else "cpu")
    wanted = hashlib.sha256(json.dumps([variant, TORCH, DEPS]).encode()).hexdigest()[:16]
    state = _read_json(data_dir() / "state.json")
    if state.get("deps") == wanted:
        return variant
    size = "about 2.5 GB" if variant != "cpu" else "about 200 MB"
    log(f"Installing PyTorch for {'your NVIDIA GPU' if variant != 'cpu' else 'CPU'} ({size}, only once)...")
    _pip([*TORCH, "--index-url", TORCH_INDEX.format(variant=variant)], log, progress)
    log("Installing the remaining libraries...")
    _pip(DEPS, log, progress)
    _write_json(data_dir() / "state.json", {**_read_json(data_dir() / "state.json"), "deps": wanted})
    return variant


def has_agent(folder: Path) -> bool:
    """True if the folder holds the compiled agent module (agent.*.pyd on Windows, agent.*.so on Linux)."""
    return folder.is_dir() and any(f.suffix in (".pyd", ".so") for f in folder.glob("agent.*"))


def ensure_agent(log: Log) -> Path:
    """Copy the compiled agent out of the app (once per app version). Returns the folder for PYTHONPATH."""
    if not PAYLOAD_DIR.is_dir():  # running from source (developers): use the repo itself
        return REPO_ROOT
    if not has_agent(PAYLOAD_DIR):
        raise SetupError("This copy of PoolForge is missing its agent. Download the latest version and try again.")
    lib = data_dir() / "lib"
    state = _read_json(data_dir() / "state.json")
    if state.get("agent") != VERSION or not has_agent(lib):
        log("Installing the PoolForge agent...")
        shutil.rmtree(lib, ignore_errors=True)
        shutil.copytree(PAYLOAD_DIR, lib)
        _write_json(data_dir() / "state.json", {**state, "agent": VERSION})
    return lib


def prepare(log: Log, progress: Progress) -> Path:
    """Run every setup step that isn't done yet. Returns the agent's PYTHONPATH folder."""
    os_name()
    ensure_python(log, progress)
    variant = ensure_deps(log, progress)
    lib = ensure_agent(log)
    log(f"Ready (PoolForge {VERSION}, {'GPU' if variant != 'cpu' else 'CPU'} build of PyTorch).")
    return lib


def agent_command(server: str, code: str) -> List[str]:
    work = data_dir() / "work"
    work.mkdir(parents=True, exist_ok=True)
    return [str(python_exe()), "-u", "-c", "import sys; from agent.cli import main; sys.exit(main())",
            "join", code, "--server", server, "--node-id", node_id(), "--workdir", str(work)]


def start_agent(server: str, code: str, lib: Path) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(lib), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    return subprocess.Popen(agent_command(server, code), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors="replace", env=env, cwd=str(data_dir() / "work"), **_no_window())


def stop_agent(proc: subprocess.Popen) -> None:
    """Ask the agent to leave the pool cleanly, then make sure it is gone."""
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":  # a windowless child can't get Ctrl+C
            proc.terminate()
        else:
            proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        proc.kill()


def _no_window() -> dict:
    """On Windows, don't flash a console window for each child process."""
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
