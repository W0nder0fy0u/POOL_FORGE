# PyInstaller spec: builds a standalone `poolforge-agent` executable
# (poolforge-agent.exe on Windows) so friends can join without installing Python.
#
#   pip install pyinstaller==6.22.3
#   pyinstaller deploy/poolforge-agent.spec        # run from the repo root
#   -> dist/poolforge-agent/poolforge-agent(.exe)
#
# Usage of the result is the same as the CLI:
#   poolforge-agent.exe join X7K2AB --server https://my-pool.onrender.com
#
# Notes:
#  * Build on Windows to get a Windows .exe (PyInstaller does not cross-compile).
#  * Install the CPU-only torch first, or the bundle includes GPU libraries (~4 GB):
#      pip install torch --index-url https://download.pytorch.org/whl/cpu
#    With CPU torch the folder is roughly 1 GB; zip it to share.
#  * Tested on Linux: the built agent joined a pool, trained a stage next to a
#    normal Python agent, and hosted the static-site sample.
#  * One-folder mode (not one-file): torch is large, and one-file would unpack
#    ~1 GB to a temp folder on every start.
#  * The static-site sample works from the exe; the Flask sample needs a real
#    Python install with Flask, and docker apps need Docker.

import os

root = os.path.abspath(os.path.join(SPECPATH, ".."))

a = Analysis(
    [os.path.join(root, "agent", "cli.py")],
    pathex=[root],
    datas=[(os.path.join(root, "deploy", "sample_apps"), os.path.join("deploy", "sample_apps"))],
    hiddenimports=["agent.worker", "agent.trainer", "agent.hosting", "common.crypto", "websockets.asyncio.client"],
    excludes=["coordinator", "fastapi", "uvicorn", "matplotlib", "tkinter", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="poolforge-agent",
    console=True,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="poolforge-agent", upx=False)
