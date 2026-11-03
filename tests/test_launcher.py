"""Desktop app: link parsing, first-run state, and what the coordinator hands out."""

import io
import sys
import tarfile

import pytest

from launcher import setup_env as env


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("POOLFORGE_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.mark.parametrize("text, expected", [
    ("https://my-pool.onrender.com/join/X7K2AB", ("https://my-pool.onrender.com", "X7K2AB")),
    ("  http://192.168.1.5:8000/join/x7k2ab/ ", ("http://192.168.1.5:8000", "X7K2AB")),
    ("my-pool.onrender.com/join/X7K2AB", ("https://my-pool.onrender.com", "X7K2AB")),
    ("https://example.com/pools/join/ABC123", ("https://example.com/pools", "ABC123")),
    ("abc123", ("https://default.example", "ABC123")),
])
def test_parse_link(text, expected):
    assert env.parse_link(text, "https://default.example/") == expected


@pytest.mark.parametrize("text", ["", "hello", "https://x.com/join/ABC", "ftp://x.com/join/ABC123", "https://x.com/ABC123"])
def test_parse_link_rejects_junk(text):
    with pytest.raises(ValueError):
        env.parse_link(text, "https://default.example")


def test_bare_code_needs_a_known_server():
    with pytest.raises(ValueError, match="whole link"):
        env.parse_link("ABC123", "")


def test_node_id_is_stable_and_valid():
    first = env.node_id()
    assert first == env.node_id()
    assert all(c.isalnum() or c in "-_." for c in first)


def test_agent_payload_copied_once_per_version(tmp_path, monkeypatch, home):
    payload = tmp_path / "payload"
    (payload / "deploy" / "sample_apps").mkdir(parents=True)
    (payload / "agent.so").write_text("v1")
    monkeypatch.setattr(env, "PAYLOAD_DIR", payload)
    monkeypatch.setattr(env, "VERSION", "1.0.0")
    logs = []
    lib = env.ensure_agent(logs.append)
    assert (lib / "agent.so").read_text() == "v1" and logs
    (payload / "agent.so").write_text("v2")
    env.ensure_agent(logs.append)
    assert (lib / "agent.so").read_text() == "v1"  # same version: not copied again
    monkeypatch.setattr(env, "VERSION", "1.0.1")
    env.ensure_agent(logs.append)
    assert (lib / "agent.so").read_text() == "v2"  # app updated: new agent installed


def test_from_source_uses_the_repo(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "PAYLOAD_DIR", tmp_path / "missing")
    assert (env.ensure_agent(lambda m: None) / "agent" / "cli.py").exists()


def test_agent_command_joins_with_fixed_node_id():
    cmd = env.agent_command("https://p.example", "ABC123")
    assert cmd[0] == str(env.python_exe())
    assert cmd[cmd.index("join") + 1] == "ABC123"
    assert cmd[cmd.index("--server") + 1] == "https://p.example"
    assert cmd[cmd.index("--node-id") + 1] == env.node_id()


def test_compiled_python_version_matches_ci():
    workflow = (env.REPO_ROOT / ".github" / "workflows" / "release-app.yml").read_text()
    assert f'python-version: "{env.PYTHON_VERSION.rsplit(".", 1)[0]}"' in workflow


# ------------------------------------------------------------------ coordinator side
async def test_source_download_off_by_default(coordinator, monkeypatch):
    monkeypatch.delenv("POOLFORGE_SERVE_SOURCE", raising=False)
    assert (await coordinator.http.get("/download/poolforge.tar.gz")).status_code == 404
    room = await coordinator.create_room()
    info = (await coordinator.http.get(f"/api/rooms/{room['room_code']}")).json()
    assert "install_url" not in info


async def test_source_download_opt_in_has_no_coordinator(coordinator, monkeypatch):
    monkeypatch.setenv("POOLFORGE_SERVE_SOURCE", "1")
    r = await coordinator.http.get("/download/poolforge.tar.gz")
    assert r.status_code == 200
    names = tarfile.open(fileobj=io.BytesIO(r.content)).getnames()
    assert "poolforge/agent/cli.py" in names
    assert not any(n.startswith("poolforge/coordinator") for n in names)


async def test_room_info_lists_app_downloads(coordinator, monkeypatch):
    monkeypatch.setenv("POOLFORGE_APP_URL_WINDOWS", "https://dl.example/PoolForge.exe")
    monkeypatch.delenv("POOLFORGE_APP_URL_LINUX", raising=False)
    room = await coordinator.create_room()
    code = room["room_code"]
    info = (await coordinator.http.get(f"/api/rooms/{code}")).json()
    # An env var overrides its link; the other falls back to the public releases repo.
    assert info["downloads"] == {
        "windows": "https://dl.example/PoolForge.exe",
        "linux": "https://github.com/W0nder0fy0u/poolforge-app/releases/latest/download/PoolForge-linux",
    }
    assert info["join_url"].endswith(f"/join/{code}")
    assert env.parse_link(info["join_url"]) == (info["server"], code)


@pytest.mark.skipif(sys.platform not in ("win32", "linux"), reason="the app supports Windows and Linux")
def test_python_url_for_this_os():
    url = env.PYTHON_URL.format(release=env.PYTHON_RELEASE, version=env.PYTHON_VERSION, target=env.PYTHON_TARGETS[env.os_name()])
    assert url.startswith("https://github.com/astral-sh/python-build-standalone/") and env.PYTHON_VERSION in url


async def test_app_downloads_always_offered(coordinator, monkeypatch):
    monkeypatch.delenv("POOLFORGE_APP_URL_WINDOWS", raising=False)
    monkeypatch.setenv("POOLFORGE_APP_URL_LINUX", "")
    links = (await coordinator.http.get("/api/downloads")).json()
    assert set(links) == {"windows", "linux"}
    assert links["windows"].endswith("/releases/latest/download/PoolForge.exe")
    assert links["linux"].endswith("/releases/latest/download/PoolForge-linux")
