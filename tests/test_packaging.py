"""Phase 8: optional payload encryption end to end, and packaging sanity checks."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CFG = {"model": "tiny", "batch_size": 16, "epochs": 1, "steps_per_epoch": 6}


async def start(coordinator, run_agent, passphrases):
    room = await coordinator.create_room()
    code, h = room["room_code"], {"X-Admin-Token": room["admin_token"]}
    for i, p in enumerate(passphrases):
        run_agent(code, coordinator.url, node_id=f"e{i}", encrypt_passphrase=p)
    await coordinator.wait_nodes(code, len(passphrases))
    job = (await coordinator.http.post(f"/api/rooms/{code}/jobs", json=CFG, headers=h)).json()
    return await coordinator.wait_job(code, job["job_id"], timeout=60)


async def test_encrypted_training_through_relay(coordinator, run_agent):
    pytest.importorskip("cryptography")
    done = await start(coordinator, run_agent, ["same secret", "same secret"])
    assert done["status"] == "completed", done


@pytest.mark.parametrize("passphrases", [["one", "two"], ["one", None]])
async def test_mismatched_encryption_fails_clearly(coordinator, run_agent, passphrases):
    pytest.importorskip("cryptography")
    done = await start(coordinator, run_agent, passphrases)
    assert done["status"] == "failed" and "encrypt-passphrase" in done["detail"], done


def test_coordinator_requirements_have_no_torch():
    lines = [l for l in (ROOT / "requirements-coordinator.txt").read_text().splitlines() if l and not l.startswith("#")]
    assert not any(l.startswith("torch") for l in lines) and any(l.startswith("fastapi") for l in lines)


def test_dockerfile_copies_only_coordinator_side():
    text = (ROOT / "deploy" / "Dockerfile").read_text()
    assert "COPY coordinator/" in text and "COPY common/" in text
    assert "COPY agent/" not in text and "requirements-coordinator.txt" in text


def test_cli_entry_point_and_version(capsys):
    from agent.cli import main

    assert main(["version"]) == 0
    assert "PoolForge agent" in capsys.readouterr().out


def test_pyinstaller_spec_present():
    spec = (ROOT / "deploy" / "poolforge-agent.spec").read_text()
    assert "agent" in spec and "sample_apps" in spec
