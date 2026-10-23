# PoolForge developer commands.
#   make setup      create .venv, install everything, build the dashboard
#   make test       run the pytest suite
#   make dev        run the coordinator (serves the dashboard on :8000)
#   make demo       coordinator + 2 capped agents on this machine
#   make dashboard  rebuild the React dashboard into dashboard/dist
#   make bench      run all benchmarks (CSV + PNG into benchmarks/results)
# On Windows without make, use scripts\*.ps1 or the raw commands below.

# Use the project venv if it exists, otherwise whatever python3 is on PATH.
ifeq ($(OS),Windows_NT)
    VENV_PY := .venv/Scripts/python.exe
else
    VENV_PY := .venv/bin/python
endif
PY := $(if $(wildcard $(VENV_PY)),$(VENV_PY),python3)

.PHONY: setup test dev demo dashboard bench docker clean

setup:
	python3 -m venv .venv
	$(VENV_PY) -m pip install -r requirements.txt
	$(VENV_PY) -m pip install -e .
	cd dashboard && npm install && npm run build

test:
	$(PY) -m pytest -q

dev:
	$(PY) -m coordinator.server

demo:
	bash scripts/dev_two_agents.sh

dashboard:
	cd dashboard && npm install && npm run build

bench:
	$(PY) -m benchmarks.run_all

docker:
	docker build -t poolforge-coordinator .

clean:
	rm -rf .pytest_cache checkpoints data poolforge.db* build dist *.egg-info
	find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
