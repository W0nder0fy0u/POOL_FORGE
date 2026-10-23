.PHONY: test dev clean lint

VENV := .venv
ifeq ($(OS),Windows_NT)
    BIN := $(VENV)/Scripts
    PY := $(BIN)/python.exe
    PYTEST := $(BIN)/pytest.exe
else
    BIN := $(VENV)/bin
    PY := $(BIN)/python
    PYTEST := $(BIN)/pytest
endif

test:
	@if [ -f "$(PYTEST)" ]; then \
		$(PYTEST) -v tests; \
	else \
		python3 -m pytest -v tests || pytest -v tests; \
	fi

dev:
	@if [ -f "$(PY)" ]; then \
		$(PY) -m coordinator.server; \
	else \
		python3 -m coordinator.server; \
	fi

clean:
	rm -rf .pytest_cache __pycache__ **/__pycache__ *.egg-info agent/*.egg-info build dist
