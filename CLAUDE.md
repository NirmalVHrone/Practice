# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

This is a fresh Python project repository. No source code, dependencies, or tooling configuration exists yet.

## Python Tooling (based on `.gitignore`)

The `.gitignore` is pre-configured for a Python project with support for:
- **Package managers**: pip, poetry, pdm, pipenv, uv, pixi
- **Testing**: pytest, coverage, tox, hypothesis
- **Type checkers**: mypy, pyre, pytype
- **Linter/formatter**: ruff
- **Web frameworks**: Django, Flask, Scrapy
- **Notebooks**: Jupyter, Marimo

Once project structure is established, typical commands will be:
- `python -m pytest` — run tests
- `python -m pytest tests/test_foo.py::test_bar` — run a single test
- `ruff check .` — lint
- `ruff format .` — format
