#!/bin/bash

# Format the codebase using ruff and black
ruff format fluxway/ tests/ setup.py
black --fast fluxway/ tests/ setup.py
ruff check fluxway/ tests/ setup.py --fix
