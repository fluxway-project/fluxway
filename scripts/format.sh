#!/bin/bash

# Get the directory of the script and change to the project root
cd "$(dirname "$0")/.."

# Format the codebase using ruff
ruff format fluxway/ tests/ setup.py
ruff check fluxway/ tests/ setup.py --fix
