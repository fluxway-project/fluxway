#!/bin/bash

# Format the codebase using ruff
ruff format fluxway/ tests/ setup.py
ruff check fluxway/ tests/ setup.py --fix
