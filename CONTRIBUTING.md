# Contributing to Fluxway

[中文版](CONTRIBUTING_zh.md)

This guide covers development setup, local checks, and pull request requirements.

## Development environment

Use Linux and Python 3.12 or 3.13. Install dependencies in a dedicated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[jax,profiler,visualization]' --group test --group lint --group build
```

## Local checks

Run formatting checks and portable CPU tests from the repository root:

```bash
bash scripts/format.sh --check
JAX_PLATFORMS=cpu python -m pytest -ra --strict-markers --strict-config \
  --ignore=tests/checkpointer/test_retention.py \
  --ignore=tests/example/test_sparse_checkpoint_e2e.py
```

Use `bash scripts/format.sh` to apply formatting fixes. The excluded files need
the optional native `sparse_embedding` and Lance environment; changes to those
features also need validation where those dependencies are available.

Use a fresh temporary directory for package checks:

```bash
mkdir -p tmp
package_check_dir=$(mktemp -d "$PWD/tmp/package-check.XXXXXX")
python -m build --outdir "$package_check_dir/dist"
python -m twine check --strict "$package_check_dir"/dist/*
python scripts/distribution.py check "$package_check_dir/dist"
python -m pytest --confcutdir=tests/packaging tests/packaging
```

See [releasing](docs/releasing.md) for isolated installation and full release
validation. Passing locally on one Python version does not replace the CI
Python 3.12/3.13 matrix.

## Pull requests

- Focus each change on one problem. Describe its trigger, resulting behavior,
  and validation.
- Provide appropriate tests for behavior changes and update both language editions.
- Include migration steps for breaking API changes.
- The final commit must pass every required CI check before merge.

## Documentation requirements

- Prefer Chinese prose and diagram labels in Chinese editions, retaining API,
  type, field, and command names. Both editions must cover the same contract.
- Use Mermaid sequence diagrams for complex flows. Show concurrency, waits,
  cancellation, backpressure, commit, and failure boundaries. Use tables or
  structural diagrams for static relationships.
- Distinguish call return, task completion, receiver admission, and durable
  commit. Check ownership and message ordering against the current implementation.
- State example dependencies, inputs, expected results, and limits. Local protocol
  checks do not establish distributed recovery or performance.
- Check links, fenced code blocks, and diagram rendering after edits. See the
  [documentation index](docs/README.md) for diagram conventions and flow navigation.

## CI artifacts

The [nightly workflow](.github/workflows/nightly.yml) reuses the complete CI
pipeline and retains verified wheels and source distributions in the
`python-distributions` artifact for 14 days. It can be dispatched manually once
present on `main`. Pull requests changing CI or the nightly workflow also run
nightly validation. See [releasing](docs/releasing.md) for tags, Trusted Publishing,
and release ordering.
