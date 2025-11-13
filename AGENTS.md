# Repository Guidelines

## Project Structure & Module Organization
`fnet/` houses the package code: CLI utilities (`fnet/cli`), neural building blocks (`fnet/nn_modules`), helpers, and unit tests under `fnet/tests`. Reference workflows and training scripts live in `examples/`, with lightweight assets stored in `data/`. Documentation sources sit in `docs/`, static figures in `resources/`, and build artifacts in `build/` or `dist/`, which should stay out of commits. A ready-to-run `examples/prefs.json` demonstrates how to configure 13-slice stacks (patch depth 13 + `model_depth: 3`); copy and edit it when bootstrapping experiments.

## Build, Test, and Development Commands
Install dev dependencies with `pip install -e .[dev]` (append `[test]` or `[examples]` when needed). Run `make build` or `tox` for the full matrix (`py36`, `py37`, `lint`) before opening a pull request. Use `pytest fnet/tests -k <pattern>` for quick feedback, `tox -e lint` for flake8, and `make gen-docs` to rebuild the Sphinx site (preview via `docs/_build/html/index.html`). Validate user flows through the CLI, e.g., `fnet train --json config/train.json` followed by `fnet predict --json config/predict.json`, and keep `prefs.json` files in sync by setting `model_depth` alongside `bpds_kwargs.patch_shape[0]` for the number of axial slices you actually store (13 by default).

## Coding Style & Naming Conventions
Follow PEP 8, 4-space indentation, and the 88-character limit defined in `setup.cfg`. Flake8 runs with `E203`, `W291`, and `W503` ignored; avoid additional `noqa` tags unless essential. Modules, functions, configs, and files use `snake_case`, while classes use `CamelCase`. Prefer explicit imports and clear argparse flag names so CLI help stays readable. When editing JSON templates (including `prefs.json`), keep `model_depth` and `fnet_model_kwargs.nn_kwargs.depth` synchronized—helper utilities now expect the same value in both locations.

## Testing Guidelines
Pytest powers all suites; colocate new tests beside related modules in `fnet/tests/test_<feature>.py` and name them after the behavior under test (`test_predict_cli_handles_missing_rows`). `tox` runs `pytest --cov=fnet` and stores HTML coverage inside `.tox/*/tmp/coverage/html`; review it whenever touching serialization, IO, or new network blocks. Changes to `examples/` or data loaders should be paired with at least a smoke test that exercises the canned `data/` assets.

## Commit & Pull Request Guidelines
History favors concise, imperative summaries with scopes (`example: fix prediction`, `data: fix loading multichannel tiffs`); mirror that style and keep each commit cohesive (code + docs + tests). Pull requests should describe motivation, list major changes, link issues, and paste the latest `tox` or focused `pytest` output. Attach screenshots or metric snippets when altering docs, training curves, or artifacts referenced in `resources/`, and tag reviewers who own shared configs or datasets.

## Security & Configuration Tips
Training JSON files often embed dataset paths or credentials—scrub sensitive values before committing and keep environment-specific overrides in ignored files. Store large checkpoints or microscopy data outside the repo (e.g., `/mnt/training`) and reference them via config paths. Prefer relative paths rooted at the repository (`data/cell_table.csv`) so automation and contributors can reproduce experiments without manual rewiring.
