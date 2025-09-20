# Repository Guidelines

## Project Structure & Modules
- `time_diff_chart.py` is the primary entry point; it ingests timestamped CSV data, computes inter-sample deltas, and can render matplotlib diagnostics.
- Sample datasets (`IOMBTC_2025-09-18_160123.csv`, related `.xlsx`) live at the repo root for reproducible experiments.
- Generated artefacts such as `delta_diag.png` are kept in the root; prefer creating a `plots/` subdirectory for new outputs to avoid clutter.

## Build, Test, and Development Commands
- Run analyses locally with `python time_diff_chart.py <path/to.csv> --show-series --show-rolling`; add `--output plots/<name>.png` to save figures.
- Install optional visualisation/stats extras via `python -m pip install matplotlib statsmodels` when plots or ADF testing are needed; the script downgrades gracefully if they are absent.
- Use `python -m pip install --upgrade pip` before dependency work to ensure modern wheels for scientific packages.

## Coding Style & Naming Conventions
- Follow PEP 8: four-space indentation, descriptive snake_case for functions/variables, and PascalCase for classes should any be introduced.
- Keep helper routines pure; prefer returning data instead of mutating globals so the CLI stays testable.
- Extend CLI options using `argparse` patterns already present; keep new flags kebab-case (e.g., `--rolling-window`).

## Testing Guidelines
- Add unit tests under `tests/` with `pytest`; mimic existing function boundaries (`to_datetime`, `consecutive_deltas`, etc.).
- Name files `test_<feature>.py`, and assert both nominal results and edge cases like empty rows or malformed timestamps.
- Target high coverage for math helpers; mock external libraries (matplotlib, statsmodels) so the suite runs in headless CI.

## Commit & Pull Request Guidelines
- Adopt Conventional Commits (`feat:`, `fix:`, `docs:`) for clarity until project history establishes a different precedent.
- Squash noisy WIP commits before sharing; each PR should describe input dataset assumptions, CLI flags touched, and include updated screenshots when plots change.
- Link related tracking issues or notebooks, and flag any new dependencies in the PR body so reviewers can validate local environments.

## Data & Plotting Notes
- Sanitize proprietary inputs before committing sample CSVs; prefer anonymised, trimmed datasets under `data/` when sharing.
- When exporting figures, embed parameters in filenames (`plots/deltas_show-rolling_1k.png`) to aid reproducibility.
