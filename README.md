# Adding Data
1. Follow our [Getting Started](https://illixr.github.io/ILLIXR/getting_started/) guide to run ILLIXR.
    * If you want meaningful results, set `profile: opt` in the `config/$YOUR_CONFIG.yaml` where `$YOUR_CONFIG.yaml` is the config you want to run for your experiment.
2. Running `./runner.sh config/$YOUR_CONFIG.yaml` should create a `metrics/` directory in `ILLIXR/`. Move this directory to `results/metrics/$NAME` where `results` is this repo and `$NAME` is an arbitrary unused directory name.
3. You must add `metrics/NAME/trial_conditions.yaml` for any data to be collected. See `analysis/util.py:TrialConditions` for the schema of this YAML.

# Running the Script
1. If this is your first time invoking thi script, install the dependencies by `cd results/analysis ; poetry install`.
2. Run `cd results/analysis ;poetry run python3 main.py`. This will run all of the analyses on all of the data.
3. Examine the results in `output/`. Note that `output/$NAME` corresponding to `metrics/$NAME`. `output/account_summaries.md` is a very good place to start.

# Adding an Analysis
- Per-trial and inter-trial analyses can be found in the `analysis/per-trial-analysis.py` and `analysis/inter-trial-analysis.py` files, respectively.
- Per-trial analyses takes a single `PerTrialData`, while inter-trial analyses take a `List[PerTrialData]`. See `analysis/util.py:PerTrialAnalysis` for what each attribute means.
- The `analysis` functions in each file run all inter-trial or per-trial analysis within.
- We've found it helpful to comment out the slow ones.
- Running the analyses is a slow way of revealing errors; To speed up development, you can quickly type check your analyses using Mypy,
  - `poetry run mypy --strict --ignore-missing-imports main.py`, and on the first time `poetry install --dev`.
  - Parameter-type: Use `:` after a variable name to signify type (e.g. `def foo(x: int)`).
  - Return-type: Use `->` to provide what is returned by function (e.g. `def foo() -> int`).
