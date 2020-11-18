# Adding an Analysis
- Per-trial and inter-trial analyses can be found in the `analysis/per-trial-analysis.py` and `analysis/inter-trial-analysis.py` files, respectively.
- Per-trial analyses takes a single `PerTrialData`, while inter-trial analyses take a `List[PerTrialData]`. See `analysis/util.py:PerTrialAnalysis` for what each attribute means.
- The `analysis` functions in each file run all inter-trial or per-trial analysis within. 
- Running the analyses is a slow way of revealing errors; To speed up development, you can quickly type check your analyses using `mypy --strict --ignore-missing-imports main.py`. Requirements for mypy to be able to check funcions: 
  - Parameter-type: Use `:` after a variable name to signify type (e.g. `def foo(x: int)`).
  - Return-type: Use `->` to provide what is returned by function (e.g. `def foo() -> int`).
  - Container type: If you create an empty container, use `:` to denote the type. (e.g. `x: List[int] = []`).

# Adding Data
- Place data in `metrics/NAME`, where `NAME` is an arbitrary unused directory.
- You must add `metrics/NAME/trial_conditions.yaml` for any data to be collected. See `analysis/util.py:TrialConditions` for the schema of this YAML.

# Running the Script
- Run `poetry run python3 main.py`. This will run all of the analyses on all of the data.
- The output corresponding to `metrics/NAME` will appear in `output/NAME`, where `NAME`.
