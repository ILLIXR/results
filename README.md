﻿- Per-trial and inter-trial analysis can be found in the `analysis/per-trial-analysis.py` and `analysis/inter-trial-analysis.py` files, respectively. The `analysis` functions in each file run all inter-trial or per-trial analysis within. 
- All data goes in `metrics/` 
- You must add `trial_conditions.yaml` for any data to be collected 
- [Fields of `PerTrialData` meaning]
- To run:
	- Run`poetry shell`, then inside that `python3 main.py`
	- `main.py` will run all of the analysis 
- You can type check using `mypy --strict --ignore-missing-imports main.py` to ensure proper types are being used. Requirements for mypy to be able to check funcions: 
	- Inputs: Use `:` after a variable name to signify type
	- Outputs: Use `->` to provide what is returned by function 

