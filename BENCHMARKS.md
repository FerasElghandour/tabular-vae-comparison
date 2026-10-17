# Reproduction and benchmark scope

The noninteractive path reruns both 80-epoch models and reproduces deterministic held-out reconstruction MSE**0.283078 for β=1** and**0.323131 for β=4**. Both state dictionaries and the training-fitted scaler are exported; fresh-model reloads verify the same scores.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Existing focused regression suite | None |
| `full-benchmark` | Original fixed 80 epoch β1/β4 notebook comparison and fresh-model checkpoint reload | None |

Supply named paths with `--input name=/absolute/path`. Exact commands are versioned in [reproduction.json](reproduction.json); direct training commands and preparation contracts are below.

## Evidence boundary

The 80-epoch budget is fixed in advance. Reconstruction quality on this dataset is not evidence of clinical utility, privacy protection or causal structure.

Keep input acquisition and source credits in [DATA.md](DATA.md). Historical results, new reference runs and small contract fixtures have separate evidential scope.

## Recorded command definitions

### checks

```text
{python} smoke_test.py
```

### full-benchmark

```text
{python} train.py --output {output}/run
{python} verify_checkpoint.py {output}/run
```

`{python}` is the current interpreter. `{output}` is the new run directory. `{input:name}` refers to a named absolute input path supplied to the driver. Install declared dependencies in a clean environment before executing. Never execute untrusted serialized models or notebooks.
