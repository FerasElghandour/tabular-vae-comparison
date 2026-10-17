# Tabular VAE and Beta-VAE Comparison

**Feras Elghandour** · Technical case study

Measure the reconstruction–regularization trade-off of two variational autoencoders on a small tabular benchmark.

## Implementation

A 30-feature, two-dimensional-latent VAE with beta values 1 and 4, deterministic mean-latent reconstruction checks, latent visualization, and generated-feature comparisons.

![Implemented VAE training path](https://raw.githubusercontent.com/FerasElghandour/FerasElghandour.github.io/main/assets/vae-training.svg)

Training uses a sampled two-dimensional latent vector and a reconstruction-plus-KL objective. Deterministic reconstruction evaluation uses the latent mean.

## Evidence and results

The release compares both models with scaling fitted only on the training partition. Corrected execution results replace earlier figures affected by full-dataset preprocessing.

Results described as saved come from the retained experiment record. See [VALIDATION.md](VALIDATION.md) for the checks performed on this release and their scope.

## Fresh reproduction — October 17, 2026

After fitting scaling only on training records, mean-latent test reconstruction MSE is 0.2831 for β=1 and 0.3231 for β=4. Evaluation is deterministic and weighted by the number of reconstructed feature values.

The environment versions, input hashes, and notebook hash are recorded in [results/reproduction.json](results/reproduction.json).

![Recorded test reconstruction comparison](https://raw.githubusercontent.com/FerasElghandour/FerasElghandour.github.io/main/assets/vae-comparison.png)

Replotted from [results/reproduction.json](results/reproduction.json): test reconstruction MSE is 0.2831 at β=1 and 0.3231 at β=4. Scaling is fitted on training data only. [Input-distribution figure](assets/figure-01.png).

## Run locally

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab analysis.ipynb
```

Prepare the inputs described in [DATA.md](DATA.md) first. Source validation is available without installing model dependencies:

```sh
python scripts/validate_source.py
```

Run the focused offline behavior checks with `python smoke_test.py` after installing the dependencies.

## Limitations

Feature generation is a statistical demonstration, not clinically valid synthetic patient data. Distribution resemblance does not establish privacy, anonymity, diagnostic utility, or downstream safety.

## Development

Changes should include a clear description of the behavior affected and verification appropriate to that behavior. Keep input data, credentials, model caches, and generated artifacts outside version control.

