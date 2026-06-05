# Installation

BYND requires Python 3.8 or later and a small set of standard scientific
Python packages.

## Recommended: virtual environment

```bash
python3 -m venv bynd-env
source bynd-env/bin/activate   # Windows: bynd-env\Scripts\activate
pip install -r requirements.txt
```

## Manual install

```bash
pip install numpy scipy scikit-learn scikit-optimize matplotlib statsmodels
```

## Verifying the installation

Run the provided example to confirm everything is working:

```bash
cp data/* examples/
cd examples/
python3 optimize_frequencies.py
```

This should complete in a few minutes and produce `spectrum_bynd_vs_exact.pdf`.

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `scipy` | Signal processing |
| `scikit-learn` | Ridge / Lasso / ElasticNet regression |
| `scikit-optimize` | Optimisation utilities |
| `matplotlib` | Plotting |
| `statsmodels` | Spectrum post-processing |
