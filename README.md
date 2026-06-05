# BYND

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12771684.svg)](https://doi.org/10.5281/zenodo.12771684)
[![Paper](https://img.shields.io/badge/Nature_Communications-10.1038%2Fs41467--024--52368--5-blue)](https://doi.org/10.1038/s41467-024-52368-5)

BYND combines approximate frequency results with exact short-time dynamics to
compute electronic excitation spectra efficiently.  It achieves highly reliable
results in regimes where other super-resolution techniques such as
compressed-sensing typically fail.

BYND is fully independent of the underlying electronic structure method or code.
The only requirements are a time-dependent dipole signal (short-time dynamics)
and a sufficiently accurate initial guess for the excitation spectrum.

## Publication

If you use BYND in your research, please cite the following article:

> M. Kick, T. Van Voorhis,
> *Beyond the short-time limit in real-time time-dependent density functional theory simulations*,
> **Nature Communications** 15, 8615 (2024).
> https://doi.org/10.1038/s41467-024-52368-5

A preprint is also available on arXiv: https://arxiv.org/abs/2401.06929

## DOI

Software archive: [10.5281/zenodo.12771684](https://doi.org/10.5281/zenodo.12771684)

## Features and planned extensions

BYND is under active development. Future versions will include support for:

- Quadrupole moments
- Basis set extrapolation
- Replacing the line search with more advanced optimisation techniques
- Routines for easier data handling

## Performance

The standard execution time on a standard laptop is a few minutes.
No non-standard hardware is required.

---

## Repository structure

### Core (`src/`)

BYND essentially consists of two routines:

1. `src/simple_line_search.py` → `perform_line_search_tensor_off_diagonal`
2. `src/continuum_amplitudes.py` → `get_continuum_freq`

These are independent of the electronic structure code — they will run as long
as the input data is provided in the correct format.

Both routines rely on helper functions in `src/helpers_line_search.py`:

- `update_amplitudes_tensor`
- `sort_amplitudes_tensor`
- `objective_tensor`
- `generate_signal_tensor`
- `get_search_grid`

### Example (`examples/`)

`examples/optimize_frequencies.py` demonstrates a full BYND calculation.
It takes a long-time RT-TDDFT signal, cuts out a short-time segment, runs the
optimisation on that segment, and finally compares the BYND spectrum against the
exact long-time result.  In a real application the long-time signal would not be
available.

**To run the example:**

```bash
# Copy the input data into the examples directory
cp data/* examples/

# Run the optimisation
cd examples/
python3 optimize_frequencies.py
```

This produces `spectrum_bynd_vs_exact.pdf` comparing the BYND spectrum with the
exact long-time RT-TDDFT reference.

### Helpers (`helpers/`)

Helper functions for I/O and spectrum calculation.  These are specific to
FHI-aims RT-TDDFT output format and will need to be adapted for other codes.

- `helpers/helpers_rt_tddft_fhiaims_utilities.py` — FHI-aims data classes
- `helpers/get_spectrum_time_domain.py` — Fourier transform pipeline
- `helpers/spectrum_helpers.py` — signal I/O utilities
- `helpers/plot_spectrum.py` — spectrum plotting

### Data (`data/`)

Input files for the example calculation:

| File | Description |
|---|---|
| `sma_tddft.data` | SMA frequencies and oscillator strengths |
| `trans_mom_sma.data` | SMA transition dipole moments |
| `x.rt-tddft.dipole.dat` | RT-TDDFT dipole signal, x-polarised field |
| `y.rt-tddft.dipole.dat` | RT-TDDFT dipole signal, y-polarised field |
| `z.rt-tddft.dipole.dat` | RT-TDDFT dipole signal, z-polarised field |
| `x.rt-tddft.ext-field.dat` | Applied external field |
