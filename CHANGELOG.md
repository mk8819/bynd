# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2024-07-01

### Added
- Type hints for all public functions across `src/` and `helpers/`
- NumPy-style docstrings for all functions
- `.gitignore` covering Python artifacts and BYND output files

### Fixed
- Compatibility with NumPy 2.x (`np.trapz` → `np.trapezoid`)

### Changed
- `requirements.txt`: removed dummy `sklearn` package, added `statsmodels`,
  relaxed version pins to minimum requirements

## [0.1.0] - 2024-01-01

### Added
- Initial release of BYND
- Core line search optimisation (`src/simple_line_search.py`)
- Continuum amplitude fitting (`src/continuum_amplitudes.py`)
- FHI-aims RT-TDDFT I/O helpers (`helpers/`)
- Example calculation for a zinc phthalocyanine derivative (`examples/`)
- Sample data for the example (`data/`)
