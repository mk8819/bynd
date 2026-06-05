"""Continuum amplitude fitting for the BYND post-processing step."""
from __future__ import annotations

import numpy as np
from helpers_line_search import filter_frequencies, read_sma_data
from helpers_line_search import generate_signal_tensor, update_amplitudes_tensor, write_to_file



def get_continuum_freq(
    target: np.ndarray,
    signal: np.ndarray,
    signal_off: np.ndarray,
    time: np.ndarray,
    sma_frq_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit continuum amplitudes to the residual of the line-search signal.

    Computes ``target - signal`` for both diagonal and off-diagonal components
    and fits the residual with Ridge regression over the full SMA frequency
    grid.  This captures broad spectral features that are not resolved by the
    line search.

    Parameters
    ----------
    target:
        Reference short-time RT-TDDFT signal, shape ``(n_time, 6)``.
        Column order: ``[xx, yy, zz, xy, xz, yz]``.
    signal:
        Diagonal line-search signal, shape ``(n_time, 3)``.
        Column order: ``[xx, yy, zz]``.
    signal_off:
        Off-diagonal line-search signal, shape ``(n_time, 3)``.
        Column order: ``[xy, xz, yz]``.
    time:
        Time grid, shape ``(n_time,)``.  Must match the first axis of
        *target*, *signal*, and *signal_off*.
    sma_frq_all:
        Full set of SMA frequencies over which continuum amplitudes are
        fitted, shape ``(n_sma_freq,)``.

    Returns
    -------
    amp_input_continuum:
        Continuum amplitudes for diagonal components,
        shape ``(3, n_sma_freq)``.
    intercept_on_continuum:
        Intercept for the diagonal continuum fit, shape ``(3,)``.
    amp_input_off_continuum:
        Continuum amplitudes for off-diagonal components,
        shape ``(3, n_sma_freq)``.
    intercept_off_continuum:
        Intercept for the off-diagonal continuum fit, shape ``(3,)``.
    """

    # subtract the signal from the target signal
    target_continuum = np.zeros((len(target[:,0]),6))
    target_continuum[:,:3] = target[:,:3] - signal
    target_continuum[:,3:] = target[:,3:] - signal_off

    # only fit the amplitudes, needs to be done with ridge regression which is on by default
    # values above 70 are a fair choice for the regression coefficient
    amp_input_continuum, intercept_on_continuum = update_amplitudes_tensor(target_continuum[:,:3], sma_frq_all, time, off_diagonal=False, reg_coef=70., intercept=True)
    amp_input_off_continuum, intercept_off_continuum = update_amplitudes_tensor(target_continuum[:,3:], sma_frq_all, time, off_diagonal=True, reg_coef=70., intercept=True)

    return amp_input_continuum, intercept_on_continuum, amp_input_off_continuum, intercept_off_continuum
