"""
Example BYND calculation: zinc phthalocyanine derivative (FHI-aims RT-TDDFT).

Loads a long-time RT-TDDFT reference signal, extracts a short-time segment,
runs the BYND frequency optimisation on that segment, adds a continuum
correction, and writes a spectrum comparison plot (spectrum_bynd_vs_exact.pdf).

All input files must be present in the working directory (copy from data/).
"""
import sys
sys.path.append('../helpers/')
sys.path.append('../src/')

import numpy as np
from helpers_line_search import (
    filter_frequencies, read_sma_data, read_RT_TDDFT_data,
    generate_signal_tensor, update_amplitudes_tensor, write_to_file,
    print_init_messages, print_iteration,
)
from simple_line_search import perform_line_search_tensor_off_diagonal
from continuum_amplitudes import get_continuum_freq
from get_spectrum_time_domain import compute_spectrum_using_time_signal
from plot_spectrum import plot_simple_spectrum

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

final_time   = 4000.   # length of the reference time signal [au]
n_time_steps = 20000   # total number of time steps

# Short-time window used as the optimisation target.
# Cutting the first steps avoids artefacts from the electric field pulse.
cut_start = 100
cut_end   = 1500

# Line-search parameters
iterations         = 85
switch_2_off       = 50       # iteration at which off-diagonal terms are included
search_radius_init = 0.05     # search half-width for iteration 0 [au]
search_radius      = 0.001    # search half-width for all later iterations [au]
grid_spacing       = 0.00001

# ----------------------------------------------------------------------------
# Load input data
# ----------------------------------------------------------------------------

dipole_xyz, _ = read_RT_TDDFT_data(t_unit_au=True)
sma_frq, _, sma_tmom = read_sma_data('sma_tddft.data', 'trans_mom_sma.data')

# Cut to the short-time optimisation window.
# Dipole scaled by 100 for numerical convenience; corrected when computing
# the final spectrum.
time, _ = np.linspace(0., final_time, n_time_steps, retstep=True, endpoint=False)
time       = time[cut_start:cut_end]
dipole_xyz = 100 * dipole_xyz[cut_start:cut_end, :]

sma_frq, sma_tmom = filter_frequencies(sma_frq, sma_tmom, threshold=1.5)

# ----------------------------------------------------------------------------
# Assemble optimisation target  [xx, yy, zz, xy, xz, yz]
# ----------------------------------------------------------------------------

target = np.zeros((len(time), 6))
target[:, 0] = dipole_xyz[:, 0]   # xx
target[:, 1] = dipole_xyz[:, 4]   # yy
target[:, 2] = dipole_xyz[:, 8]   # zz
target[:, 3] = dipole_xyz[:, 1]   # xy
target[:, 4] = dipole_xyz[:, 2]   # xz
target[:, 5] = dipole_xyz[:, 5]   # yz

freq_input     = sma_frq
freq_sma_start = sma_frq.copy()   # fixed reference for the regularisation anchor

amp_input, intercept_on = update_amplitudes_tensor(
    target[:, :3], freq_input, time,
    off_diagonal=False, intercept=True, method='ridge', reg_coef=20.01,
)
amp_input_off, intercept_off = update_amplitudes_tensor(
    target[:, 3:], freq_input, time,
    off_diagonal=True, intercept=True, method='ridge', reg_coef=20.01,
)

# ----------------------------------------------------------------------------
# Line search optimisation
# ----------------------------------------------------------------------------

np.set_printoptions(precision=3)
print_init_messages(
    iterations, switch_2_off, search_radius, search_radius_init,
    grid_spacing, freq_sma_start, len(target),
)

for i in range(iterations):

    if i == 0:
        freq_input, amp_input, amp_input_off = perform_line_search_tensor_off_diagonal(
            target, amp_input, amp_input_off, freq_input, time, freq_sma_start,
            rf=search_radius_init, df=grid_spacing,
            diagonal_only=True, calc_intercept=True,
            reg_method='ridge', alpha_value=20.01, l1_ratio_value=0.8,
            initial='random',
        )
    elif i >= switch_2_off:
        freq_input, amp_input, amp_input_off = perform_line_search_tensor_off_diagonal(
            target, amp_input, amp_input_off, freq_input, time, freq_sma_start,
            rf=search_radius, df=grid_spacing,
            diagonal_only=False, calc_intercept=True,
            reg_method='ridge', alpha_value=20.01,
        )
    else:
        freq_input, amp_input, amp_input_off = perform_line_search_tensor_off_diagonal(
            target, amp_input, amp_input_off, freq_input, time, freq_sma_start,
            rf=search_radius, df=grid_spacing,
            diagonal_only=True, calc_intercept=True,
            reg_method='ridge', alpha_value=20.01,
        )

    print_iteration(i, freq_input, amp_input)

write_to_file('line_search_amp.out',     amp_input,     freq_input, intercept=intercept_on)
write_to_file('line_search_amp_off.out', amp_input_off, freq_input, off_diag=True, intercept=intercept_off)

# ----------------------------------------------------------------------------
# Continuum correction
# ----------------------------------------------------------------------------

signal     = generate_signal_tensor(amp_input.T,     freq_input, time, intercept=intercept_on)
signal_off = generate_signal_tensor(amp_input_off.T, freq_input, time, intercept=intercept_off)

# Reload with a looser threshold to cover the full SMA frequency grid
sma_frq_all, _, sma_tmom_all = read_sma_data('sma_tddft.data', 'trans_mom_sma.data')
sma_frq_all, sma_tmom_all = filter_frequencies(sma_frq_all, sma_tmom_all, threshold=0.07)

amp_cont, intercept_cont, amp_cont_off, intercept_cont_off = get_continuum_freq(
    target, signal, signal_off, time, sma_frq_all,
)

signal     += generate_signal_tensor(amp_cont.T,     sma_frq_all, time, intercept=intercept_cont)
signal_off += generate_signal_tensor(amp_cont_off.T, sma_frq_all, time, intercept=intercept_cont_off)

write_to_file('line_search_amp_continuum.out',     amp_cont,     sma_frq_all, intercept=intercept_cont)
write_to_file('line_search_amp_continuum_off.out', amp_cont_off, sma_frq_all, off_diag=True, intercept=intercept_cont_off)

# ----------------------------------------------------------------------------
# Spectrum
# ----------------------------------------------------------------------------

compute_spectrum_using_time_signal(flag_plot_results=False)
plot_simple_spectrum()
