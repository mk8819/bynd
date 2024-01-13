#############################################################################################################
# 
# This is essentially a wrapper around different functions which perform the entire job
# The complexity is needed cause of the RT-TDDFT data format of FHIaims.
#
# This is a wrapper for FHIaims data only, this will not work for other RT-TDDFT signals from other codes
#
#############################################################################################################

import sys
sys.path.append('../helpers/')
sys.path.append('../src/')
sys.path.append('../data/')
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
from helpers_line_search import filter_frequencies, read_sma_data, read_RT_TDDFT_data
from helpers_line_search import generate_signal_tensor, update_amplitudes_tensor, write_to_file
from simple_line_search import perform_line_search_tensor_off_diagonal
from helpers_line_search import print_init_messages, print_iteration
from continuum_amplitudes import get_continuum_freq
from get_spectrum_time_domain import compute_spectrum_using_time_signal
from plot_spectrum import plot_simple_spectrum
# -----------------------------------------------------------------------------------------------------------
# START
# -----------------------------------------------------------------------------------------------------------
# global settings
# ------------------------------------------------------------------------------------------------------------

final_time   = 4000. # length of the reference time signal in au
n_time_steps = 20000 # total time steps


# this is for cutting the short time signal out of the long time reference
cut_start    = 100   # start point of the signal which we use as target in the optimization procedure
cut_end      = 1500  # end point of the signal which we use as target in the optimization procedure
# Note: We cut the first piece of the signal, the electric field puls at t=0 sometimes 
#       causes spectral artifacts.

# Read in all the data which must be in the current folder
# ------------------------------------------------------------------------------------------------------------

# Essential tasks which need to be done before optimization:
# 1) We need to load the dipole data
# 2) We need to load the SMA data
# 3) We need to select the SMA narrow features 
# 4) We need to prepare the short-time target signal for the optimization

dipole_xyz, tddft_time = read_RT_TDDFT_data(t_unit_au=True) 
sma_frq, sma_osc, sma_tmom = read_sma_data('sma_tddft.data',  'trans_mom_sma.data')


time, dt = np.linspace(0., final_time, n_time_steps, retstep=True, endpoint=False)

# we mutiply it with 100, gives nicer values for the amplitude output
# we correct for it later on when we calculate the spectrum
# just for example purposes
dipole_xyz = 100* dipole_xyz[cut_start:cut_end,:]
time = time[cut_start:cut_end]

# Important, first we only select the sparse subset of the frequencies
sma_frq, sma_tmom = filter_frequencies(sma_frq, sma_tmom, 1.5)

amp_input = np.abs(sma_tmom[:,:]) # we x, y z
freq_input = sma_frq

# set target values of complete problem
# the ordering is [xx, yy, zz, xy, xz, yz]
target = np.zeros((len(time),6))
target[:,0] = dipole_xyz[:,0]
target[:,1] = dipole_xyz[:,4]
target[:,2] = dipole_xyz[:,8]
target[:,3] = dipole_xyz[:,1]
target[:,4] = dipole_xyz[:,2]
target[:,5] = dipole_xyz[:,5]

# input amplitudes just for initialization
amp_input, intercept_on = update_amplitudes_tensor(target[:,:3], freq_input, time, off_diagonal=False, intercept=True,method='ridge', reg_coef=20.01, ratio=0.8)
amp_input_off, intercept_off = update_amplitudes_tensor(target[:,3:], freq_input, time, off_diagonal=True, intercept=True,method='ridge',reg_coef=20.01, ratio=0.8) 

# perform optimization
# ------------------------------------------------------------------------------------------------------------
np.set_printoptions(precision=3)

freq_sma_start = freq_input

# set parameters
iterations         = 85 # was 35
switch_2_off       = 50
search_radius_init = 0.05
search_radius      = 0.001
grid_spacing       = 0.00001

# print out initial messages
print_init_messages(iterations, switch_2_off, search_radius, search_radius_init, grid_spacing, freq_sma_start, len(target))

# start iterations
# ------------------------------------------------------------------------------------------------------------

# Non linear-optimization is happening here:

#    - The first iteration has a search radius of 0.05
#    - In the first iteration we also randomly modify the amplitudes
#    - all other iterations have a search radius of 0.001
#    - the search grid density is 0.00001

#    This example is not checking if the result is converged
#    We simply set the number of iterations to a sufficient high
#    value.

#    We also included the possibility to use "off-diagonal"
#    signals at later iterations. This can improve the 
#    convergence behaviour.

#    We include off-diagonal signals at iteration 50.
#    This as a very safe choice. 


for i in range(iterations):

    if i == 0:

        freq_input, amp_input, amp_input_off = perform_line_search_tensor_off_diagonal(target, amp_input, amp_input_off, freq_input, time, freq_sma_start, rf=search_radius_init, df=grid_spacing, diagonal_only=True, calc_intercept=True, reg_method='ridge', alpha_value=20.01, l1_ratio_value=0.8, initial='random')

    elif i >= switch_2_off: # we also include the off diagonal elements

        freq_input, amp_input, amp_input_off = perform_line_search_tensor_off_diagonal(target, amp_input, amp_input_off, freq_input, time, freq_sma_start, rf=search_radius, df=grid_spacing, diagonal_only=False, calc_intercept=True, reg_method='ridge', l1_ratio_value=20.8, alpha_value=20.01)

    else: # only diagonal is considered

        freq_input, amp_input, amp_input_off = perform_line_search_tensor_off_diagonal(target, amp_input, amp_input_off, freq_input, time, freq_sma_start, rf=search_radius, df=grid_spacing, diagonal_only=True, calc_intercept=True, reg_method='ridge', l1_ratio_value=20.8, alpha_value=20.01)

    print_iteration(i, freq_input, amp_input)


# save narrow features results
# ------------------------------------------------------------------------------------------------------------

time, dt = np.linspace(0., final_time, n_time_steps, retstep=True, endpoint=False)
signal = generate_signal_tensor(np.transpose(amp_input), freq_input, time, dipole=False, intercept=intercept_on)

write_to_file('line_search_amp.out', amp_input, freq_input, intercept=intercept_on)
write_to_file('line_search_amp_off.out', amp_input_off, freq_input, off_diag=True, intercept=intercept_off)


# get the continuum frequencies
# ------------------------------------------------------------------------------------------------------------

# generate the signal which is going to be subtracted from the target signal
time, dt = np.linspace(0., final_time, n_time_steps, retstep=True, endpoint=False)
time = time[cut_start:cut_end]
signal = generate_signal_tensor(np.transpose(amp_input), freq_input, time, dipole=False, intercept=intercept_on)
signal_off = generate_signal_tensor(np.transpose(amp_input_off), freq_input, time, dipole=False, off_diagonal=True, intercept=intercept_off)


# get all sma frequencies
sma_frq_all, sma_osc_all, sma_tmom_all = read_sma_data('sma_tddft.data',  'trans_mom_sma.data')
# we do not need all of them we apply a minimum threshhold this filters out all
# frequencies which do not have an amplitude anyway and saves memory and computing time
sma_frq_all, sma_tmom_all = filter_frequencies(sma_frq_all, sma_tmom_all, 0.07)
# get the amplitudes for the continuum region
amp_input_continuum, intercept_on_continuum, amp_input_off_continuum, intercept_off_continuum = get_continuum_freq(target, signal, signal_off, time, sma_frq_all) 

# calculate complete signal
signal = signal + generate_signal_tensor(np.transpose(amp_input_continuum), sma_frq_all, time, dipole=False, intercept=intercept_on_continuum)
signal_off = signal_off + generate_signal_tensor(np.transpose(amp_input_off_continuum), sma_frq_all, time, dipole=False, off_diagonal=True, intercept=intercept_off_continuum)

# write continuum information to file
write_to_file('line_search_amp_continuum.out', amp_input_continuum, sma_frq_all, intercept=intercept_on_continuum)
write_to_file('line_search_amp_continuum_off.out', amp_input_off_continuum, sma_frq_all, off_diag=True, intercept=intercept_off_continuum)

# calculate, plot and save the spectrum
# -----------------------------------------------------------------------------------------------------------
compute_spectrum_using_time_signal(flag_plot_results=False)
plot_simple_spectrum()
# -----------------------------------------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------------------------------------
