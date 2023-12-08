
import numpy as np
from helpers_line_search import filter_frequencies, read_sma_data
from helpers_line_search import generate_signal_tensor, update_amplitudes_tensor, write_to_file



def get_continuum_freq(target, signal, signal_off, time, sma_frq_all):

   """
   This functions calculates the continuum amplitudes.
   This is achieved by applying ridge regression
   to the residual of target - signal/signal_off
   
   Input:

   o) target:     reference short time signal in
                    [xx, yy, zz, xy, xz, yz] format

   o) signal:     signal from line search
                    [xx, yy, zz] format
   o) signal_off: signal from line search
                    [xy, xz, yz]

   o) time: time array of length len(target[:,0])
            this needs to match the target and signal

   o) sma_frq_all: SMA frequencies for which the
                   continuum amplitudes should
                   be adjusted.

   * target and signals need to match in length

   Output:

   o) amp_input_continuum:     continuum amplitudes for
                               signal [xx, yy, zz]

   o) intercept_on_continuum:  intercept for signal

   o) amp_input_off_continuum: continuum amplitudes for
                               signal_off [xy, xz, yz]

   o) intercept_off_continuum: intercept for signal_off

   
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
