"""Post-processing: compute absorption spectra from BYND and RT-TDDFT signals."""
from __future__ import annotations

import sys
sys.path.append('../helpers/')

import numpy as np
from helpers_line_search import read_sma_data, generate_signal_tensor,\
        read_sma_data, read_RT_TDDFT_data, filter_frequencies, update_amplitudes_tensor
from spectrum_helpers import get_line_search_dipole_data,\
        generate_signal_and_save_to_file
from helpers_rt_tddft_fhiaims_utilities import DipoleData, FieldData, ElectronicDipole, set_external_field

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import matplotlib.font_manager as fm
from mpl_toolkits import mplot3d


def compute_spectrum_using_time_signal(flag_plot_results: bool = True) -> None:
    """Compute and compare absorption spectra from BYND and long-time RT-TDDFT.

    Reads the line-search and continuum amplitude files, reconstructs the
    corresponding dipole signals, runs the FHI-aims Fourier-transform
    pipeline on both signals, and optionally plots the resulting absorption
    spectra side-by-side.

    Workflow:

    1. Read optimised frequencies and amplitudes from ``line_search_amp.out``
       and ``line_search_amp_continuum.out``.
    2. Synthesise dipole signals and write them in FHI-aims RT-TDDFT format.
    3. Construct :class:`ElectronicDipole` objects for both the RT-TDDFT
       reference and the BYND result.
    4. Perform analytic Fourier transforms and compute absorption spectra.

    Parameters
    ----------
    flag_plot_results:
        If ``True``, generate and display plots of the final spectra.

    Notes
    -----
    In a real application the long-time RT-TDDFT signal is not available;
    this function is primarily used for validation against the exact result.
    """
    
    # convert the output to FHI-aims rt-tddft format for analysis
    #-------------------------------------------------------------

    # get data from line search output
    frq, frq_con, amp, amp_con, inter, inter_con  = get_line_search_dipole_data('line_search_amp.out', 'line_search_amp_continuum.out', continuum=True)
    amp = amp*0.01; amp_con = amp_con*0.01; inter =inter*0.01; inter_con =inter_con*0.01

    final_time   = 4000.3
    n_time_steps = 20003

    # generate signal and save it in the correct format
    generate_signal_and_save_to_file(amp, amp_con, frq, frq_con, 0, final_time, n_time_steps, continuum=True, intercept_value=inter, intercept_value_continuum=inter_con)


    dip_x = DipoleData('x.rt-tddft.dipole.dat', "dipole_x")
    dip_y = DipoleData('y.rt-tddft.dipole.dat', "dipole_y")
    dip_z = DipoleData('z.rt-tddft.dipole.dat', "dipole_z")


    # line-search
    line_search_dip_x = DipoleData('line_search_x.dipole.dat', "dipole_x")
    line_search_dip_y = DipoleData('line_search_y.dipole.dat', "dipole_y")
    line_search_dip_z = DipoleData('line_search_z.dipole.dat', "dipole_z")


    # external field, it is the same for both signals
    field = ['x.rt-tddft.ext-field.dat',None,None]

    fld_x = FieldData(field[0], "field_x", True)
    fld_y = FieldData(field[1], "field_y", True) if field[1] is not None else None
    fld_z = FieldData(field[2], "field_z", True) if field[2] is not None else None

    input_field = set_external_field(fld_x, fld_y, fld_z, flag_copy_ref=True, flag_analytic_field=True)

    # for standard fourier
    dipole = ElectronicDipole(flag_plot=flag_plot_results, flag_normalize=False, flag_intp_plot=False,   \
                   flag_analytic_ft=True, intp_fac=None, t_min=2.1, t_max=-1.0,    \
                   t_shift=0.0, f_min=0.0, f_max=20.0, lpeak_hgt=None,    \
                   lpeak_wdt=None, add_zeros=0, t0_damp=2, damp_type='poly', \
                   expfac_dp = 0.0000, n_ft_pade=-1, fmin_pade=0.0, fmax_pade=-1.0, \
                   data_x=dip_x, data_y=dip_y, data_z=dip_z, \
                   input_field=input_field, out_file_name='rt-tddft')

    line_search_dipole = ElectronicDipole(flag_plot=flag_plot_results, flag_normalize=False, flag_intp_plot=False,   \
                   flag_analytic_ft=True, intp_fac=None, t_min=2.1, t_max=-1.0,    \
                   t_shift=0.0, f_min=0.0, f_max=20.0, lpeak_hgt=None,    \
                   lpeak_wdt=None, add_zeros=0, t0_damp=2, damp_type='poly', \
                   expfac_dp = 0.0000, n_ft_pade=-1, fmin_pade=0.0, fmax_pade=-1.0, \
                   data_x=line_search_dip_x, data_y=line_search_dip_y, data_z=line_search_dip_z, \
                   input_field=input_field, out_file_name='line-search-tddft')

    plt.rc('mathtext', fontset="cm")
    plt.show()
    plt.clf()
