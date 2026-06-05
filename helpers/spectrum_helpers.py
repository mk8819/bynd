"""I/O helpers for reading and writing BYND line-search results."""
from __future__ import annotations

import numpy as np
from helpers_line_search import read_sma_data, generate_signal_tensor
import matplotlib.pyplot as plt

AU_TO_FS = 0.024188843265857   # atomic units of time → femtoseconds



def get_line_search_dipole_data(
    amp_file: str,
    amp_continuum_file: str,
    continuum: bool = False,
) -> tuple[np.ndarray, ...]:
    """Read line-search frequencies and amplitudes from output files.

    The wrapper script ``optimize_frequencies.py`` writes optimised
    amplitudes and frequencies to plain-text files.  This function reads
    those files back for post-processing without repeating the optimisation.

    Parameters
    ----------
    amp_file:
        Path to the narrow-feature amplitude file produced by the line
        search (e.g. ``line_search_amp.out``).
    amp_continuum_file:
        Path to the continuum amplitude file.  Only read when *continuum*
        is ``True``.
    continuum:
        If ``True``, also read the continuum amplitude file and return all
        six arrays; otherwise return only the three narrow-feature arrays.

    Returns
    -------
    When *continuum* is ``False``:
        ``(frq, amp, intercept_value)``

    When *continuum* is ``True``:
        ``(frq, frq_continuum, amp, amp_continuum,
        intercept_value, intercept_value_continuum)``

    frq:
        Narrow-feature frequencies, shape ``(n_freq,)``.
    frq_continuum:
        Continuum frequencies, shape ``(n_sma_freq,)``.
    amp:
        Narrow-feature amplitudes ``[xx, yy, zz]``, shape ``(3, n_freq)``.
    amp_continuum:
        Continuum amplitudes ``[xx, yy, zz]``, shape ``(3, n_sma_freq)``.
    intercept_value:
        Static offset for the narrow-feature signal, shape ``(3,)``.
    intercept_value_continuum:
        Static offset for the continuum signal, shape ``(3,)``.
    """

    intercept=False

    if continuum == False:

        frq = []
        amp_xx = []
        amp_yy = []
        amp_zz = []
        intercept_value = []

        with open(amp_file,'r') as f:
            for line in f:
                if line.startswith('# intercept:'):
                    intercept=True
                    intercept_value.append(float(line.split()[2]))
                    intercept_value.append(float(line.split()[3]))
                    intercept_value.append(float(line.split()[4]))

                elif line.startswith('# frequency'):
                    continue
                else:
                    frq.append(float(line.split()[0]))
                    amp_xx.append(float(line.split()[1]))
                    amp_yy.append(float(line.split()[2]))
                    amp_zz.append(float(line.split()[3]))

        intercept_value = np.asarray(intercept_value)

        frq    = np.asarray(frq)
        amp_xx = np.asarray(amp_xx)
        amp_yy = np.asarray(amp_yy)
        amp_zz = np.asarray(amp_zz)

        amp = np.zeros((3, len(frq))) # we store it as a n_target, n_feature array

        amp[0,:] = amp_xx
        amp[1,:] = amp_yy
        amp[2,:] = amp_zz

        return frq, amp, intercept_value

    else:

        intercept_value = []
        intercept_value_continuum = []

        frq = []
        frq_continuum = []
        amp_xx = []
        amp_yy = []
        amp_zz = []

        amp_continuum_xx = []
        amp_continuum_yy = []
        amp_continuum_zz = []

        with open(amp_file,'r') as f:
            for line in f:
                if line.startswith('# intercept:'):
                    intercept=True
                    intercept_value.append(float(line.split()[2]))
                    intercept_value.append(float(line.split()[3]))
                    intercept_value.append(float(line.split()[4]))

                elif line.startswith('# frequency'):
                    continue
                else:
                    frq.append(float(line.split()[0]))
                    amp_xx.append(float(line.split()[1]))
                    amp_yy.append(float(line.split()[2]))
                    amp_zz.append(float(line.split()[3]))

        with open(amp_continuum_file,'r') as f:
            for line in f:
                if line.startswith('# intercept:'):
                    intercept=True
                    intercept_value_continuum.append(float(line.split()[2]))
                    intercept_value_continuum.append(float(line.split()[3]))
                    intercept_value_continuum.append(float(line.split()[4]))

                elif line.startswith('# frequency'):
                    continue
                else:
                    frq_continuum.append(float(line.split()[0]))
                    amp_continuum_xx.append(float(line.split()[1]))
                    amp_continuum_yy.append(float(line.split()[2]))
                    amp_continuum_zz.append(float(line.split()[3]))

        intercept_value =  np.asarray(intercept_value)
        intercept_value_continuum =  np.asarray(intercept_value_continuum)


        frq    = np.asarray(frq)
        amp_xx = np.asarray(amp_xx)
        amp_yy = np.asarray(amp_yy)
        amp_zz = np.asarray(amp_zz)

        frq_continuum    = np.asarray(frq_continuum)
        amp_continuum_xx = np.asarray(amp_continuum_xx)
        amp_continuum_yy = np.asarray(amp_continuum_yy)
        amp_continuum_zz = np.asarray(amp_continuum_zz)

        amp = np.zeros((3, len(frq))) # we store it as a n_target, n_feature array
        amp_continuum = np.zeros((3, len(frq_continuum))) # we store it as a n_target, n_feature array

        amp[0,:] = amp_xx
        amp[1,:] = amp_yy
        amp[2,:] = amp_zz

        amp_continuum[0,:] = amp_continuum_xx
        amp_continuum[1,:] = amp_continuum_yy
        amp_continuum[2,:] = amp_continuum_zz

        return frq, frq_continuum, amp, amp_continuum, intercept_value, intercept_value_continuum


def generate_signal_and_save_to_file(
    amp: np.ndarray,
    amp_continuum: np.ndarray,
    frq: np.ndarray,
    frq_continuum: np.ndarray,
    start: float,
    end: float,
    n_points: int,
    intercept_value: np.ndarray | None = None,
    intercept_value_continuum: np.ndarray | None = None,
    continuum: bool = False,
) -> None:
    """Reconstruct dipole signals from line-search results and write FHI-aims files.

    Synthesises the time-domain dipole signal from the optimised frequencies
    and amplitudes, then saves it as FHI-aims RT-TDDFT dipole files
    (one per Cartesian direction).

    Parameters
    ----------
    amp:
        Narrow-feature amplitudes ``[xx, yy, zz]``, shape ``(3, n_freq)``.
    amp_continuum:
        Continuum amplitudes ``[xx, yy, zz]``, shape ``(3, n_sma_freq)``.
    frq:
        Narrow-feature frequencies in atomic units, shape ``(n_freq,)``.
    frq_continuum:
        Continuum frequencies in atomic units, shape ``(n_sma_freq,)``.
    start:
        Start time of the output grid in atomic units.
    end:
        End time of the output grid in atomic units.
    n_points:
        Number of time points in the output grid.
    intercept_value:
        Static offset for the narrow-feature signal, shape ``(3,)``.
    intercept_value_continuum:
        Static offset for the continuum signal, shape ``(3,)``.
    continuum:
        If ``True``, add the continuum signal to the narrow-feature signal
        and write to ``line_search_{x,y,z}.dipole.dat``; otherwise write
        ``line_search_{x,y,z}_select_only.dipole.dat``.
    """

    time, dt = np.linspace(start, end, n_points, retstep=True, endpoint=False) # in au

    if continuum==False:

        signal = generate_signal_tensor(np.transpose(amp), frq, time, intercept=intercept_value)

    else:

        signal = generate_signal_tensor(np.transpose(amp), frq, time, intercept=intercept_value)+ \
                generate_signal_tensor(np.transpose(amp_continuum), frq_continuum, time, intercept=intercept_value_continuum)

    # convert time [au] to [fs] for output

    time = time * AU_TO_FS

    if continuum==True:

       with open('line_search_x.dipole.dat', 'w') as f:
       
           f.write('# TIME-DEPENDENT DIPOLE MOMENT FROM RT-TDDFT' + "\n")
           f.write('# UNITS: [TIME] = fs | [DIPOLE] = a.u.' + "\n")
           f.write('# TIME | DIPOLE_X | DIPOLE_Y | DIPOLE_Z' + "\n")
          
           for i in range(len(time)):
               f.write('{:12f}'.format(time[i])+'   '+'{:8f}'.format(signal[i,0])+'   '+'{:8f}'.format(0.00000000000)+'   '+'{:.8f}'.format(0.00000000000)+"\n")
       
       with open('line_search_y.dipole.dat', 'w') as f:
       
           f.write('# TIME-DEPENDENT DIPOLE MOMENT FROM RT-TDDFT' + "\n")
           f.write('# UNITS: [TIME] = fs | [DIPOLE] = a.u.' + "\n")
           f.write('# TIME | DIPOLE_X | DIPOLE_Y | DIPOLE_Z' + "\n")
          
           for i in range(len(time)):
               f.write('{:12f}'.format(time[i])+'   '+'{:8f}'.format(0.00000000000)+'   '+'{:8f}'.format(signal[i,1])+'   '+'{:.8f}'.format(0.00000000000)+"\n")
       
       with open('line_search_z.dipole.dat', 'w') as f:
       
           f.write('# TIME-DEPENDENT DIPOLE MOMENT FROM RT-TDDFT' + "\n")
           f.write('# UNITS: [TIME] = fs | [DIPOLE] = a.u.' + "\n")
           f.write('# TIME | DIPOLE_X | DIPOLE_Y | DIPOLE_Z' + "\n")
          
           for i in range(len(time)):
               f.write('{:12f}'.format(time[i])+'   '+'{:8f}'.format(0.00000000000)+'   '+'{:8f}'.format(0.00000000000)+'   '+'{:.8f}'.format(signal[i,2])+"\n")

    else:

       with open('line_search_x_select_only.dipole.dat', 'w') as f:
       
           f.write('# TIME-DEPENDENT DIPOLE MOMENT FROM RT-TDDFT' + "\n")
           f.write('# UNITS: [TIME] = fs | [DIPOLE] = a.u.' + "\n")
           f.write('# TIME | DIPOLE_X | DIPOLE_Y | DIPOLE_Z' + "\n")
          
           for i in range(len(time)):
               f.write('{:12f}'.format(time[i])+'   '+'{:8f}'.format(signal[i,0])+'   '+'{:8f}'.format(0.00000000000)+'   '+'{:.8f}'.format(0.00000000000)+"\n")
       
       with open('line_search_y_select_only.dipole.dat', 'w') as f:
       
           f.write('# TIME-DEPENDENT DIPOLE MOMENT FROM RT-TDDFT' + "\n")
           f.write('# UNITS: [TIME] = fs | [DIPOLE] = a.u.' + "\n")
           f.write('# TIME | DIPOLE_X | DIPOLE_Y | DIPOLE_Z' + "\n")
          
           for i in range(len(time)):
               f.write('{:12f}'.format(time[i])+'   '+'{:8f}'.format(0.00000000000)+'   '+'{:8f}'.format(signal[i,1])+'   '+'{:.8f}'.format(0.00000000000)+"\n")
       
       with open('line_search_z_select_only.dipole.dat', 'w') as f:
       
           f.write('# TIME-DEPENDENT DIPOLE MOMENT FROM RT-TDDFT' + "\n")
           f.write('# UNITS: [TIME] = fs | [DIPOLE] = a.u.' + "\n")
           f.write('# TIME | DIPOLE_X | DIPOLE_Y | DIPOLE_Z' + "\n")
          
           for i in range(len(time)):
               f.write('{:12f}'.format(time[i])+'   '+'{:8f}'.format(0.00000000000)+'   '+'{:8f}'.format(0.00000000000)+'   '+'{:.8f}'.format(signal[i,2])+"\n")
