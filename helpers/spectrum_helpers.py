import numpy as np
from helpers_line_search import read_sma_data, generate_signal_tensor
import matplotlib.pyplot as plt



def get_line_search_dipole_data(amp_file, amp_continuum_file, continuum=False):

    """
    The wrapper script optimize_frequencies writes the final amplitdes and
    frequencies to a file. This function can read from these files.
    This I/O might not be very elegant but we can do postprocessing
    without recalculating the frequencies.

    Input:

    o) amp_file:           file containing the amplitudes of the narrow features

    o) amp_continuum_file: file containing continuum amplitudes

    o) continuum:          boolean if the continuum should be read or not


    Output:

    o) frq:                       frequencies of narrow features

    o) frq_continuum:             frequencies of continuum

    o) amp:                       amplitudes of narrow features

    o) amp_continuum:             amplitudes of continuum

    o) intercept_value:           intercept of narrow feature signal

    o) intercept_value_continuum: intercept of continuum signal

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


def generate_signal_and_save_to_file(amp, amp_continuum, frq, frq_continuum, start, end, n_points, intercept_value=None, intercept_value_continuum=None, continuum=False):

    """
    This functions generates a signal out of the line search frequencies and amplitudes.
    It saves the signal to file and accounts for the right FHIaims format.
    """

    time, dt = np.linspace(start, end, n_points, retstep=True, endpoint=False) # in au

    if continuum==False:

        signal = generate_signal_tensor(np.transpose(amp), frq, time, intercept=intercept_value)

    else:

        signal = generate_signal_tensor(np.transpose(amp), frq, time, intercept=intercept_value)+ \
                generate_signal_tensor(np.transpose(amp_continuum), frq_continuum, time, intercept=intercept_value_continuum)

    # convert time [au] to [fs] for output

    time = time * 0.02418884254 

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
