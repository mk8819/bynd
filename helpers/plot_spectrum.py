import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib import rcParams, cycler
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import find_peaks



def read_data(file_name):
    """ 
    We read the final spectrum from a file
    consisting only of two columns
    """
    omega = []
    abs_str = []

    with open(file_name, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            elif line.startswith(' #'):
                continue
            else:
                omega.append(float(line.split()[0]))
                abs_str.append(float(line.split()[1]))

    omega_array = np.asarray(omega)
    abs_str_array = np.asarray(abs_str)

    return omega_array, abs_str_array


def plot_simple_spectrum():
   # settings
   rcParams['font.family'] = 'sans-serif'
   rcParams['font.sans-serif'] = ['Arial']
   rcParams['font.size'] = 24
   rcParams['axes.linewidth'] = 1.8
   rcParams['axes.labelpad'] = 10.0
   #plot_color_cycle = cycler('color', ['000000', '0000FE', 'FE0000', '008001', 'FD8000', '8c564b', 
   #                                    'e377c2', '7f7f7f', 'bcbd22', '17becf', '#1181C8'])
   #rcParams['axes.prop_cycle'] = plot_color_cycle
   rcParams['axes.xmargin'] = 0
   rcParams['axes.ymargin'] = 0
   rcParams.update({"figure.figsize" : (6.4,4.8),
                    "figure.subplot.left" : 0.177, "figure.subplot.right" : 0.946,
                    "figure.subplot.bottom" : 0.156, "figure.subplot.top" : 0.965,
                    "axes.autolimit_mode" : "round_numbers",
                    "xtick.major.size"     : 7,
                    "xtick.minor.size"     : 3.5,
                    "xtick.major.width"    : 1.1,
                    "xtick.minor.width"    : 1.1,
                    "xtick.major.pad"      : 5,
                    "xtick.minor.visible" : True,
                    "ytick.major.size"     : 7,
                    "ytick.minor.size"     : 3.5,
                    "ytick.major.width"    : 1.1,
                    "ytick.minor.width"    : 1.1,
                    "ytick.major.pad"      : 5,
                    "ytick.minor.visible" : True,
                    "lines.markersize" : 10,
                    "lines.markerfacecolor" : "none",
                    "lines.markeredgewidth"  : 0.8})
   
   
   frq2, line_search = read_data('line-search-tddft.abs_strength.dat')
   frq1, rt_tddft = read_data('rt-tddft.abs_strength.dat')
   
   
   # start work on plotting
   
   plt.plot(frq1, np.abs(rt_tddft), lw=3.0, color='black', label='exact')
   plt.plot(frq2, np.abs(line_search), lw=3., color='C1', label='final')
   
   
   plt.xlabel('E [eV]')
   plt.ylabel('Int. [au]')
   plt.ylim(0., 4.)
   plt.xlim(0., 20.)
   plt.legend()
   plt.savefig('spectrum_bynd_vs_exact.pdf')
   #plt.show()

