"""
Helper functions for the BYND line search and FHI-aims RT-TDDFT I/O.

Contains the regression-based amplitude update, the objective function,
signal generation, search-grid construction, and file I/O routines used
by the main optimisation loop.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score

from helpers_rt_tddft_fhiaims_utilities import DipoleData, FieldData

import matplotlib.pyplot as plt

import pprint

# Physical conversion constants (values taken from FHI-aims)
AU_TO_FS = 0.024188843265857   # atomic units of time → femtoseconds
HA_TO_EV = 27.211386245988     # Hartree → electron volts


def update_amplitudes_tensor(
    Y: np.ndarray,
    input_freq: np.ndarray,
    time: np.ndarray,
    off_diagonal: bool = False,
    intercept: bool = False,
    method: str = 'ridge',
    reg_coef: float = 0.1,
    ratio: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit oscillation amplitudes to a target signal via linear regression.

    Constructs a sine-wave design matrix from *input_freq* and *time*, then
    solves the multivariate regression problem

        Y_ij = X_il * A_lj

    where *Y* is the target signal, *X* the design matrix, and *A* the
    amplitude matrix to be found.

    Parameters
    ----------
    Y:
        Target RT-TDDFT signal, shape ``(n_time, n_targets)``.
        Pass only diagonal components when *off_diagonal* is ``False``,
        and only off-diagonal components when it is ``True``.
        The full external array should be ordered ``[xx, yy, zz, xy, xz, yz]``.
    input_freq:
        Frequencies involved in the regression, shape ``(n_freq,)``.
    time:
        Time grid, shape ``(n_time,)``.
    off_diagonal:
        ``True`` to fit off-diagonal polarizability components (allows
        negative amplitudes); ``False`` for diagonal (non-negative only).
    intercept:
        ``True`` to fit a static (DC) offset term.  Use when the dipole
        signal has a non-zero baseline.
    method:
        Regression algorithm: ``'ridge'``, ``'lasso'``, or ``'elasticnet'``.
    reg_coef:
        Regularisation strength (alpha).
    ratio:
        ElasticNet L1/L2 mixing ratio (ignored for other methods).

    Returns
    -------
    coef:
        Fitted amplitudes, shape ``(n_targets, n_freq)``.
    intercept_:
        Fitted intercept terms, shape ``(n_targets,)``.  All zeros when
        *intercept* is ``False``.

    References
    ----------
    Pemmaraju et al., J. Chem. Theory Comput. 2018, 14 (4), 1910–1927.
    """

    n_samples  = len(Y[:,0])
    n_features = len(input_freq)

    X = np.zeros((n_samples, n_features))
    for i in range(n_features):
        X[:,i] = -1.0 * np.sin(input_freq[i]*time)

    # Diagonal elements require non-negative amplitudes; off-diagonal do not.
    positive = not off_diagonal

    if method == 'ridge':
        clf = Ridge(alpha=reg_coef, fit_intercept=intercept, positive=positive, tol=1e-8)
    elif method == 'elasticnet':
        clf = ElasticNet(alpha=reg_coef, l1_ratio=ratio, fit_intercept=intercept, positive=positive, tol=0.001)
    elif method == 'lasso':
        clf = Lasso(alpha=reg_coef, fit_intercept=intercept, positive=True, tol=1e-3)
    else:
        print('ERROR: wrong regression method.')

    clf.fit(X, Y)

    return clf.coef_, clf.intercept_



def sort_amplitudes_tensor(amplitude_2D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return index arrays that sort frequencies by total amplitude.

    Parameters
    ----------
    amplitude_2D:
        Amplitude array, shape ``(n_targets, n_freq)``.

    Returns
    -------
    index_sort:
        Indices that reorder frequencies from highest to lowest summed
        absolute amplitude across all targets.
    index_undo_sort:
        Indices that restore the original frequency ordering after
        applying *index_sort*.
    """

    tmp = np.zeros(len(amplitude_2D[0,:]))

    for i in range(len(amplitude_2D[:,0])):
        tmp += np.abs(amplitude_2D[i,:])

    # -1. yields the highest one first
    index_sort = np.argsort(-1.* tmp)

    index_undo_sort = np.argsort(index_sort)


    return index_sort, index_undo_sort



def objective_tensor(
    predicted: np.ndarray,
    reference: np.ndarray,
    f_in: np.ndarray,
    f_sma: np.ndarray,
    method: str = 'RMSE',
    reg_coef: float = 1.0,
) -> float:
    """Evaluate the regularised loss between a predicted and reference signal.

    The loss has two components: a fidelity term measuring how well
    *predicted* matches *reference*, and a regularisation term that penalises
    large deviations of the current frequency from the initial SMA guess.

    Parameters
    ----------
    predicted:
        Model signal at the candidate frequency, shape ``(n_time, n_targets)``.
    reference:
        RT-TDDFT target signal, shape ``(n_time, n_targets)``.
    f_in:
        Sine wave evaluated at the current candidate frequency,
        shape ``(n_time,)``.
    f_sma:
        Sine wave evaluated at the initial SMA frequency (regularisation
        anchor), shape ``(n_time,)``.
    method:
        Loss function: ``'L1'``, ``'L2'``, ``'MAE'``, ``'MSE'``, or
        ``'RMSE'``.
    reg_coef:
        Regularisation strength (lambda).  A near-zero value effectively
        disables the frequency-deviation penalty.

    Returns
    -------
    float
        Scalar objective value for the current candidate frequency.
    """

    # for safety reasons
    objective = 0.0

    if method=='RMSE':

        for i in range(len(reference[0,:])):
            objective += np.sqrt(np.square(np.subtract(reference[:,i], predicted[:,i]))).mean()
        
        objective += reg_coef * np.sqrt(np.square(np.subtract(f_in, f_sma))).mean()


    elif method=='MSE': # for mean squared error
        for i in range(len(reference[0,:])):
            objective += np.square(np.subtract(reference[:,i], predicted[:,i])).mean()

        objective +=reg_coef * np.square(np.subtract(f_in, f_sma)).mean()


    elif method=='L2':
        for i in range(len(reference[0,:])):
            objective += np.sqrt(np.square(np.subtract(reference[:,i], predicted[:,i]))).sum()
        
        objective += reg_coef * np.sqrt(np.square(np.subtract(f_in, f_sma))).sum()

    
    elif method=='L1':
        for i in range(len(reference[0,:])):
            objective += np.abs(np.subtract(reference[:,i], predicted[:,i])).sum()
        
        objective += reg_coef * np.abs(np.subtract(f_in, f_sma)).sum()


    elif method=='MAE': # for mean absolute error
        for i in range(len(reference[0,:])):
            objective += np.abs(np.subtract(reference[:,i], predicted[:,i])).mean()
        
        objective += reg_coef * np.abs(np.subtract(f_in, f_sma)).mean()


    else:

        print("No valid choice for the objective function was provided, please use L1 or L2")

    return objective



def generate_signal_tensor(
    amp: np.ndarray,
    freq: np.ndarray,
    time: np.ndarray,
    dipole: bool = False,
    off_diagonal: bool = False,
    intercept: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct a multi-target signal from amplitudes and frequencies.

    Evaluates the model signal

        Y_ti = X_tj * Amp_ji

    where ``X_tj = -sin(freq_j * time_t)`` is the sine-wave design matrix.
    Note: *amp* must be passed as the **transpose** of the
    ``(n_targets, n_freq)`` array produced by :func:`update_amplitudes_tensor`,
    i.e. shape ``(n_freq, n_targets)``.

    Parameters
    ----------
    amp:
        Amplitude array, shape ``(n_freq, n_targets)``.
    freq:
        Frequencies, shape ``(n_freq,)``.
    time:
        Time grid in atomic units, shape ``(n_time,)``.
    dipole:
        Reserved for future use; currently unused.
    off_diagonal:
        Included for API symmetry; the signal formula is identical for
        diagonal and off-diagonal components.
    intercept:
        Static offset per target, shape ``(n_targets,)``.  Pass ``None``
        (or an array of length 1) to skip the intercept correction.

    Returns
    -------
    signal:
        Reconstructed signal, shape ``(n_time, n_targets)``.
    """

    n_features = len(freq)
    n_samples = len(time)

    X = np.zeros((n_samples, n_features))
    for i in range(n_features):
        X[:,i] = -1.0 * np.sin(freq[i]*time)

    signal = np.dot(X, amp)

    if np.size(intercept) > 1:
        for i in range(len(intercept)):
            signal[:,i] = signal[:,i] + intercept[i]

    return signal



def get_search_grid(f: float, rf: float, df: float) -> np.ndarray:
    """Build a symmetric frequency search grid centred on *f*.

    Parameters
    ----------
    f:
        Centre frequency (atomic units).
    rf:
        Half-width of the search window (atomic units).
    df:
        Grid spacing (atomic units).

    Returns
    -------
    np.ndarray
        Evenly spaced grid over ``[f - rf, f + rf]`` with step *df*.
    """

    N = (2.* rf)/df + 1

    N = int(N)

    grid = np.linspace(f-rf,f+rf, N)

    return grid



def read_RT_TDDFT_data(t_unit_au: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load RT-TDDFT dipole data from FHI-aims output files.

    Wrapper around :class:`DipoleData` that reads the three dipole files
    produced by separate x, y, and z field-pulse simulations and assembles
    them into a single array ordered as
    ``[xx, xy, xz, yx, yy, yz, zx, zy, zz]``.

    Expected files in the working directory::

        x.rt-tddft.dipole.dat
        y.rt-tddft.dipole.dat
        z.rt-tddft.dipole.dat

    Parameters
    ----------
    t_unit_au:
        If ``True``, convert the time axis from femtoseconds to atomic units.

    Returns
    -------
    dipole_xyz:
        Dipole tensor signal, shape ``(n_time, 9)``.
    tddft_time:
        Time grid in femtoseconds (default) or atomic units when
        *t_unit_au* is ``True``, shape ``(n_time,)``.
    """

    print('loading RT-TDDFT data')

    dip_x = DipoleData('x.rt-tddft.dipole.dat', "dipole_x")
    dip_y = DipoleData('y.rt-tddft.dipole.dat', "dipole_y")
    dip_z = DipoleData('z.rt-tddft.dipole.dat', "dipole_z")

    # in total we have 9 signals!
    dipole_xyz = np.zeros((len(dip_x.data['x']),9))

    dipole_xyz[:,0] = np.asarray(dip_x.data['x'])
    dipole_xyz[:,1] = np.asarray(dip_x.data['y'])
    dipole_xyz[:,2] = np.asarray(dip_x.data['z'])

    dipole_xyz[:,3] = np.asarray(dip_y.data['x'])
    dipole_xyz[:,4] = np.asarray(dip_y.data['y'])
    dipole_xyz[:,5] = np.asarray(dip_y.data['z'])

    dipole_xyz[:,6] = np.asarray(dip_z.data['x'])
    dipole_xyz[:,7] = np.asarray(dip_z.data['y'])
    dipole_xyz[:,8] = np.asarray(dip_z.data['z'])

    tddft_time = np.asarray(dip_x.data['t'])

    if t_unit_au==True:
        tddft_time = tddft_time / AU_TO_FS

    return dipole_xyz, tddft_time



def read_sma_data(
    file_osci: str | None,
    file_trans: str | None,
    frq_eV: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Read excitation frequencies, oscillator strengths, and transition dipoles from SMA output.

    Parameters
    ----------
    file_osci:
        Path to the SMA oscillator-strength file.  Returns ``None`` if
        ``None`` is passed.
    file_trans:
        Path to the SMA transition-dipole-moment file.  Returns ``None``
        if ``None`` is passed.
    frq_eV:
        If ``True``, read frequencies from the eV column; otherwise from
        the Hartree column (default).

    Returns
    -------
    frq:
        Excitation frequencies in Hartree (or eV when *frq_eV* is ``True``),
        shape ``(n_exc,)``.
    ostren:
        Oscillator strengths, shape ``(n_exc,)``.
    trans_mom:
        Transition dipole moments ``[x, y, z]`` per excitation,
        shape ``(n_exc, 3)``.
    """

    print('loading SMA data')

    if file_osci is None:
        return

    if file_trans is None:
        return

    trans_mom_x = []
    trans_mom_y = []
    trans_mom_z = []
 
    frq = []
    ostren = []

    if frq_eV==False: # if we wanna have eV or ha
        with open(file_osci,'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                else:
                    frq.append(float(line.split()[2]))
                    ostren.append(float(line.split()[4]))
    else:
        with open(file_osci,'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                else:
                    frq.append(float(line.split()[3]))
                    ostren.append(float(line.split()[4]))


    with open(file_trans,'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            else:
                trans_mom_x.append(float(line.split()[4]))
                trans_mom_y.append(float(line.split()[5]))
                trans_mom_z.append(float(line.split()[6]))


    # we only return numpy arrays
    frq         = np.asarray(frq)
    ostren      = np.asarray(ostren)
    trans_mom_x = np.asarray(trans_mom_x)
    trans_mom_y = np.asarray(trans_mom_y)
    trans_mom_z = np.asarray(trans_mom_z)

    trans_mom = np.zeros((len(trans_mom_x),3))

    trans_mom[:,0] = trans_mom_x
    trans_mom[:,1] = trans_mom_y
    trans_mom[:,2] = trans_mom_z


    return frq, ostren, trans_mom



def filter_frequencies(
    frequencies: np.ndarray,
    trans_mom_data: np.ndarray,
    threshold: float,
    n_frequencies: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain only SMA frequencies with a significant transition dipole moment.

    An excitation is kept when at least one of its x, y, or z transition
    dipole moment components exceeds *threshold* in absolute value.

    Parameters
    ----------
    frequencies:
        SMA excitation frequencies, shape ``(n_exc,)``.
    trans_mom_data:
        Transition dipole moments, shape ``(n_exc, 3)``.
    threshold:
        Minimum absolute transition dipole moment required for inclusion.
    n_frequencies:
        Reserved for future use (maximum number of frequencies to keep).

    Returns
    -------
    freq:
        Filtered frequencies, shape ``(n_kept,)``.
    trans_mom:
        Transition dipole moments for the kept excitations,
        shape ``(n_kept, 3)``.
    """

    #TODO: restrict the maximum number of frequencies

    index = []

    for i in range(len(frequencies)):
        # uncomment for 2D
        if(np.abs(trans_mom_data[i,0]) > threshold or np.abs(trans_mom_data[i,1]) > threshold or np.abs(trans_mom_data[i,2]) > threshold):
            index.append(i)

    trans_mom = np.zeros((len(index),3))

    freq = frequencies[index]
    trans_mom[:,0] = trans_mom_data[index,0]
    trans_mom[:,1] = trans_mom_data[index,1]
    trans_mom[:,2] = trans_mom_data[index,2]

    return freq, trans_mom



def write_to_file(
    file_name: str,
    amp: np.ndarray,
    freq: np.ndarray,
    off_diag: bool = False,
    intercept: np.ndarray | None = None,
) -> None:
    """Write optimised frequencies and amplitudes to a plain-text file.

    Parameters
    ----------
    file_name:
        Output file path.
    amp:
        Amplitudes.  Either a 1-D array ``(n_freq,)`` or a 2-D array
        ``(n_targets, n_freq)``.
    freq:
        Frequencies corresponding to the columns of *amp*, shape
        ``(n_freq,)``.
    off_diag:
        If ``True``, write off-diagonal header labels (xy, xz, yz);
        otherwise write diagonal labels (xx, yy, zz).
    intercept:
        Static offset per target, shape ``(n_targets,)``.  Written as a
        comment line when its length exceeds 1.
    """


    # check if amp is a 2D array

    if off_diag == False:
        if len(amp.shape)==2:
            
            with open(file_name, 'w') as f:

                if np.size(intercept) > 1: # we need to write an intercept to file as well
                    f.write('# intercept: ' + '{:5f}'.format(intercept[0])+'   '+'{:3f}'.format(intercept[1])+'   '+'{:3f}'.format(intercept[2])+"\n")
      

                f.write('# frequency [au]' + '   amp_xx [au]' + '   amp_yy [au]' + '   amp_zz [au]' + "\n")
       
                for i in range(len(freq)):
                    f.write('{:5f}'.format(freq[i])+'   '+'{:3f}'.format(amp[0,i])+'   '+'{:3f}'.format(amp[1,i])+'   '+'{:.3f}'.format(amp[2,i])+"\n")
       
        else: # amp is 1D array
       
            with open(file_name, 'w') as f:
       
                f.write('# frequency [au]' + '   amp [au]' + "\n")
       
                for i in range(len(freq)):
                    f.write('{:5f}'.format(freq[i])+'   '+'{:3f}'.format(amp[0,i])+ "\n")
                    
    else: # we have off diagonal amplitudes
        
        # off diagonal signal also just have sine contributions
        # however their amplitude is negative
        if len(amp.shape)==2:
            
            with open(file_name, 'w') as f:

                if np.size(intercept) > 1: # we need to write an intercept to file as well
                    f.write('# intercept: ' + '{:5f}'.format(intercept[0])+'   '+'{:3f}'.format(intercept[1])+'   '+'{:3f}'.format(intercept[2])+"\n")

       
                f.write('# frequency [au]' + '   amp_xy [au]' + '   amp_xz [au]' + '   amp_yz [au]' + "\n")
       
                for i in range(len(freq)):
                    f.write('{:5f}'.format(freq[i])+'   '+'{:3f}'.format(amp[0,i])+'   '+'{:3f}'.format(amp[1,i])+'   '+'{:.3f}'.format(amp[2,i])+"\n")
       
        else: # amp is 1D array
            
            print('ERROR: off diagonal amplitudes should be a 2D array.')
            quit()



def print_init_messages(
    n_iterations: int,
    switch_to_off: int,
    search_radius: float,
    search_radius_init: float,
    grid_spacing: float,
    input_freq: np.ndarray,
    len_signal: int,
) -> None:
    """Print a formatted summary of the optimisation settings to stdout.

    Parameters
    ----------
    n_iterations:
        Total number of line-search iterations.
    switch_to_off:
        Iteration number at which off-diagonal optimisation begins.
    search_radius:
        Frequency search half-width used in later iterations.
    search_radius_init:
        Frequency search half-width used in the first iteration.
    grid_spacing:
        Step size of the frequency search grid.
    input_freq:
        Initial frequency array (printed in eV), shape ``(n_freq,)``.
    len_signal:
        Number of time steps in the target signal.
    """

    print('|----------------------------------------------------------')
    print('| Starting frequency optimization')
    print('|----------------------------------------------------------')
    print('|')
    print('| Number of iterations: {0:2d}'.format(n_iterations))
    print('| Switch to off diagonal at iterations: {0:2d}'.format(switch_to_off))
    print('|')
    print('| Initial search radius: {0:f}'.format(search_radius_init))
    print('| Search radius: {0:f}'.format(search_radius))
    print('| Grid spacing: {0:f}'.format(grid_spacing))
    print('| Length of target: {0:2d}'.format(len_signal))
    print('| Number of frequencies: {0:2d}'.format(len(input_freq)))
    print('|')
    print('| Printing input frequencies next ... [E=h_quer*w]')
    print('|')
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print('|',input_freq*HA_TO_EV)
    print('|')
    print('| Initialization End')
    print('|----------------------------------------------------------')



def print_iteration(iteration: int, freq: np.ndarray, amp: np.ndarray) -> None:
    """Print the current frequency and amplitude values for one iteration.

    Parameters
    ----------
    iteration:
        Zero-based iteration index (displayed as ``iteration + 1``).
    freq:
        Current frequencies in atomic units, shape ``(n_freq,)``
        (printed converted to eV).
    amp:
        Current amplitude array, shape ``(n_targets, n_freq)``.
    """

    print('|')
    print('| Current iteration: {0:2d}'.format(iteration+1))
    print('|')
    print('| Printing frequencies next ... [E=h_quer*w]')
    print('|')
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print('|',freq*HA_TO_EV)
    print('|')
    print('| Printing amplitudes next ... [xx],[yy],[zz]')
    print('|')
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print('|',amp)
    print('|')
    print('| Current iteration End')
    print('|----------------------------------------------------------')
