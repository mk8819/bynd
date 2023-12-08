import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score

from helpers_rt_tddft_fhiaims_utilities import DipoleData, FieldData

import matplotlib.pyplot as plt

import pprint


"""
This file contains all helper functions for the line search and
functions for I/O of FHIaims RT-TDDFT data.

"""

def update_amplitudes_tensor(Y, input_freq, time, off_diagonal=False, intercept=False, method='ridge', reg_coef=0.1, ratio=0.9):

    """
    Performes a linear regression to fit the
    amplitudes to the target signal
    does this as a multi_variate regression

    Y_ij = X_il * A_lj

    input:

    o) Y : this is the real-time target signal
           if off-diagonal=True target signal is only 
           the off diagonal elements
           if off-diagonal=False Y should only contain
           the diagonal elements.

           the total target array outside should be ordered
           in the following way

           Y[xx, yy, zz, xy, xz, yz]

           Note that xy = yx, xz = zx, zy = yz due to
           symmetry of the polarizability tensor

    o) input_freq: all the frequencies involved int the line
                   search, 1D array

    o) off_diagonal: boolean, optional, specifies if diagonal or 
                     off diagonal elements should be fitted

    o) intercept: boolean, optional
                  this should be set to true if the electric dipole
                  has a static component

    output:

    o) clf.coef_: amplitueds (n_targets x n_features)-array

    o) clf.intercept_: intercept terms (n_targets)-array


    The dipole osscilates with sine waves only:

    Journal of Chemical Theory and Computation 2018 14 (4), 1910-1927

    """

    if off_diagonal==False:

        n_samples  = len(Y[:,0])
        n_targets  = len(Y[0,:])
        n_features = len(input_freq)

        X = np.zeros((n_samples,n_features))

        for i in range(n_features):
            X[:,i] = -1.0 * np.sin(input_freq[i]*time)

        if method=='ridge':
            clf = Ridge(alpha=reg_coef, fit_intercept=intercept, positive=True, tol=1e-8)
        elif method=='elasticnet':
            clf = ElasticNet(alpha=reg_coef, l1_ratio=ratio, fit_intercept=intercept, positive=True, tol=0.001)            
        elif method=='lasso':
            clf = Lasso(alpha=reg_coef, fit_intercept=intercept, positive=True, tol=1e-3)
        else:
            print('ERROR: wrong regression method.')


        clf.fit(X, Y)

    else: # fit the off_diagonal elements

        n_samples  = len(Y[:,0])
        n_targets  = len(Y[0,:])
        n_features = len(input_freq)

        X = np.zeros((n_samples,n_features)) # we have also only sine here

        # the first part are all the sin frequencies
        for i in range(n_features):
            X[:,i] = -1.0 * np.sin(input_freq[i]*time)

        # non-negative constraint does only apply for the off-diagonal
        if method=='ridge':
            clf = Ridge(alpha=reg_coef, fit_intercept=intercept, positive=False, tol=1e-8)
        elif method=='elasticnet':
            clf = ElasticNet(alpha=reg_coef, l1_ratio=ratio, fit_intercept=intercept, positive=False, tol=0.001)            
        elif method=='lasso':
            clf = Lasso(alpha=reg_coef, fit_intercept=intercept, positive=True, tol=1e-3)
        else:
            print('ERROR: wrong regression method.')


        clf.fit(X, Y)

    # if intercept = False the intercept = 0.0

    return clf.coef_, clf.intercept_



def sort_amplitudes_tensor(amplitude_2D):

    """
    Returns the indexing array which sorts
    the frequencies according to their amplitude

    input:

    o) amplitudes_2D: (n_targets x n_features)-array
                      which contains the amplitudes

    output:
    
    o) index_sort: the array which sorts the frequencies
                   the one with the highest amplitude
                   comes first

    o) index_undo_sort: restors the original sorting
                        of the frequencies

    """

    tmp = np.zeros(len(amplitude_2D[0,:]))

    for i in range(len(amplitude_2D[:,0])):
        tmp += np.abs(amplitude_2D[i,:])

    # -1. yields the highest one first
    index_sort = np.argsort(-1.* tmp)

    index_undo_sort = np.argsort(index_sort)


    return index_sort, index_undo_sort



def objective_tensor(predicted, reference, f_in, f_sma, method='RMSE', reg_coef=1.0):

    """
    calculates the error between prediction and reference signal
    using the L1 or L2 norm.

    In addition it also adds a penalty if frequencies move to far away from 
    the initial SMA guess.

    Input:
    o) prediction, N dimensional numpy array
    o) reference, N dimensional numpy array
    o) method, string, either L1 or L2 or RMSE
    o) reg_coef: regularization coefficients usually called lambda
    o) f_in: single sine wave with current frequency
    o) f_sma: single sine wave with initial frequency

    Output:
    o) objective, the value of the objective function for the 
                  two input frequencies
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



def generate_signal_tensor(amp, freq, time, dipole=False, off_diagonal=False, intercept=None):

    """
    Generates the predicted signal if the amplitudes are
    stored in a 2D array.

    The signal is created by

    Y_ti = X_tj * Amp_ji

    however Amp is stored as Amp_ij so it needs to be
    transposed before the function call.

    input:

    o) amp: 2D amplitude array
    o) freq: array which contains all frequencies
    o) time: time in au
    o) intercept: the intercept from the amplitude fitting
                  (n_target)-array
    
    output:

    o) signal: the current signal which is under optimization
    """

    n_features = len(freq)
    n_samples = len(time)

    if off_diagonal==False:
        # this is our design matrix

        X = np.zeros((n_samples,n_features))

        # TODO: we could actually store this
        for i in range(n_features):
            X[:,i] = -1.0 *  np.sin(freq[i]*time)

        signal = np.dot(X,amp)

    else:

        X = np.zeros((n_samples, n_features))

        for i in range(n_features): 
            X[:,i] = -1.0 * np.sin(freq[i]*time)

        signal = np.dot(X,amp)


    if np.size(intercept) > 1:
        for i in range(len(intercept)):
            signal[:,i] = signal[:,i] + intercept[i]


    return signal



def get_search_grid(f, rf, df):

    """
    This function returns a symmetric search grid around an initial frequency f
    
    Input:
    o) f: frequency
    o) rf: search radius around f
    o) df: search grid spacing

    """

    N = (2.* rf)/df + 1

    N = int(N)

    grid = np.linspace(f-rf,f+rf, N)

    return grid



def read_RT_TDDFT_data(t_unit_au=False):

    """
    This function reads in all necessary input from the RT-TDDFT data.
    It is basically a wrapper for the FHI-aims RT-TDDFT utilities
    contained in the helpers_rt-tdddft-fhiaims-utilities.py file.

    This functions assumes that x, y and z dipole are stored 
    seperately in

      x.rt-tddft.dipole.dat
      y.rt-tddft.dipole.dat
      z.rt-tddft.dipole.dat

    The external field has to be stored in 

      x.rt-tddft.ext-field.dat

    Input:

    o) t_unit_au: boolean switch to have the time in au instead of
                  fs

    Output:

    o) dipole_xyz: a compact array containing all dipole signals
    o) tddft_time: default unit depends on FHI-aims output
                   can be either fs or au

    """

    print('loading RT-TDDFT data')

    # for converting fs in au
    au = 0.024188843265857 # this value have been taken from FHI-aims

    dip_x = DipoleData('x.rt-tddft.dipole.dat', "dipole_x")
    dip_y = DipoleData('y.rt-tddft.dipole.dat', "dipole_y")
    dip_z = DipoleData('z.rt-tddft.dipole.dat', "dipole_z")

    #field = ['x.rt-tddft.ext-field.dat',None,None]

    #fld_x = FieldData(field[0], "field_x", True)
    #fld_y = FieldData(field[1], "field_y", True) if field[1] is not None else None
    #fld_z = FieldData(field[2], "field_z", True) if field[2] is not None else None

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
        tddft_time = tddft_time/au

    return dipole_xyz, tddft_time



def read_sma_data(file_osci, file_trans, frq_eV=False):

    """
    just for reading in the ouput from the SMA

    The obtained frequency or excitations are read in in hartree
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



def filter_frequencies(frequencies, trans_mom_data, threshold, n_frequencies=None):

    """
    This routine filters the sma frequencies according to the
    trans. dipole moment. Only frequencies which actually
    show a trans. dipole moment are included. And 
    which are above a certain amplitude threshold
    
    input:
    
    o) frequencies: SMA frequencies
    o) trans_mom_data: SMA transition dipole moment
    o) threshold: threshold for the transition dipole moment

    output:

    o) freq: filtered frequencies
    o) trans_mom: filtered transition diople moments

    """

    #TODO: restrict the maximum number of frequencies

    index = []

    for i in range(len(frequencies)):
        # uncomment for 2D
        if(np.abs(trans_mom_data[i,0]) > threshold or np.abs(trans_mom_data[i,1]) > threshold or np.abs(trans_mom_data[i,2]) > threshold):
        #if(np.abs(trans_mom_data[i,0]) > threshold):
            index.append(i)

    trans_mom = np.zeros((len(index),3))

    freq = frequencies[index]
    trans_mom[:,0] = trans_mom_data[index,0]
    trans_mom[:,1] = trans_mom_data[index,1]
    trans_mom[:,2] = trans_mom_data[index,2]

    return freq, trans_mom



def write_to_file(file_name, amp, freq, off_diag=False, intercept=None):
    
    """
    writes frequencies and amplitues to a file which is
    specified by file_name

    input:

    o) file_name
    o) amp: Amplitueds, can be a vector or a 2D array
    o) freq: frequencies corresponding to the amplitudes

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



def print_init_messages(n_iterations, switch_to_off, search_radius, search_radius_init, grid_spacing, input_freq, len_signal):

    """
    This routine prints initial messages and important settings to std out
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
    print('|',input_freq*27.211384500)
    print('|')
    print('| Initialization End')
    print('|----------------------------------------------------------')



def print_iteration(iteration, freq, amp):

    """
    This routine prints the current iteration results
    """

    print('|')
    print('| Current iteration: {0:2d}'.format(iteration+1))
    print('|')
    print('| Printing frequencies next ... [E=h_quer*w]')
    print('|')
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print('|',freq*27.211384500)
    print('|')
    print('| Printing amplitudes next ... [xx],[yy],[zz]')
    print('|')
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print('|',amp)
    print('|')
    print('| Current iteration End')
    print('|----------------------------------------------------------')
