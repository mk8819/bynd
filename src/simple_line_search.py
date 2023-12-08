"""
This routine does not depend on the underlying electronic structure code.
One only need to provide the correct target arrays.

"""
### general imports
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

### imports from helper routines
from helpers_line_search import get_search_grid
from helpers_line_search import update_amplitudes_tensor, sort_amplitudes_tensor, generate_signal_tensor, \
        objective_tensor



def perform_line_search_tensor_off_diagonal(target, input_a, input_a_off, input_f, time_grid, reference_f, rf=0.5, df=0.01, \
        amplitude_only=False, diagonal_only=True, diag_dim=3, calc_intercept=False, reg_method='ridge', alpha_value=0.1, l1_ratio_value=0.8, \
        initial=None):

    """
    Target is now a 2D array, and it handles also off_diagonal elements.

    input:

    o) target: 2D array of RT-TDDFT signals first dimension is time
               second dimension is n_targets long (time,n_targets).
               n_targets have to be ordered in the following way:

               [xx, yy, zz, xy, xz, yz]

    o) input_a: amplitudes for xx, yy, zz (n_target, n_frequency) array

    o) input_a_off: amplitudes for xy, xz, yz (n_target, 2*n_frequency) array
                    Note, that the off-diagonal elements contain sin and cos
                    thus we have twice the amount of amplitudes.

    o) input_f: 1D array of all the frequencies which should take part in the 
                line search

    o) time_grid: 1D array of time, has to be as long as the target time dimension

    o) reference_f: 1D, the initial sma frequencies

    o) rf: radius of the line search

    o) df: search grid spacing in the line search

    o) amplitude_only: boolean, defines if only amplitudes should be fitted
                       or not

    o) diagonal_only: boolean, defines if only diagaonal elements should be 
                      considered or if also off-diagonal elements should be
                      included in the line search

    o) diag_dim: specifies how many diagonal elements are present,
                 set this to 1 and diagonal_only to True for considering
                 only one target 

    output:

    o) work_f: optimized frequencies, 1D array, original ordering

    o) work_a: optimized amplitudes of all diagonal elements,
               2D array (n_targets, n_features), original ordering

    o) work_a_off: optimized amplitudes of all off diagonal elements,
                   2D array (n_targets, 2*n_features), original ordering
                   if diagonal_only is true this is only a dummy integer

    """


    # create working arrays for frequencies and amplitudes
    # this is not really necessary but we do it for safety
    # later we may want to derive a score value to 
    # judge how good the solution is
    
    work_a = input_a
    work_a_off = input_a_off # off-diagonal has twice the amount of featrues as we have cos and sin
    work_f = input_f

    target_work = target
    reference_f_work = reference_f

    stime = time.time()

    # amplitude update with linear regression
    if diagonal_only==True:
        work_a, intercept_on = update_amplitudes_tensor(target_work[:,:diag_dim], work_f, time_grid, off_diagonal=False, intercept=calc_intercept, method=reg_method, reg_coef=alpha_value, ratio=l1_ratio_value)
        if initial=='random':
            max_ele = np.amax(work_a)
            work_a = work_a * 0.9 + np.random.rand(len(work_a[:,0]), len(work_a[0,:]))*0.1*max_ele
    else:
        work_a, intercept_on = update_amplitudes_tensor(target_work[:,:diag_dim], work_f, time_grid, off_diagonal=False, intercept=calc_intercept, method=reg_method, reg_coef=alpha_value, ratio=l1_ratio_value)
        work_a_off, intercept_off = update_amplitudes_tensor(target_work[:,diag_dim:], work_f, time_grid, off_diagonal=True, intercept=calc_intercept, method=reg_method, reg_coef=alpha_value, ratio=l1_ratio_value)


    # only amplitude fit
    if amplitude_only==True:

        if diagonal_only==True:

            dummy = 0
            return work_f, work_a, dummy # we return also a integer dummy

        else:

            return work_f, work_a, work_a_off

    etime = time.time()
    print('| Time for amplitude fitting:', time.strftime("%H:%M:%S", time.gmtime(etime-stime)))

    # sort amplitudes and frequencies, frequencies with highest amplitude first
    # we only sort according to the diagonal amplitudes for now
    index_sort, index_undo_sort = sort_amplitudes_tensor(work_a)

    work_f = work_f[index_sort]
    reference_f_work = reference_f_work[index_sort]

    work_a = work_a[:,index_sort]

    # we need to sort work_a_off too
    if diagonal_only==False:
        work_a_off = work_a_off[:,index_sort]

    stime = time.time()
    elapstime_signal = 0.0
    elapstime_objective = 0.0

    for i in range(len(work_f)):

            search_grid = get_search_grid(work_f[i], rf, df)
            #search_grid = get_search_grid(reference_f_work[i], rf, df)
            objective_tmp = np.zeros_like(search_grid)

            if diagonal_only==True:

                work_a_i = 0.0
                for k in range(len(work_a[:,0])):
                    work_a_i += work_a[k,i]

                for j in range(len(search_grid)):

                    work_f[i] = search_grid[j]

                    stime_signal = time.time()

                    prediction = generate_signal_tensor(np.transpose(work_a), work_f, time_grid, dipole=False, intercept=intercept_on)

                    etime_signal = time.time()
                    elapstime_signal += etime_signal-stime_signal

                    f_i_signal = -1.*np.sin(work_f[i] * time_grid)# * work_a_i
                    reference_signal_i = -1.*np.sin(reference_f_work[i] * time_grid)# * work_a_i

                    stime_objective = time.time()

                    # in this example we calculate the loss function with L1 norm feel free to change
                    objective_tmp[j] = objective_tensor(prediction, target_work[:,:diag_dim], f_i_signal, reference_signal_i,\
                            method="L1", reg_coef=0.00001) # reg_coef is essentially zero we just do not set it to zero cause of numerics

                    etime_objective = time.time()
                    elapstime_objective += etime_objective-stime_objective



                index_min = np.argmin(objective_tmp)


                work_f[i] = search_grid[index_min]

                if initial=='random':
                    continue
                else:
                    work_a, intercept_on = update_amplitudes_tensor(target_work[:,:diag_dim], work_f, time_grid, off_diagonal=False, intercept=calc_intercept, method=reg_method, reg_coef=alpha_value, ratio=l1_ratio_value)

            else: # both diagonal and off-diagonal

                work_a_i = 0.0
                for k in range(len(work_a[:,0])):
                    work_a_i += work_a[k,i]
                    work_a_i += np.abs(work_a_off[k,i])
                    # we can have negative amplitues in work_a_off

                    for j in range(len(search_grid)):

                        work_f[i] = search_grid[j]

                        # we have now two predictions one for the diagonal and one for off-diagonal
                        stime_signal = time.time()

                        prediction_on = generate_signal_tensor(np.transpose(work_a), work_f, time_grid, dipole=False, off_diagonal=False, intercept=intercept_on)
                        prediction_off = generate_signal_tensor(np.transpose(work_a_off), work_f, time_grid, dipole=False, off_diagonal=True, intercept=intercept_off)

                        etime_signal = time.time()
                        elapstime_signal += etime_signal-stime_signal

                        # we put both together to create the prediction array with entries like the full target array
                        prediction = np.hstack((prediction_on, prediction_off))

                        f_i_signal = -1.*np.sin(work_f[i] * time_grid) #* work_a_i
                        reference_signal_i = -1.*np.sin(reference_f_work[i] * time_grid) #* work_a_i


                        # here we have now the full target array
                        stime_objective = time.time()

                        # in this example we calculate the loss function with L1 norm feel free to change
                        objective_tmp[j] = objective_tensor(prediction, target_work, f_i_signal, reference_signal_i,\
                                method="L1", reg_coef=0.00001) # was 0.000001

                        etime_objective = time.time()
                        elapstime_objective += etime_objective-stime_objective



                index_min = np.argmin(objective_tmp)

                work_f[i] = search_grid[index_min]

                work_a, intercept_on = update_amplitudes_tensor(target_work[:,:diag_dim], work_f, time_grid, off_diagonal=False, intercept=calc_intercept, method=reg_method, reg_coef=alpha_value, ratio=l1_ratio_value)
                work_a_off, intercept_off = update_amplitudes_tensor(target_work[:,diag_dim:], work_f, time_grid, off_diagonal=True, intercept=calc_intercept, method=reg_method, reg_coef=alpha_value, ratio=l1_ratio_value)



    etime = time.time()
    print('| Total time for line search:', time.strftime("%H:%M:%S", time.gmtime(etime-stime)))
    print('| Time for signal generation:', time.strftime("%H:%M:%S", time.gmtime(elapstime_signal)))
    print('| Time for loss function eval.:', time.strftime("%H:%M:%S", time.gmtime(elapstime_objective)))


    # restore original ordering in arrays
    if diagonal_only==True:

        work_f = work_f[index_undo_sort]
        work_a = work_a[:,index_undo_sort]
        work_a_off = 0 # return an integer

    else:

        work_f = work_f[index_undo_sort]
        work_a = work_a[:,index_undo_sort]
        work_a_off = work_a_off[:,index_undo_sort]


    return work_f, work_a, work_a_off
# -------------------------------------------------------------------------------------------
