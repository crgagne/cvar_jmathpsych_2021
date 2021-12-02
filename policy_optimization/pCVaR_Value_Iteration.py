from scipy.io import loadmat
import numpy as np
from scipy.optimize import minimize
import os
import multiprocessing
from multiprocessing import Pool
import time
from itertools import starmap
import pickle

import sys
sys.path.append('../shared')

from pCVaR_Shared import apply_pCVaR_max_operator

def pCVaR_VI(V0,Ns,Ny,Na,P_full,gamma,y_set,r,
            check_pointwise_iters=3,
            pointwise_window=3,
            min_multi_starts=2,
            max_multi_starts=5,
            max_iters=100,
            verbose=True,
            converg_thresh = 0.01,
            converg_thresh_pr = 0.1,
            converg_thresh_nonpr = 0.2,
            yi_pr=None,
            converg_type = 'full_eval'):

    '''Runs pCVaR Value Iteration Until Convergence
    '''

    start_time = time.time()

    # Initializations
    V  = V0.copy()
    Q = np.zeros((Ns,Ny,Na))
    V_old = V.copy()
    Q_old = Q.copy()
    Xi = np.zeros_like(P_full)
    Pi = np.zeros_like(Q)

    V_storage = [V]
    Q_storage = [Q]
    Xi_storage = [Xi]
    pointwise_error = np.zeros(((max_iters,)+V.shape))
    V_converged = np.zeros_like(V)
    multi_starts = np.ones_like(V)*min_multi_starts

    if verbose:
        print('Value Function')
        print(np.round(V[0:(Ns-1),1],1))
        print(np.round(V[0:(Ns-1),12],1))
        print(np.round(V[0:(Ns-1),19],1))
        print('')

    # Iterations of Algo
    for i in range(max_iters):
        if verbose:
            print('VI iter'+str(i))

        # Loop over (x,y)
        for x in range(Ns):
            for yi in range(Ny):

                (v_est,xis,q_ests,pi,xps) = apply_pCVaR_max_operator(x, # location
                                            yi, # threshold
                                            V_old, # value function to use for next state value function
                                            P_full, # P matrix (x,x',y,a)
                                            gamma, # discount
                                            y_set, # set of interpolation points for y
                                            r,
                                            Na=Na)

                V[x,yi] = v_est # update value funciton
                Q[x,yi,:] = q_ests
                Xi[x,xps,yi,:] = xis.T # store weights
                Pi[x,yi,:] = pi

        if verbose:
            print(np.round(V[0:(Ns-1),1],1))
            print(np.round(V[0:(Ns-1),12],1))
            print(np.round(V[0:(Ns-1),19],1))
            print('')

        # Convergence
        abs_err = np.abs(V_old - V)

        # Pointwise convergence to Determine Multi-Start Numbers for Each State
        pointwise_error[i,:,:]=abs_err

        if i>check_pointwise_iters:
            #import pdb; pdb.set_trace()
            for x in range(Ns):
                for yi in range(Ny):
                    if np.all(pointwise_error[(i-pointwise_window):(i),x,yi]<converg_thresh): # if previous
                        V_converged[x,yi]=1
            if verbose:
                print('pointwise error')
                print(pointwise_error[i,:,1])
                print(pointwise_error[i,:,12])
                print(pointwise_error[i,:,19])
                print('pointwise convergence')
                print(V_converged[:,1])
                print(V_converged[:,12])
                print(V_converged[:,19])

            multi_starts[V_converged==0]+=1 # augment the multi-start count for states that have not converged
            multi_starts[multi_starts>max_multi_starts]=max_multi_starts # don't let more than 10 multi-starts
            multi_starts[V_converged==1]=min_multi_starts # set the multi_starts count to 1 for states that have converged
            if verbose:
                print('multi-starts')
                print(multi_starts[:,1])
                print(multi_starts[:,12])
                print(multi_starts[:,19])

        # Max value difference
        max_abs_err = np.max(abs_err)
        where_max_abs_err = np.where(abs_err==max_abs_err)

        # Max value difference at prioritization threshold
        if yi_pr is not None:
            max_abs_err_pr = np.max(abs_err[:,yi_pr])
            where_max_abs_err_pr = np.where(abs_err==max_abs_err_pr)

        # Choose type of convergence
        converged = False
        if converg_type == 'full_eval':
            converged = max_abs_err<converg_thresh
        elif converg_type == 'full_window':
            converged = np.all(pointwise_error[(i-pointwise_window):(i),:,:]<converg_thresh)
        elif converg_type == 'pr_eval':
            converged = (max_abs_err_pr<converg_thresh_pr) & (max_abs_err<converg_thresh_nonpr)

        # Print Max Error
        if verbose:
            print('max error='+str(max_abs_err))
            print('at x='+str(where_max_abs_err[0][0])+' y='+str(where_max_abs_err[1][0]))
            if yi_pr is not None:
                print('max pr error='+str(max_abs_err_pr))
                print('at x='+str(where_max_abs_err_pr[0][0])+' y='+str(where_max_abs_err_pr[1][0]))

        # If converged
        if converged:
            print('Converged in iters='+str(i))
            print('max error='+str(max_abs_err))
            print('at x='+str(where_max_abs_err[0][0])+' y='+str(where_max_abs_err[1][0]))
            if yi_pr is not None:
                print('max pr error='+str(max_abs_err_pr))
                print('at x='+str(where_max_abs_err_pr[0][0])+' y='+str(where_max_abs_err_pr[1][0]))
            break
        else:
            # replace V_i
            V_old = V.copy()
            V_storage.append(V_old)

    print("--- %s seconds ---" % (time.time() - start_time))

    return(V,Q,Xi,Pi,V_storage,Q_storage,Xi_storage,pointwise_error,V_converged,multi_starts)
