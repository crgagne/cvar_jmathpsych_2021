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
sys.path.append('../policy_evaluation')

from shared import induce_P
from nCVaR_Shared import distorted_exp_exact, apply_nCVaR_max_operator
from nCVaR_Policy_Eval import nCVaR_PE


def hypothetical_update_and_evaluate_dynamic(x,V_prev,V_CVaR_old,Pol_hyps,out):

    ''''''

    # Set up a few local lists to return
    v_ests = []
    q_ests_s = []
    pis = []

    # Hypothetically update single state;
    for yi in range(out['Ny']):
        (v_est,_,q_ests,pi,_) = apply_nCVaR_max_operator(x, # location
                                    yi, # threshold
                                    V_prev, # value function to use for next state value function
                                    out['P_full'], # P matrix (x,x',y,a)
                                    out['gamma'], # discount
                                    out['y_set'], # set of interpolation points for y
                                    out['r'],
                                    Na = out['Na'])
        v_ests.append(v_est)
        q_ests_s.append(q_ests)
        pis.append(pi) # the best pi for each threshold is figured out.

    # convert to arrays
    v_ests = np.array(v_ests)
    q_ests_s = np.array(q_ests_s)
    pis = np.array(pis)

    # hypothetically update policy
    # (use the full policy across other states too)
    max_pol_diff  = np.max(np.abs(pis[0,:]-Pol_hyps[x,x,0,:])) # this changes to 0 from 1:
    Pol_hyps[x,x,:,:]= pis

    # get new induced probability matrix
    P_pi_new = induce_P(out['P'],Pol_hyps[x,:,:,:],out['Ny'],out['Ns'],None,pol_type='stochastic',Na=out['Na'])
    Xi_new = np.zeros_like(P_pi_new)*np.nan

    # check if the hypothetical update has any effect on Q_starting q-values
    max_absdiff_between_q_v = np.amax(np.abs(v_ests[:,np.newaxis] - q_ests_s),axis=1) # per threshold
    if np.all(max_absdiff_between_q_v<out['q_v_diff_eval_thresh']):
        print('skipping evaluating policy after hypothetical update x='+str(x))
        V_CVaR_new = V_CVaR_old

    # skip evaluation for various reasons to save time
    elif np.max(np.abs(V_prev[x,:]-v_ests))<out['q_v_diff_eval_thresh']:
        print('skipping evaluating policy after hypothetical update x='+str(x)+'because values did not change')
        V_CVaR_new = V_CVaR_old
    elif max_pol_diff==0:
        print('skipping evaluating policy after hypothetical update x='+str(x)+'because value changed but policy did not')
        V_CVaR_new = V_CVaR_old
    else:
        print('evaluating policy after hypothetical update x='+str(x))
        (V_CVaR_new,Xi_new,_,_,_,_) = nCVaR_PE(V_CVaR_old, # initialize at true CVaR for old policy
                                            out['Ns'],
                                            out['Ny'],
                                            P_pi_new, #  new hypothetical transition matrix
                                            out['gamma'],
                                            out['maze'],
                                            out['y_set'],
                                            out['r'],
                                            check_pointwise_iters=6, # how many iters to start increasing multi-start
                                            pointwise_window=3, # how many iters back to assess convergence for multi-start.
                                            min_multi_starts=out['min_multi_starts'], # minimum number of times to run minimization for single operator
                                            max_multi_starts=out['max_multi_starts'], # max ""
                                            max_iters=30, # max iters for evaluation algorithm
                                            verbose=out['verbose'],
                                            converg_thresh = 0.01,
                                            converg_thresh_pr= out['converg_thresh_pr'], # convergence threshold for y of interest
                                            converg_thresh_nonpr= out['converg_thresh_nonpr'], # convergence threshold  for other ys
                                            yi_pr=out['yi_pr'], # evaluation with convergence based on prioritized threshold
                                            converg_type = 'pr_eval')
        print('DONE evaluating policy after hypothetical update x='+str(x))



    return(V_CVaR_new,P_pi_new,pis,v_ests,q_ests_s,Xi_new)


def nCVaR_Prioritized_Value_Iteration(
                                    example_name,
                                    testcase,
                                    maze,
                                    yi_pr,
                                    x_eval,
                                    start_iter = 0,
                                    end_iter = 3,
                                    maxIter_inner = 100,
                                    min_multi_starts = 2,
                                    max_multi_starts = 5,
                                    converg_thresh_pr=0.001,
                                    converg_thresh_nonpr=0.001,
                                    init_me = True,
                                    parallel=True,
                                    extra_name='',
                                    verbose = False,
                                    q_v_diff_eval_thresh=0.01,
                                    subcase='',
                                    set_values_as_rewards=True, # setting to false does not work at the moment, no first update is performed
                                    special_init=None):

    out = {}

    # store some parameters
    out['example_name'] = example_name
    out['testcase'] = testcase
    out['yi_pr']=yi_pr
    out['x_eval']=x_eval
    out['maze']=maze
    out['start_iter'] = start_iter
    out['end_iter'] = end_iter
    out['converg_thresh_pr']=converg_thresh_pr
    out['converg_thresh_nonpr'] = converg_thresh_nonpr
    out['q_v_diff_eval_thresh'] = q_v_diff_eval_thresh
    out['min_multi_starts'] = min_multi_starts
    out['max_multi_starts'] = max_multi_starts
    out['verbose'] = verbose

    # Load CVaR Value Iteration
    mat = loadmat('../simulation_results/'+example_name+'_optimal_policy.mat')

    # Changing the y-set
    y_set_tmp = mat['Y_set_all'][0,:]

    out['y_pr'] = y_pr = y_set_tmp[yi_pr] # find the value for y_pr
    out['yi_pr_orig'] = yi_pr
    out['yi_pr'] = yi_pr = 0 # re-set index yi_pr
    y_set = [y_pr] # new y_set is just [y_pr]
    out['y_set'] = y_set

    # Get some useful quantities
    out['Ny'] = Ny = 1
    out['Ns'] = Ns = mat['Y_set_all'].shape[0]
    if 'Na' in mat.keys():
        out['Na'] = Na = mat['Na'][0][0]
        print('actions='+str(out['Na']))
    else:
        if example_name=='1D':
            out['Na'] = Na = 3
        else:
            out['Na'] = Na = 4

    out['gamma'] = gamma = mat['dis'][0][0]
    out['P'] = P = mat['P'] # "(x,x',a)"
    out['r'] = r = mat['r'][:,0] # "(x,y)"
    out['P_full'] = P_full = np.repeat(P[:,:,np.newaxis,:],Ny,2)

    # save name
    out['fname'] = fname = example_name+'_prioritized_VI_nCVaR_ypr'+str(round(y_pr,3))+'_iters'+str(end_iter)+'.mat'
    print(fname)

    # Set-up storage (actual estimates)
    out['V_storage'] = np.zeros((end_iter,Ns,Ny))
    out['Q_storage'] = np.zeros((end_iter,Ns,Ny,Na))
    out['Pol_storage'] = np.zeros((end_iter,Ns,Ny,Na))
    out['pe_iters_storage'] = np.zeros((end_iter,Ns))

    # Set-up storage (estimated values for potential updates)
    out['V_est_new_storage'] = np.zeros((end_iter,Ns,Ns,Ny))
    out['Q_est_new_storage'] = np.zeros((end_iter,Ns,Ns,Ny,Na))
    out['Pol_new_storage'] = np.zeros((end_iter,Ns,Ns,Ny,Na))

    # Set-up storage (true values for potential updates)
    out['V_true_new_storage'] = np.zeros((end_iter,Ns,Ns,Ny))
    out['V_true_old_storage'] = np.zeros((end_iter,Ns,Ny))
    out['P_pi_new_storage'] = np.zeros((end_iter,Ns,Ns,Ns,Ny))
    out['P_pi_old_storage'] = np.zeros((end_iter,Ns,Ns,Ny))
    out['Xi_new_storage'] =  np.zeros((end_iter,Ns,Ns,Ns,Ny))
    out['Xi_old_storage'] = np.zeros((end_iter,Ns,Ns,Ny))

    # Set-up storage for value function differences
    out['V_diff_true'] = np.zeros((end_iter,Ns,Ny))
    out['replay_storage'] = np.zeros(end_iter)

    if special_init is None:

        # load normal value iteration values for initialization
        fname = '../simulation_results/'+example_name+'_random_policy_nCVaR.npz'
        container = np.load(fname)
        data = [container[key] for key in container]
        V_0 = data[0]

        # initialize value function to be 1/10 of optimal value function for y=1;
        # ensures convexity of it.
        V = np.repeat(V_0[:,[out['yi_pr_orig']]]/1000,Ny,1)*0

        # Add rewards in environment (otherwise the first prioritization step is random)
        if set_values_as_rewards:
            for x in range(Ns):
                if r[x]!=0:
                    V[x,:] = -1*r[x];

        # initialize Q value
        Q = np.repeat(V[:,:,np.newaxis],Na,2)

        # start with random policy
        Pol = np.ones((Ns,Ny,Na))*1/Na

        # get induced state transition matrix
        P_pi_old = induce_P(P,Pol,Ny,Ns,None,pol_type='stochastic',Na=Na)

        print('initial policy evaluation')

    # Load previous policy and value to start at
    else:
        pass

    # CVaR Policy Evaluation to get the true new value function (for first iteration)
    (V_CVaR_old,Xi_old,V_storage_tmp,_,_,_) = nCVaR_PE(V, # intialization for value function Copy
                               Ns,
                               Ny,
                               P_pi_old, # evaluate on old induced policy
                               gamma,maze,y_set,r,
                               check_pointwise_iters=6, # how many iters to start increasing multi-start
                               pointwise_window=3, # how many iters back to assess convergence for multi-start.
                               min_multi_starts=2, # minimum number of times to run minimization for single operator
                               max_multi_starts=5, # max ""
                               max_iters=100, # max iters for evaluation algorithm
                               verbose=out['verbose'],
                               converg_thresh = 0.001,
                               yi_pr=None,
                               converg_type = 'full_eval')

    #############
    # Main Loop #
    for i in range(start_iter,end_iter):
        print('main iter = '+str(i))

        # store the current true value of things
        out['V_true_old_storage'][i,:,:] = V_CVaR_old

        # rename current estimates as previous estimates
        V_prev = V;
        Q_prev = Q;
        Pol_prev = Pol;

        # get induced P from current policy
        P_pi_old = induce_P(P,Pol,Ny,Ns,y_pr,pol_type='stochastic',Na=Na)
        out['P_pi_old_storage'][i,:,:,:] = P_pi_old
        out['Xi_old_storage'][i,:,:,:] = Xi_old

        # set up storage just for this loop (estimates)
        V_hyps = np.repeat(V_prev[np.newaxis,:,:],Ns,0); # % first dim is possible updates
        Q_hyps = np.repeat(Q_prev[np.newaxis,:,:,:],Ns,0);
        Pol_hyps = np.repeat(Pol_prev[np.newaxis,:,:,:],Ns,0);

        # (true new value)
        V_CVaR_news = np.zeros((Ns,Ns,Ny)); #

        locs = range(Ns)

        # Inner Loop For Choosing Prioritization
        if parallel:
            args = []
            with Pool(multiprocessing.cpu_count()) as p:
                map_result = p.starmap(hypothetical_update_and_evaluate_dynamic, [(x,V_prev,V_CVaR_old,Pol_hyps,out) for x in locs])

        else:
            # for debugging
            #tmp2 = hypothetical_update_and_evaluate_dynamic(locs[0],V_prev,V_CVaR_old,Pol_hyps,out)
            map_result = starmap(hypothetical_update_and_evaluate_dynamic, [(x,V_prev,V_CVaR_old,Pol_hyps,out) for x in locs])
            map_result = list(map_result)

        # Unpack
        for x in locs:

            # update specific position for updated state
            V_CVaR_news[x,:,:] = map_result[x][0]
            Pol_hyps[x,x,:,:]=  map_result[x][2]
            V_hyps[x,x,:] =  map_result[x][3]
            Q_hyps[x,x,:,:] =  map_result[x][4]

            # store whole arrays for each updated state
            out['P_pi_new_storage'][i,x,:,:] = map_result[x][1]
            out['Xi_new_storage'][i,x,:,:,:] = map_result[x][5]
            out['V_est_new_storage'][i,x,:,:] = V_hyps[x,:,:];
            out['Q_est_new_storage'][i,x,:,:,:] = Q_hyps[x,:,:,:];
            out['Pol_new_storage'][i,x,:,:,:] = Pol_hyps[x,:,:,:];

            # True differences at prioritization state
            for yi in range(Ny):
                out['V_diff_true'][i,x,yi] = V_CVaR_news[x,x_eval,yi] - V_CVaR_old[x_eval,yi];

        # choose best state to back-up
        min_change = min(out['V_diff_true'][i,:,yi_pr]);
        min_ind = np.where(out['V_diff_true'][i,:,yi_pr]==min_change)[0];

        if len(min_ind)>1:
            print('tie for priority')
            min_ind = np.max(min_ind)

        print('CVaR changes')
        print(np.round(out['V_diff_true'][i,:,yi_pr],2))
        print('backing up state x='+str(min_ind))
        print('reducing CVaR by'+str(min_change))

        # get the current estimates for the next iteration
        V = np.squeeze(V_hyps[min_ind,:,:])
        Q = np.squeeze(Q_hyps[min_ind,:,:,:])
        Pol = np.squeeze(Pol_hyps[min_ind,:,:,:])

        Xi_old = np.squeeze(out['Xi_new_storage'][i,min_ind,:,:,:].copy())

        # store them
        if out['V_storage'][i,:,:].shape != V.shape: # for 1-D the extra dimension got squeezed out
            V = V[:,np.newaxis]
            Q = Q[:,np.newaxis,:]
            Pol = Pol[:,np.newaxis,:]
            Xi_old = Xi_old[:,:,np.newaxis]

        out['V_storage'][i,:,:] = V
        out['Q_storage'][i,:,:,:] = Q
        out['Pol_storage'][i,:,:,:] = Pol

        # reset the current true values
        V_CVaR_old = np.squeeze(V_CVaR_news[min_ind,:,:])
        if out['V_true_old_storage'][i,:,:].shape != V_CVaR_old.shape: # for 1-D the extra dimension got squeezed out
            V_CVaR_old = V_CVaR_old[:,np.newaxis]

        # store stuff from deliberation
        out['V_true_new_storage'][i,:,:,:] = V_CVaR_news[:,:,:]
        out['replay_storage'][i] = min_ind

        # store iteration
        out['i'] = i

        # save information with name
        savename_prev = '../simulation_results/'+example_name+'_prioritized_VI_nCVaR_ypr'+str(out['yi_pr_orig'])+'_xeval'+str(x_eval)+'_iters'+str(i-1)+extra_name
        savename = '../simulation_results/'+example_name+'_prioritized_VI_nCVaR_ypr'+str(out['yi_pr_orig'])+'_xeval'+str(x_eval)+'_iters'+str(i)+extra_name
        out['savefolder']=savename # where to save

        # remove previous version
        if os.path.isfile(savename_prev+'.pkl'):
            os.remove(savename_prev+'.pkl')

        # save new version
        with open(savename+'.pkl','wb') as f:
            pickle.dump(out,f)
