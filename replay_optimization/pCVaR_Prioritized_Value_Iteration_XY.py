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
from pCVaR_Shared import distorted_exp, apply_pCVaR_max_operator, apply_pCVaR_operator
from pCVaR_Policy_Eval import pCVaR_PE


def hypothetical_update_and_evaluate_XY(x,yi,V_prev,V_CVaR_old,Pol_hyps,out):

    ''''''

    # Hypothetically update single (x,y) pair
    (v_est,_,q_ests,pi,_) = apply_pCVaR_max_operator(x, # location
                                yi, # threshold
                                V_prev, # value function to use for next state value function
                                out['P_full'], # P matrix (x,x',y,a)
                                out['gamma'], # discount
                                out['y_set'], # set of interpolation points for y
                                out['r'],
                                Na = out['Na'])


    # hypothetically update policy
    max_pol_diff  = np.max(np.abs(pi-Pol_hyps[x,yi,x,yi,:]))
    Pol_hyps[x,yi,x,yi,:]= pi

    # get new induced probability matrix (this will differ across Y)
    P_pi_new = induce_P(out['P'],Pol_hyps[x,yi,:,:,:],out['Ny'],out['Ns'],None,pol_type='stochastic',Na=out['Na'])
    Xi_new = np.zeros_like(P_pi_new)*np.nan

    # check if the hypothetical update doesn't change q-values from previous estimates? (or just starting Q-values)
    max_absdiff_between_q_v = np.amax(np.abs(np.array(v_est)[np.newaxis] - q_ests))
    if max_absdiff_between_q_v<out['q_v_diff_eval_thresh']:
        print('skipping evaluating policy after hypothetical update  xy=('+str(x)+','+str(yi)+') because V and Q dont differ')
        V_CVaR_new = V_CVaR_old

    # check if hypothetical update doesn't change values
    elif np.max(np.abs(V_prev[x,yi]-v_est))<out['q_v_diff_eval_thresh']:
        print('skipping evaluating policy after hypothetical update xy=('+str(x)+','+str(yi)+') because values did not change')
        V_CVaR_new = V_CVaR_old

    # check if hypothetical update doesn't change policies
    elif max_pol_diff==0:
        print('skipping evaluating policy after hypothetical update  xy=('+str(x)+','+str(yi)+')because value changed but policy did not')
        V_CVaR_new = V_CVaR_old

    # if it does, evaluate policy
    else:
        print('evaluating policy after hypothetical update xy=('+str(x)+','+str(yi)+')')
        (V_CVaR_new,Xi_new,_,_,_,_) = pCVaR_PE(V_CVaR_old, # initialize at true CVaR for old policy
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
                                            converg_thresh = 0.1,
                                            converg_thresh_pr= out['converg_thresh_pr'], # convergence threshold for y of interest
                                            converg_thresh_nonpr= out['converg_thresh_nonpr'], # convergence threshold  for other ys
                                            yi_pr=out['yi_pr'], # evaluation with convergence based on prioritized threshold
                                            converg_type = 'pr_eval')
        print('DONE evaluating policy after hypothetical update xy=('+str(x)+','+str(yi)+')')

    return(V_CVaR_new,P_pi_new,pi,v_est,q_ests,Xi_new)


def pCVaR_Prioritized_Value_Iteration_XY(
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
                                    converg_thresh_pr=0.01,
                                    converg_thresh_nonpr=0.1,
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
    out['testcase'] = testcase
    out['yi_pr']=yi_pr # this should really by called y_eval
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

    # Get some useful quantities about the MDP
    out['Ny'] = Ny = mat['Y_set_all'].shape[1]
    out['Ns'] = Ns = mat['Y_set_all'].shape[0]
    if 'Na' in mat.keys():
        out['Na'] = Na = mat['Na'][0][0]
        print('n actions='+str(out['Na']))
    else:
        if example_name=='1D':
            out['Na'] = Na = 3
        else:
            out['Na'] = Na = 4
    print('n ys='+str(out['Ny']))
    print('n xs='+str(out['Ns']))
    out['gamma'] = gamma = mat['dis'][0][0]
    out['y_set'] = y_set = mat['Y_set_all'][0,:]
    out['P'] = P = mat['P'] # "(x,x',a)"
    out['r'] = r = mat['r'][:,0] # "(x,y)"
    out['P_full'] = P_full = np.repeat(P[:,:,np.newaxis,:],Ny,2)

    #
    out['y_pr'] = y_pr = y_set[yi_pr]

    # save name
    out['fname'] = fname = example_name+'_prioritized_VI_pCVaR_ypr'+str(round(y_pr,3))+'_iters'+str(end_iter)+'.mat'
    print(fname)

    # Set-up storage (actual estimates)
    out['V_storage'] = np.zeros((end_iter,Ns,Ny))
    out['Q_storage'] = np.zeros((end_iter,Ns,Ny,Na))
    out['Pol_storage'] = np.zeros((end_iter,Ns,Ny,Na))
    out['pe_iters_storage'] = np.zeros((end_iter,Ns))

    # Set-up storage (estimated values for potential updates)
    out['V_est_new_storage'] = np.zeros((end_iter,Ns,Ny,Ns,Ny)) # dimenions 2,3 are hypothetical updates
    out['Q_est_new_storage'] = np.zeros((end_iter,Ns,Ny,Ns,Ny,Na))
    out['Pol_new_storage'] = np.zeros((end_iter,Ns,Ny,Ns,Ny,Na))

    # Set-up storage (true values for potential updates)
    out['V_true_new_storage'] = np.zeros((end_iter,Ns,Ny,Ns,Ny))
    out['V_true_old_storage'] = np.zeros((end_iter,Ns,Ny))
    out['P_pi_new_storage'] = np.zeros((end_iter,Ns,Ny,Ns,Ns,Ny))
    out['P_pi_old_storage'] = np.zeros((end_iter,Ns,Ns,Ny))
    out['Xi_new_storage'] = np.zeros((end_iter,Ns,Ny,Ns,Ns,Ny))
    out['Xi_old_storage'] = np.zeros((end_iter,Ns,Ns,Ny))

    # Set-up storage for value function differences
    out['V_diff_true'] = np.zeros((end_iter,Ns,Ny,Ny)) # at start state
    out['replay_storage'] = []

    if special_init is None:

        # load normal value iteration values for initialization
        fname = '../simulation_results/'+example_name+'_random_policy_pCVaR.npz'
        tmp = loadmat( '../simulation_results/'+example_name+'_random_policy.mat')
        V_0 = tmp['V_Exp']

        # initialize value function to be 1/10 of optimal value function for y=1;
        # ensures convexity of it.
        V = np.repeat(V_0/1000,Ny,1)*0

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


    if os.path.isfile(fname):

        # Load
        container = np.load(fname)
        data = [container[key] for key in container]
        V_CVaR_old = data[0]
        Xi_old = data[1]

    else:

        # CVaR Policy Evaluation to get the true new value function (for first iteration)
        (V_CVaR_old,Xi_old,V_storage_tmp,_,_,_) = pCVaR_PE(V, # intialization for value function Copy
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
                                   converg_thresh = 0.01,
                                   yi_pr=None,
                                   converg_type = 'full_eval')
        np.savez(fname,*[V_CVaR_old,Xi_old,V_storage_tmp])

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
        out['P_pi_old_storage'][i,:,:] = P_pi_old
        out['Xi_old_storage'][i,:,:] = Xi_old

        # set up storage just for this loop (estimates)
        V_hyps = np.repeat(np.repeat(V_prev[np.newaxis,:,:],Ny,0)[np.newaxis,:,:,:],Ns,0); # % first 2 dims are for possible updates
        Q_hyps = np.repeat(np.repeat(Q_prev[np.newaxis,:,:,:],Ny,0)[np.newaxis,:,:,:,:],Ns,0)
        Pol_hyps = np.repeat(np.repeat(Pol_prev[np.newaxis,:,:,:],Ny,0)[np.newaxis,:,:,:,:],Ns,0)

        # (true new value)
        V_CVaR_news = np.zeros((Ns,Ny,Ns,Ny)); #

        xypairs = []
        for x in range(Ns):
            for yi in range(Ny):
                xypairs.append((x,yi))

        # Inner Loop For Choosing Prioritization
        if parallel:
            args = []
            with Pool(multiprocessing.cpu_count()) as p:
                #TODO: Change to x,y
                map_result = p.starmap(hypothetical_update_and_evaluate_XY, [(x,yi,V_prev,V_CVaR_old,Pol_hyps,out) for (x,yi) in xypairs])

        else:
            # for debugging
            #V_CVaR_new,P_pi_new,pi,v_est,q_ests=hypothetical_update_and_evaluate_XY(xypairs[0][0],xypairs[0][1],V_prev,V_CVaR_old,Pol_hyps,out)

            map_result = starmap(hypothetical_update_and_evaluate_XY, [(x,yi,V_prev,V_CVaR_old,Pol_hyps,out) for (x,yi) in xypairs])
            map_result = list(map_result)

        # Loop over (x,y) pairs and unpack results of hypothetical backup
        for j,(x,yi) in enumerate(xypairs):

            # update specific position for updated state
            V_CVaR_news[x,yi,:,:] = map_result[j][0]
            Pol_hyps[x,yi,x,yi,:]=  map_result[j][2]
            V_hyps[x,yi,x,yi] =  map_result[j][3]
            Q_hyps[x,yi,x,yi,:] =  map_result[j][4]

            # store whole arrays for each updated state
            out['P_pi_new_storage'][i,x,yi,:,:,:] = map_result[j][1]
            out['Xi_new_storage'][i,x,yi,:,:,:] = map_result[j][5]
            out['V_est_new_storage'][i,x,yi,:,:] = V_hyps[x,yi,:,:];
            out['Q_est_new_storage'][i,x,yi,:,:,:] = Q_hyps[x,yi,:,:,:];
            out['Pol_new_storage'][i,x,yi,:,:,:] = Pol_hyps[x,yi,:,:,:];

            # Value differences at location x_eval used for prioritization
            # Loops through possible y's to use for evaluation even though only one will be used, i.e. yi_pr
            for yi2 in range(Ny):
                out['V_diff_true'][i,x,yi,yi2] = V_CVaR_news[x,yi,x_eval,yi2] - V_CVaR_old[x_eval,yi2];

        # choose best (x,y) to back-up
        min_change = np.inf
        j = 0
        for x in range(Ns):
            for yi in range(Ny):
                change = out['V_diff_true'][i,x,yi,yi_pr] # here's where we specify the priorizied y
                if change<min_change:
                    min_change=change
                    min_change_xypair=(x,yi)
                    min_change_ind=j
                j+=1

        print('CVaR changes')
        print(np.round(out['V_diff_true'][i,:,::-1,yi_pr].T,2))
        print('backing up state xy='+str(min_change_xypair))
        print('reducing CVaR by '+str(min_change))

        # get the current estimates for the next iteration
        V = np.squeeze(V_hyps[min_change_xypair[0],min_change_xypair[1],:,:])
        Q = np.squeeze(Q_hyps[min_change_xypair[0],min_change_xypair[1],:,:,:])
        Pol = np.squeeze(Pol_hyps[min_change_xypair[0],min_change_xypair[1],:,:,:])

        # store them
        out['V_storage'][i,:,:] = V[:,:]
        out['Q_storage'][i,:,:,:] = Q[:,:,:]
        out['Pol_storage'][i,:,:,:] = Pol[:,:,:]

        # reset the current true values
        V_CVaR_old = np.squeeze(V_CVaR_news[min_change_xypair[0],min_change_xypair[1],:,:])

        Xi_old = out['Xi_new_storage'][i,min_change_xypair[0],min_change_xypair[1],:,:,:].copy()

        # store stuff from deliberation
        out['V_true_new_storage'][i,:,:,:] = V_CVaR_news[:,:,:]
        out['replay_storage'].append(min_change_xypair)

        # store iteration
        out['i'] = i

        # save information with name
        savename_prev = '../simulation_results/'+example_name+'_prioritized_VI_pCVaR_ypr'+str(yi_pr)+'_xeval'+str(x_eval)+'_iters'+str(i-1)+extra_name
        savename = '../simulation_results/'+example_name+'_prioritized_VI_pCVaR_ypr'+str(yi_pr)+'_xeval'+str(x_eval)+'_iters'+str(i)+extra_name
        out['savefolder']=savename # where to save

        # remove previous version
        if os.path.isfile(savename_prev+'.pkl'):
            os.remove(savename_prev+'.pkl')

        # save new version
        with open(savename+'.pkl','wb') as f:
            pickle.dump(out,f)
