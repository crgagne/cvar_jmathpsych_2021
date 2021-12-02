import numpy as np
from scipy.optimize import minimize
import time


def distorted_exp_exact(xis,V,xps,p,y,y_set):
    '''
    Function which tries to minimize value over next state's by choosing distortion weights.
    Difference to "distorted_exp" is that the current function does not need to be interpolated, because you never move in y-space.

    Inputs:
    xis = weights (usually 4)
    V = full value function (x,y); costs will be positive and rewards will be negative. So we want to maximize
        the distorted expectation.
    xps = next states (usually 4)
    p = next state probabilities (usually 4)
    y = current threshold
    y_set = set of y's interpolation points, passing to the interpolation function.

    '''
    if np.any(np.isnan(xis)):
        return(np.inf)

    Vp = np.empty(len(xis))
    yps = np.empty(len(xis))

    # loop over next states and weights
    for i,(xp,xi) in enumerate(zip(xps,xis)):
        yp = y # get next state y (difference from pCVaR, which is y*xi)
        yps[i]=yp # store
        yi = np.where(y_set==yp)[0] # get index for next state y
        Vp[i] = V[xp,yi] # get next state values # difference from pCVaR, which is interpolate_yV(V,xp,yp,y_set)

    distorted_exp = -1*np.sum(p*xis*Vp) # difference from pCVaR, which is V_interp/y

    # maximize distorted expectation which is positive for costs.
    return(distorted_exp)


def apply_nCVaR_operator(x,yi,V_old,P_ssy,gamma,y_set,r,max_inner_iters=100,multi_starts=10,same_answer_ns=3):
    '''For a single (x,y) state apply the nCVaR Bellman operator (Evaluation Operator)
        (1) find the current y value
        (2) set up constraints
        (3) find optimal weights
        (4) unpack solution
        (5) add in immediate reward

    '''
    assert same_answer_ns<=multi_starts

    # get current y value.
    y = y_set[yi]

    # get next states and probabilities
    xps = np.where(P_ssy[x,:,yi]!=0)[0] # find non-zero prob states
    p = P_ssy[x,xps,yi] # get probs

    # check if only one next state and that state value is exactly 0 (i.e. the absorbing state)
    # no weightings will matter
    if len(xps)==1:
        if V_old[xps,yi][0]==0:
            v_est = -1*r[x]
            success = True
            xis = np.array([1])
            return(v_est,xis,success,xps)

    # check if we are at y=0; choose mininum next state values.
    # weights will be 0 and 1/p for the worst transition
    if y == 0.0:
        maxV = np.max(V_old[xps,0])
        v_est = -1*r[x] + gamma*maxV # take the max over next state values; maximally distorted expectation
        xis = np.zeros(len(xps))
        maxV_idcs = np.where(V_old[xps,0]==maxV)[0] # find the indices
        xis[maxV_idcs]=1/(p[maxV_idcs]*len(maxV_idcs)) # take the weight as 1/prob, but if there are more than 1 additionally divide by total)
        assert (xis*p).sum()==1
        success = True
        return(v_est,xis,success,xps)

    # weight bounds
    bnds = tuple(((0.0,1.0/y) for i in range(len(p))))

    # sum to 1 constraint
    def sum_to_1_constraint(xi):
        zero = np.dot(xi,p)-1
        return(zero) # p is found in one python env up;

    cons = ({'type': 'eq', 'fun': sum_to_1_constraint})

    succeeded_at_least_once = False
    results_list = []
    fun_mins = []

    # Loop until successfully optimized
    for i in range(max_inner_iters):

        # weight initial values
        if len(y_set)>1:
            y_min = np.min(y_set[y_set!=0])
            xis_init = np.random.uniform(y_min,1/y,len(p))
        else:
            xis_init = np.random.uniform(0.001,1/y,len(p))

        # optimize inner objective (via sequential least squares programming)
        results = minimize(distorted_exp_exact, xis_init, args=(V_old,xps,p,y,y_set),method='SLSQP',
                           bounds=bnds,
                           constraints=cons)

        # whether scipy thinks it succeeded
        if results.success:
            succeeded_at_least_once=True
            results_list.append(results)
            fun_mins.append(results.fun)

        # exit early if all N minimums are the same; N specified.
        if len(fun_mins)>same_answer_ns:
            if np.all(np.array(fun_mins)==np.array(fun_mins)):
                break

        # exit after number of multi-starts have been exceeded.
        if len(fun_mins)>multi_starts:
            break

    # find minimum over multi-starts
    argmin_funs = np.argmin(np.array(fun_mins))
    results = results_list[argmin_funs]

    # unpack results
    xis = results.x
    vp_est = -1*results.fun
    success = results.success
    if success==False:
        print('Failed x='+str(x)+' yi='+str(yi))

    # Add in immediate reward (needs to be negative so costs are positive), and discount
    v_est = -1*r[x] + gamma*vp_est

    return(v_est,xis,success,xps)


def apply_nCVaR_max_operator(x, # location
                            yi, # threshold
                            V_old, # value function to use for next state value function
                            P_full, # P matrix (x,x',y,a)
                            gamma, # discount
                            y_set, # set of interpolation points for y
                            r, # reward function (x)
                            invtmp=10,
                            Na=4, # number of actions
                            max_inner_iters=100,multi_starts=10,same_answer_ns=3,
                            roundoff=10, # minimum resolution for minimizing Q
                            ):
    '''For a single (x,y) state apply the nCVaR Optimality Bellman operator.
    This function is basically same as one above but loops over actions and applies max.

    '''
    assert same_answer_ns<=multi_starts

    # get current y value.
    y = y_set[yi]

    q_ests = []
    xis_list = []
    for a in range(Na):

        # get next states and probabilities
        xps = np.where(P_full[x,:,yi,a]!=0)[0] # find non-zero prob states
        p = P_full[x,xps,yi,a] # get probs

        # check if we are at y=0; choose mininum next state values. (i.e. infinite weights)
        if y == 0.0:
            q_est = -1*r[x] + gamma*np.max(V_old[xps,0]) # take the max over next state values; maximally distorted expectation
            xis = np.ones(len(xps))*np.nan # don't both with returning weights here
            success = True
        else:

            # weight bounds
            bnds = tuple(((0.0,1.0/y) for i in range(len(p)))) # restrict it to not allow 0 weights; use minimum interpolation

            # sum to 1 constraint
            def sum_to_1_constraint(xi):
                zero = np.dot(xi,p)-1
                return(zero) # probs is found in one python env up;

            cons = ({'type': 'eq', 'fun': sum_to_1_constraint})

            succeeded_at_least_once = False
            results_list = []
            fun_mins = []

            # max iters, increase scipy doesn't think it ever succeeds.
            for i in range(max_inner_iters):

                # weight initial values
                if len(y_set)>1:
                    y_min = np.min(y_set[y_set!=0])
                    xis_init = np.random.uniform(y_min,1/y,len(p))
                else:
                    xis_init = np.random.uniform(0.001,1/y,len(p))

                # optimize inner objective (via sequential least squares programming)
                results = minimize(distorted_exp_exact, xis_init, args=(V_old,xps,p,y,y_set),
                                   bounds=bnds,
                                   constraints=cons)

                # whether scipy thinks it succeeded
                if results.success:
                    succeeded_at_least_once=True
                    results_list.append(results)
                    fun_mins.append(results.fun)

                # exit early if all N minimums are the same; N specified.
                if len(fun_mins)>same_answer_ns:
                    if np.all(np.array(fun_mins)==np.array(fun_mins)):
                        break

                # exit after number of multi-starts have been exceeded.
                if len(fun_mins)>multi_starts:
                    break

            # find minimum over multi-starts
            argmin_funs = np.argmin(np.array(fun_mins))
            results = results_list[argmin_funs]

            # unpack results
            xis = results.x
            vp_est = -1*results.fun
            success = results.success
            if success==False:
                print('Failed x='+str(x)+' yi='+str(yi))
                import pdb; pdb.set_trace()

            # Add in immediate reward (needs to be negative so costs are positive), and discount
            q_est = -1*r[x] + gamma*vp_est
        q_ests.append(q_est)
        xis_list.append(xis)

    q_ests = np.array(q_ests)
    xis_list = np.array(xis_list); # (a,s') #organized

    # Now choose max over Q values (min because everything is negative)
    v_est = np.min(q_ests)

    # find best actions
    best_actions = np.where(np.equal(q_ests,v_est))[0]
    if len(best_actions)==1:
        pi=np.zeros(Na)
        pi[best_actions]=1;
    else:
        # choose policy using soft-max if we have ties
        pi = np.exp(-1*invtmp*q_ests) / np.sum(np.exp(-1*invtmp*q_ests));

    return(v_est,xis_list,q_ests,pi,xps)
