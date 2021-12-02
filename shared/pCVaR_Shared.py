import numpy as np
from scipy.optimize import minimize
import time


def interpolate_yV(V,xp,yp,y_set):
    '''interpolate the value function times the risk preference:  y*V(x,y)'''

    yp_i_nearest = np.argmin(np.abs(y_set-yp))
    yp_nearest = y_set[yp_i_nearest]

    assert y_set[0]==0.0

    if yp>1:
        # this is possible
        return(V[xp,len(y_set)-1])
    elif yp<0:
        # should never be less than 0, so break if it does happen
        import pdb; pdb.set_trace()
    elif yp==yp_nearest:
        # no need for interpolation.
        return(yp*V[xp,yp_i_nearest])
    else:
        # find lower and upper y_nearest.
        if yp_nearest<yp:
            yp_i_upper = yp_i_nearest+1
            yp_upper = y_set[yp_i_upper]
            yp_i_lower = yp_i_nearest
            yp_lower = yp_nearest
        elif yp_nearest>yp:
            yp_i_upper = yp_i_nearest
            yp_upper = yp_nearest
            yp_i_lower = yp_i_nearest-1
            yp_lower = y_set[yp_i_lower]

        # Slope
        slope = (yp_upper*V[xp,yp_i_upper] - yp_lower*V[xp,yp_i_lower]) / (yp_upper - yp_lower)

        # Start at lower and difference times the slope
        V_interp = yp_lower*V[xp,yp_i_lower] + slope*(yp-yp_lower)

        return(V_interp)


def distorted_exp(xis,V,xps,p,y,y_set,verbose=False):
    '''
    Function which minimizes next states' summed values by choosing distortion weights.

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

    # get interpolated value for each next state
    V_interp = np.empty(len(xis))
    yps = np.empty(len(xis))

    # loop over next states and weights
    for i,(xp,xi) in enumerate(zip(xps,xis)):
        yp = y*xi # get next state y
        yps[i]=yp
        V_interp[i] = interpolate_yV(V,xp,yp,y_set) # interpolate

    distorted_exp = -1*np.sum(p*V_interp/y) # from Chow's interpoled Bellman equation

    if verbose:
        print('next states='+str(np.round(xps,3)))
        print('weights='+str(np.round(xis,3)))
        print('probs='+str(np.round(p,3)))
        print('weights*probs='+str(np.round(xis*p,3)))
        print('adjusted alpha='+str(np.round(yps,3)))
        print('interpolated value (undiscounted)='+str(np.round(-1*V_interp/y,3)))
        print('interpolated value x prob (undiscounted)='+str(np.round(p*-1*V_interp/y,3)))

    # maximize distorted expectation which is positive for costs.
    return(distorted_exp)


def apply_pCVaR_operator(x,yi,V_old,P_ssy,gamma,y_set,r,max_inner_iters=100,multi_starts=10,same_answer_ns=3):
    '''For a single (x,y) state apply the CVaR Bellman operator (Evaluation Operator)
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
    if yi == 0:
        maxV = np.max(V_old[xps,0])
        v_est = -1*r[x] + gamma*maxV # take the max over next state values; maximally distorted expectation
        xis = np.zeros(len(xps))
        maxV_idcs = np.where(V_old[xps,0]==maxV)[0] # find the indices
        xis[maxV_idcs]=1/(p[maxV_idcs]*len(maxV_idcs)) # take the weight as 1/prob, but if there are more than 1 additionally divide by total
        assert (xis*p).sum()==1
        success = True
        return(v_est,xis,success,xps)

    # weight bounds
    bnds = tuple(((0.0,1.0/y) for i in range(len(p))))

    # sum to 1 constraint
    def sum_to_1_constraint(xi):
        zero = np.dot(xi,p)-1
        return(zero) # p is found in one python env up

    cons = ({'type': 'eq', 'fun': sum_to_1_constraint})

    succeeded_at_least_once = False
    results_list = []
    fun_mins = []

    # Loop until successfully optimized
    for i in range(max_inner_iters):

        # weight initial values
        xis_init = np.random.uniform(y_set[1],1/y,len(p))  # this seems to be better than between 0 and 1/y

        # optimize inner objective (via sequential least squares programming)
        results = minimize(distorted_exp, xis_init, args=(V_old,xps,p,y,y_set),method='SLSQP',
                           bounds=bnds,
                           constraints=cons)

        # figure out whether scipy thinks it succeeded
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

def apply_pCVaR_max_operator(x, # location
                            yi, # threshold
                            V_old, # value function to use for next state value function
                            P_full, # P matrix (x,x',y,a)
                            gamma, # discount
                            y_set, # set of interpolation points for y
                            r, # reward function (x)
                            invtmp=10,
                            Na=4, # number of actions
                            max_inner_iters=100,multi_starts=10,same_answer_ns=3):
    '''For a single (x,y) state apply the CVaR Optimality Bellman operator.
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

        # check if we are at y=0; choose mininum next state values.
        if yi == 0:
            q_est = -1*r[x] + gamma*np.max(V_old[xps,0]) # take the max over next state values; maximally distorted expectation
            xis = np.ones(len(xps))*np.nan # dont both storing weights
            success = True

        else:

            # weight bounds
            bnds = tuple(((0.0,1.0/y) for i in range(len(p))))

            # sum to 1 constraint
            def sum_to_1_constraint(xi):
                zero = np.dot(xi,p)-1
                return(zero) # p is found in one python env up

            cons = ({'type': 'eq', 'fun': sum_to_1_constraint})

            succeeded_at_least_once = False
            results_list = []
            fun_mins = []

            # max iters, increase scipy doesn't think it ever succeeds.
            for i in range(max_inner_iters):

                # weight initial values
                xis_init = np.random.uniform(y_set[1],1/y,len(p))

                # optimize inner objective (via sequential least squares programming)
                results = minimize(distorted_exp, xis_init, args=(V_old,xps,p,y,y_set),
                                   bounds=bnds,
                                   constraints=cons)

                # figure out whether scipy thinks it succeeded
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
    best_actions = np.where(q_ests==v_est)[0]
    if len(best_actions)==1:
        pi=np.zeros(Na)
        pi[best_actions]=1;
    else:
        # choose policy using soft-max if we have ties
        pi = np.exp(-1*invtmp*q_ests) / np.sum(np.exp(-1*invtmp*q_ests));

    return(v_est,xis_list,q_ests,pi,xps)


def sample_traj_from_xy_policy(Pol, # Pol (x,y,a)
                               P,   # P (x,x',a)
                               Xi,  # Xi (x,x',y), or Xi(x,xi,y,a) but need to set use_Xi_w_a to true
                               r_fun, # r (x)
                               gamma,
                               num_sims = 10000,
                               T = 15,
                               x0 = 12,
                               y_ind0 = 11,
                               y_set = None,
                               move_y = True,
                               use_Xi_w_a = False,
                               disc_first_step=False):


    y0 = y_set[y_ind0]

    y_sims = []
    y_ind_sims = []
    x_sims = []
    a_sims = []
    xis_sims = []
    r_sims = []
    R_sims = []

    for sim in range(num_sims):

        x = x0
        y_ind = y_ind0
        y = y0

        # storage
        ys = [y]
        y_inds = [y_ind]
        xs = [x]
        actions = []
        xis = []
        rs = []

        # loop through timesteps;
        for t in range(T):

            # get reward
            r = r_fun[x]
            rs.append(r)

            # select an action according to optimal policy at current threshold.
            pi = Pol[x,y_ind,:]
            a = np.random.choice(np.arange(len(pi)),p=pi)
            actions.append(a)

            # observe next state according to model (use Python code)
            p = P[x,:,a].flatten()
            xp = np.random.choice(np.arange(len(p)),p=p)
            xs.append(xp)

            # find the distortion weight used for the transition.
            if use_Xi_w_a:
                xi_curr = Xi[x,xp,y_ind,a]
            else:
                xi_curr = Xi[x,xp,y_ind]
            xis.append(xi_curr)

            # calculate new threshold by multiplying by Xi, and find new closest index
            if move_y==True:
                yp = y*xi_curr
                yp_ind_closest = np.argmin(np.abs(y_set-yp))
                yp_closest = y_set[yp_ind_closest]
            elif move_y==False:
                yp_closest = y
                yp_ind_closest = y_ind

            ys.append(yp_closest)
            y_inds.append(yp_ind_closest)

            # prepare for next loop
            x = xp
            y = yp_closest
            y_ind = yp_ind_closest

        # calculate discounted return
        R = np.sum((gamma**np.arange(T))*np.array(rs)) # start state as t=0
        if disc_first_step:
            R = np.sum((gamma**np.arange(1,T+1))*np.array(rs))

        y_sims.append(ys)
        y_ind_sims.append(y_inds)
        x_sims.append(xs)
        a_sims.append(actions)
        xis_sims.append(xis)
        r_sims.append(rs)
        R_sims.append(R)

    y_sims = np.array(y_sims)
    y_ind_sims = np.array(y_ind_sims)
    x_sims = np.array(x_sims)
    a_sims = np.array(a_sims)
    xis_sims = np.array(xis_sims)
    r_sims = np.array(r_sims)
    R_sims = np.array(R_sims)
    if disc_first_step:
        print('note: discounting from first step')

    return(y_sims,y_ind_sims,x_sims,a_sims,xis_sims,r_sims,R_sims)
