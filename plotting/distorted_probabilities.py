import numpy as np

def create_distorted_probability_matrix(P_ssy, #
                                        Xi, # same size as P_ssy
                                        Ns,Ny,Na,
                                        y_set,
                                        yis = [5,4,3,2,1,0], # which y values do you want for plotting
                                        adjust_y=True,
                                        ):
    '''Expands the transition kernel so that it includes next risk preference levels y in addition to physical states x.
       i.e. P(x,x',y) --> P(x,y,x',y')

    '''

    y_set_tmp = [y_set[yi] for yi in yis]

    # set up empty matrices for undistorted and distorted transition probabilites.
    # note: the y entries for 0 will be the first y value given above; usually in reverse order for plotting
    P_xy_xpyp = np.zeros((Ns,len(y_set_tmp),Ns,len(y_set_tmp))) # non-distorted p(x',y'|x,y)
    P_xy_xpyp_alpha = np.zeros((Ns,len(y_set_tmp),Ns,len(y_set_tmp))) # distorted p(x',y'|x,y)*Xi(x'|x,y)
    Xi_xy_xpyp= np.zeros((Ns,len(y_set_tmp),Ns,len(y_set_tmp))) # Xi(x'|x,y)

    for x in range(Ns):
        for yi,yi_orig in enumerate(yis):
            xps = np.where(P_ssy[x,:,yi_orig]!=0)[0] # get next states
            ps = P_ssy[x,xps,yi_orig] # get probabilities
            xis = Xi[x,xps,yi_orig] # get distortion weights
            ps_xis = ps*xis  # distort the probabilties
            y = y_set[yi_orig] # get the actual value for y, not the indicator
            if adjust_y:
                yps = y*xis # multiply y by the distortion weights to get y'
            else:
                yps = y*np.ones_like(xis)
            yips_closest = [np.argmin(np.abs(y_set_tmp-yp)) for yp in yps] # find the closest indicator for y'

            for pxi,xi,p,xp,yip in zip(ps_xis,xis,ps,xps,yips_closest):
                P_xy_xpyp[x,yi,xp,yip]=p
                P_xy_xpyp_alpha[x,yi,xp,yip]=pxi
                Xi_xy_xpyp[x,yi,xp,yip]=xi

    return(P_xy_xpyp,P_xy_xpyp_alpha,Xi_xy_xpyp)
