import numpy as np

def induce_P(P,Pol_CVaR,Ny,Ns,y_pr,fixed_y=False,pol_type='deterministic',Na=4):
    '''Convert P matrix (x,x',a) to induced P matrix over (x,x',y);
    i.e. get rid of action dimensions and introduce risk preference dimension'''

    P_ssy = np.zeros((P.shape[0:2])+(Ny,))

    for x in range(Ns):
        for y in range(Ny):
            if pol_type=='random':
                for a in range(Na):
                    P_ssy[x,:,y] += (1/Na)*P[x,:,a]
            elif pol_type=='stochastic':
                for a in range(Na):
                    P_ssy[x,:,y] += Pol_CVaR[x,y,a]*P[x,:,a]
            elif pol_type=='deterministic':
                if fixed_y:
                    P_ssy[x,:,y] = P[x,:,Pol_CVaR[x,y_pr]-1] # policy fixed in y (y_pr)
                else:
                    P_ssy[x,:,y] = P[x,:,Pol_CVaR[x,y]-1] # policy varying in y
            elif pol_type=='reversed':
                P_ssy[x,:,y] = P[x,:,Pol_CVaR[x,(Ny-1)-y]-1] # policy varying in y

    return(P_ssy)
