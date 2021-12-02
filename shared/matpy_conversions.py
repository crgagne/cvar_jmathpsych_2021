import numpy as np
from scenarios import state2idcs

def convert_python_s_to_mat(s_py,
                            agent,
                            maze_for_matlab = None,
                            neg_term_states = [],
                            obst = None, #[[4,0],[4,1],[4,2],[4,3],[4,4],[4,5]],
                            num_interp_points=21,
                            num_actions=4):

    maze = agent.maze
    # reshape
    state_order_mat = np.arange(maze.shape[0]*maze.shape[1]).reshape(maze.shape).flatten(order='F')

    # these provide the opposite style of index
    state_order_py = np.arange(maze.shape[0]*maze.shape[1]).reshape(maze.shape,order='C').flatten(order='F')


    # remove neg term states, if they are in obstacles
    #neg_term_states=[idcs2state(obs,maze_for_matlab) for obs in obst ]

    for ss in neg_term_states:
        state_order_mat = np.delete(state_order_mat,np.where(state_order_mat==ss))
        # contains [0,6,18, etc.] which are the python numbering in the order of matlab.

    # find where the python states
    s_mat = np.where(state_order_mat==s_py)[0][0]
    return(s_mat)


def convert_mat_s_to_python(s_mat,
                            agent,
                            maze_for_matlab = None,
                            neg_term_states = [],
                            obst = None, #[[4,0],[4,1],[4,2],[4,3],[4,4],[4,5]],
                            num_interp_points=21,
                            num_actions=4):
    maze = agent.maze
    # reshape
    state_order_mat = np.arange(maze.shape[0]*maze.shape[1]).reshape(maze.shape).flatten(order='F')

    # remove neg term states
    #neg_term_states=[idcs2state(obs,maze_for_matlab) for obs in obst ]
    for ss in neg_term_states:
        state_order_mat = np.delete(state_order_mat,np.where(state_order_mat==ss))

    try:
        return(state_order_mat[s_mat])
    except:
        #import pdb; pdb.set_trace()
        return(-1)




def convert_SI_function(V,agent,neg_term_states=[],num_inter_points=21):
    '''
    Input: S x NumInterpPoints
    Output: Python ordered version
    '''
    Ns = agent.num_states
    maze = agent.maze
    V_CVaR_mat = np.zeros((Ns,num_inter_points))
    state_order_mat = np.arange(Ns).reshape(maze.shape).flatten(order='F')

    i = 0
    for s in state_order_mat:
        if s in neg_term_states:
            V_CVaR_mat[s,:] = V[-1][:] # takes last value from matlab (crashes are always last state)
        else:
            V_CVaR_mat[s,:] = V[i,:]
            i+=1
    return(V_CVaR_mat)

def convert_SIA_function(Q,agent,neg_term_states=[],num_inter_points=21,pi_or_q='q'):
    '''
    Input: S x NumInterpPoints X A  (Q-values or Pis)
    Output: Python ordered version
    '''
    Ns = agent.num_states
    Na = agent.num_actions
    maze = agent.maze
    Q_CVaR_mat = np.zeros((Ns,num_inter_points,Na))
    state_order_mat = np.arange(Ns).reshape(maze.shape).flatten(order='F')

    i = 0
    for s in state_order_mat:
        if s in neg_term_states:
            if pi_or_q=='q':
                Q_CVaR_mat[s,:,:] = Q[-1,:,:] # takes last value from matlab (crashes are always last state)
            elif pi_or_q=='pi':
                Q_CVaR_mat[s,:,:] = 0
        else:
            Q_CVaR_mat[s,:,:] = Q[i,:,:]
            i+=1
    return(Q_CVaR_mat)

def convert_SA_function(Q,agent,neg_term_states=[],p_or_q='q'):
    '''
    Input: S X A  (Q-values or Pis)
    Output: Python ordered version
    '''
    Ns = agent.num_states
    Na = agent.num_actions
    maze = agent.maze
    Q_CVaR_mat = np.zeros((Ns,Na))
    state_order_mat = np.arange(Ns).reshape(maze.shape).flatten(order='F')

    i = 0
    for s in state_order_mat:
        if s in neg_term_states:
            if pi_or_q=='q':
                Q_CVaR_mat[s,:] = Q[-1,:] # takes last value from matlab (crashes are always last state)
            elif pi_or_q=='pi':
                Q_CVaR_mat[s,:] = 0
        else:
            Q_CVaR_mat[s,:] = Q[i,:]
            i+=1
    return(Q_CVaR_mat)

def convert_IntPol_function(Pol,agent,neg_term_states=[],num_inter_points=21,p_or_q='q'):
    '''
    Input: S X NumInterpPoints  with integer values for policy (Pol)
    Output: Python ordered version
    '''
    Ns = agent.num_states
    Na = agent.num_actions
    maze = agent.maze
    Pol_CVaR_mat = np.zeros((Ns,num_inter_points,Na))
    state_order_mat = np.arange(Ns).reshape(maze.shape).flatten(order='F')

    i = 0
    for s in state_order_mat:
        if s in neg_term_states:
            Pol_CVaR_mat[s,:,:] = 0
        else:
            for thresh in range(num_inter_points):
                a = Pol[i,thresh]-1
                Pol_CVaR_mat[s,thresh,a] = 1
            i+=1
    return(Pol_CVaR_mat)


def convert_P_alpha(P_alpha_mat,agent,neg_term_states=[]):
    '''
    Input: S x S;
    e.g.
    mat['P_pi_old_alpha_storage'][0,:,:,y_ind]

    '''
    Ns_mat = P_alpha_mat.shape[0] # agent.num_states # using Matlab size
    Ns_mat = Ns_mat - 1 # remove last state which is absorbing

    Ns_py = agent.num_states
    maze = agent.maze
    P_alpha_py = np.zeros((Ns_py,Ns_py))
    state_order_mat = np.arange(Ns_py).reshape(maze.shape).flatten(order='F')

    for ss in neg_term_states:
        state_order_mat = np.delete(state_order_mat,np.where(state_order_mat==ss))

    for s in range(Ns_mat):
        s_py = state_order_mat[s] # convert to python
        if s_py in neg_term_states:
            P_alpha_py[s_py,:]=0
        else:
            for sp in range(Ns_mat):
                sp_py = state_order_mat[sp]
                if sp_py in neg_term_states:
                    P_alpha_py[s_py,sp_py]=0
                else:
                    P_alpha_py[s_py,sp_py] = P_alpha_mat[s,sp] #mat['P_pi_old_alpha_storage'][0,s,sp,y_ind]
    return(P_alpha_py)

def convert_s_by_s_to_by_a(P,agent):
    '''
    Input: SxS

    Returns SxA for plotting
    '''

    P_by_actions = np.zeros((agent.num_states,4))

    for s in range(agent.num_states):
        s1s_visited = []
        for a in range(4):
            s_idcs = state2idcs(s,agent.maze) # get current state idcs
            p_next_state = get_next_state_prob_from_maze(s_idcs,a,agent.maze,noise_mode='none') # get deterministic next state
            assert len(np.where(p_next_state)[0])<2 # make sure only one next state
            s1 = np.where(p_next_state)[0][0] # get its index
            s1s_visited.append(s1)
            P_by_actions[s,a]=P[s,s1] # put in the probabilities from the s x s matrix.

        # probably adjustment for same state visited
        #if s==0:
        #    import pdb; pdb.set_trace()
        s1s_visited=np.array(s1s_visited)
        (unique, counts) = np.unique(s1s_visited, return_counts=True)
        for s1,s1counts in zip(unique[counts>1],counts[counts>1]):
            P_by_actions[s,s1s_visited==s1]=P_by_actions[s,s1s_visited==s1]/s1counts


    return(P_by_actions)
