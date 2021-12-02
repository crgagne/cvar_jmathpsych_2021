import numpy as np
import pickle

class Agent:
    def __init__(self, p: dict, maze):
        self.maze = maze
        self.num_states = p['num_states']
        self.num_actions = p['num_actions']
        self.start_states = p['start_states']
        self.r_params = p['r_params']


def init_maze(side_i, side_j, obst: list):
    '''Initialise the maze environment'''
    maze   = np.zeros((side_i, side_j))
    for i in range(len(obst)):
        maze[obst[i][0], obst[i][1]] = 1
    return maze

def idcs2state(idcs: list, maze):
    '''Convert state idcs to state id
       Row major [[1,2,3],[4,5,6]]
       Different from M&D and Matlab, which is column major
    '''
    si = idcs[0]
    sj = idcs[1]
    side_j = maze.shape[1]

    return si*side_j + sj

def state2idcs(s: int, maze, state_mult=1):
    '''Convert state id to state idcs'''

    # convert state to location id
    num_locs = maze.shape[0]*maze.shape[1]
    loc = s%(num_locs)

    # convert location id to i,j coordinates
    side_j = maze.shape[1]
    si = loc // side_j
    sj = loc % side_j

    return [int(si), int(sj)]


def setup_params(testcase=0,extra_name=''):

    # Create maze
    side_i = 3
    side_j = 9

    obst = []
    obst_states =[]

    state_mult = 1
    num_actions = 4

    if testcase==1:
        side_i = 5
        side_j = 7
        # cliff scenario #
        r_function=np.zeros((side_i*side_j*state_mult,side_i*side_j*state_mult,num_actions,1)) # R(s,s',a,outcomes)
        r_function[:,29:34,:,:] = 1
        r_function[:,27,:,:] = -3
        start_states = [21]

    elif testcase==2:
        r_function=np.zeros((side_i*side_j*state_mult,side_i*side_j*state_mult,num_actions,1)) # R(s,s',a,outcomes)
        r_function[:,18,:,:] = 20
        r_function[:,19,:,:] = -2
        r_function[:,25,:,:] = -1
        start_states = [4]

    elif testcase==3:
        r_function=np.zeros((side_i*side_j*state_mult,side_i*side_j*state_mult,num_actions,1)) # R(s,s',a,outcomes)
        r_function[:,18,:,:] = 20
        r_function[:,25,:,:] = -1
        start_states = [6]
    elif testcase==4:
        num_actions = 3
        side_i = 1
        side_j = 9
        r_function=np.zeros((side_i*side_j*state_mult,side_i*side_j*state_mult,num_actions,1)) # R(s,s',a,outcomes)
        r_function[:,0,:,:] = 10
        r_function[:,1,:,:] = -2
        r_function[:,7,:,:] = -1
        start_states = [4]
    if testcase==6:
        side_i = 5
        side_j = 7
        # cliff scenario #
        r_function=np.zeros((side_i*side_j*state_mult,side_i*side_j*state_mult,num_actions,1)) # R(s,s',a,outcomes)
        r_function[:,29:34,:,:] = 1
        r_function[:,27,:,:] = -5
        start_states = [21]

    num_locs = side_i*side_j

    # terminal states states
    goal_states = []

    # Reward function, shape = R(s,a,outcomes)

    r_params = {'type':'(s,a)',
         'r_function': r_function,
         'Qrange':[-1.2,1.2], # .............. #
         }

    maze = init_maze(side_i, side_j, obst)
    start_idcs = np.array([2, 0])

    # Parameters
    p = {} #default_p # load Mattar & Daw defaults

    p['start_location']=start_idcs # .... # Start location indices
    p['start_states']=start_states # .... # Starting to replace with this.
    p['goal_states']=goal_states
    p['obst_states'] = obst_states
    p['r_params']=r_params # ................... # Reward parameters

    # transition structure of MDP
    p['noise_mode'] = 'noisy'
    p['err_prob'] = 0.1


    return(p,maze)
