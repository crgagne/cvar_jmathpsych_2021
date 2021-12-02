import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scenarios import state2idcs, idcs2state


def myround(x, prec=1, base=.5):
    return round(base * round(float(x)/base),prec)

def action2str(a,num_actions=4):
    if num_actions==2:
        if a == 0: # right
            return('right')
        elif a == 1:  # left
            return('left')
    elif num_actions==3:
        if a == 0: # up
            return('stay')
        elif a == 1: # right
            return('right')
        elif a == 2:  # left
            return('left')
    elif num_actions==4:
        if a == 0: # up
            return('up')
        elif a == 1: # down
            return('down')
        elif a == 2: # right
            return('right')
        elif a == 3:  # left
            return('left')

def label_cell_at_action(grid_ax,idxloc,value,action):

     # xy are backwards
    if action==0: # up
        relloc = [0.2,0.40] # relative location to index loc
    elif action==1: # down
        relloc = [0.95,0.40]
    elif action==2: # right
        relloc = [0.55,0.68]
    elif action==3: # left
        relloc = [0.55,0.02]
    # text
    grid_ax.annotate(np.round(value,2),(idxloc[1]+relloc[1],idxloc[0]+relloc[0]),fontsize=8)


def plot_goal(ax,goal_idcs):

    # Plot goal location
    if goal_idcs is not None:
        if len(goal_idcs.shape)>1: # more than one goal?
            for goal_idx in goal_idcs:
                goal = ax.scatter(goal_idx[1],goal_idx[0], s=100, c='crimson', marker=r'$\clubsuit$', alpha=0.7)
        else:
            goal = ax.scatter(goal_idcs[1]+0.5,goal_idcs[0]+0.5, s=1000, c='crimson', marker=r'$\clubsuit$', alpha=0.7)

def add_walls(ax,maze):
    # find wall locations #
    wall_loc_coords = np.array(np.where(maze==1)).T # make vertical
    for i in range(wall_loc_coords.shape[0]):
        wcoord = wall_loc_coords[i,:]
        ax.add_patch(patches.Rectangle((wcoord[1]-0.5,wcoord[0]-0.5),1,1,linewidth=1,edgecolor='k',facecolor='k'))

def add_agents(ax,start_idcs):
    agent_circles = []
    agent_circles.append(plt.Circle(start_idcs[::-1],radius=0.1,color='b',alpha=1))
    agent_circles.append(plt.Circle(start_idcs[::-1],radius=0.1,color='b',alpha=0.5))
    ax.add_patch(agent_circles[0])
    ax.add_patch(agent_circles[1])
    return(agent_circles)




def add_triangles(ax,maze,cm,add_labels=False,fs=6,cm_empty=10,ec=(0.4,0.4,0.4,0.1),ls='-',term_states_for_plotting=[]):
    #print(ls)
    triangles = {}
    labels = {}

    for x in range(maze.shape[1]): # these are reversed
        for y in range(maze.shape[0]):

            if not idcs2state([y,x],maze) in term_states_for_plotting:

                triangles[str(x)+'_'+str(y)+'_up']=plt.Polygon([[x-0.2,y-0.25], [x+0.2,y-0.25], [x+0,y-0.45]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)
                triangles[str(x)+'_'+str(y)+'_down']=plt.Polygon([[x-0.2,y+0.25], [x+0.2,y+0.25], [x+0,y+0.45]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)
                triangles[str(x)+'_'+str(y)+'_left']=plt.Polygon([[x-0.25,y-0.2], [x-0.25,y+0.2], [x-0.45,y+0]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)
                triangles[str(x)+'_'+str(y)+'_right']=plt.Polygon([[x+0.25,y-0.2], [x+0.25,y+0.2], [x+0.45,y+0]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)

                ax.add_patch(triangles[str(x)+'_'+str(y)+'_up'])
                ax.add_patch(triangles[str(x)+'_'+str(y)+'_down'])
                ax.add_patch(triangles[str(x)+'_'+str(y)+'_left'])
                ax.add_patch(triangles[str(x)+'_'+str(y)+'_right'])

                if add_labels:
                    labels[str(x)+'_'+str(y)+'_up'] = ax.annotate('',[x-0.1,y-0.3],fontsize=fs)
                    labels[str(x)+'_'+str(y)+'_down'] = ax.annotate('',[x-.1,y+0.35],fontsize=fs)
                    labels[str(x)+'_'+str(y)+'_left'] = ax.annotate('',[x-0.45,y],fontsize=fs)
                    labels[str(x)+'_'+str(y)+'_right'] = ax.annotate('',[x+0.25,y],fontsize=fs)

    return(triangles,labels)

def add_horiz_triangles(ax,maze,cm,add_labels=False,fs=6,cm_empty=10,ec=(0.4,0.4,0.4,0.1),ls='-',term_states_for_plotting=[]):
    #print(ls)
    triangles = {}
    labels = {}

    for x in range(maze.shape[1]): # these are reversed
        for y in range(maze.shape[0]):

            if not idcs2state([y,x],maze) in term_states_for_plotting:

                #triangles[str(x)+'_'+str(y)+'_up']=plt.Polygon([[x-0.2,y-0.25], [x+0.2,y-0.25], [x+0,y-0.45]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)
                #triangles[str(x)+'_'+str(y)+'_down']=plt.Polygon([[x-0.2,y+0.25], [x+0.2,y+0.25], [x+0,y+0.45]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)
                triangles[str(x)+'_'+str(y)+'_left']=plt.Polygon([[x-0.2,y-0.15], [x-0.2,y+0.15], [x-0.4,y+0]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)
                triangles[str(x)+'_'+str(y)+'_right']=plt.Polygon([[x+0.2,y-0.15], [x+0.2,y+0.15], [x+0.4,y+0]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)

                #ax.add_patch(triangles[str(x)+'_'+str(y)+'_up'])
                #ax.add_patch(triangles[str(x)+'_'+str(y)+'_down'])
                ax.add_patch(triangles[str(x)+'_'+str(y)+'_left'])
                ax.add_patch(triangles[str(x)+'_'+str(y)+'_right'])

                if add_labels:
                    #labels[str(x)+'_'+str(y)+'_up'] = ax.annotate('',[x-0.1,y-0.3],fontsize=fs)
                    #labels[str(x)+'_'+str(y)+'_down'] = ax.annotate('',[x-.1,y+0.35],fontsize=fs)
                    labels[str(x)+'_'+str(y)+'_left'] = ax.annotate('',[x-0.45,y],fontsize=fs)
                    labels[str(x)+'_'+str(y)+'_right'] = ax.annotate('',[x+0.25,y],fontsize=fs)

    return(triangles,labels)

def add_stayhoriz_triangles(ax,maze,cm,add_labels=False,fs=6,cm_empty=10,ec=(0.4,0.4,0.4,0.1),ls='-',term_states_for_plotting=[]):
    #print(ls)
    triangles = {}
    labels = {}

    for x in range(maze.shape[1]): # these are reversed
        for y in range(maze.shape[0]):

            if not idcs2state([y,x],maze) in term_states_for_plotting:

                triangles[str(x)+'_'+str(y)+'_stay']=plt.Polygon([[x-0.2,y+0.2],[x-0.2,y-0.25],[x+0.2,y-0.25],[x+0.2,y+0.2]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)
                #triangles[str(x)+'_'+str(y)+'_down']=plt.Polygon([[x-0.2,y+0.25], [x+0.2,y+0.25], [x+0,y+0.45]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)
                triangles[str(x)+'_'+str(y)+'_left']=plt.Polygon([[x-0.25,y-0.2], [x-0.25,y+0.2], [x-0.45,y+0]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)
                triangles[str(x)+'_'+str(y)+'_right']=plt.Polygon([[x+0.25,y-0.2], [x+0.25,y+0.2], [x+0.45,y+0]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)

                ax.add_patch(triangles[str(x)+'_'+str(y)+'_stay'])
                #ax.add_patch(triangles[str(x)+'_'+str(y)+'_down'])
                ax.add_patch(triangles[str(x)+'_'+str(y)+'_left'])
                ax.add_patch(triangles[str(x)+'_'+str(y)+'_right'])

                if add_labels:
                    labels[str(x)+'_'+str(y)+'_stay'] = ax.annotate('',[x-0.1,y-0.3],fontsize=fs)
                    #labels[str(x)+'_'+str(y)+'_down'] = ax.annotate('',[x-.1,y+0.35],fontsize=fs)
                    labels[str(x)+'_'+str(y)+'_left'] = ax.annotate('',[x-0.45,y],fontsize=fs)
                    labels[str(x)+'_'+str(y)+'_right'] = ax.annotate('',[x+0.25,y],fontsize=fs)

    return(triangles,labels)

def add_stay_3_triangles(ax,maze,cm,add_labels=False,fs=6,cm_empty=10,ec=(0.4,0.4,0.4,0.1),ls='-',term_states_for_plotting=[]):
    #print(ls)
    triangles = {}
    labels = {}

    for x in range(maze.shape[1]): # these are reversed
        for y in range(maze.shape[0]):

            if not idcs2state([y,x],maze) in term_states_for_plotting:

                #triangles[str(x)+'_'+str(y)+'_stay']=plt.Polygon([[x-0.2,y+0.2],[x-0.2,y-0.25],[x+0.2,y-0.25],[x+0.2,y+0.2]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)
                triangles[str(x)+'_'+str(y)+'_stay']=plt.Polygon([[x-0.2,y-0.45], [x+0.2,y-0.45], [x+0,y-0.25]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)
                triangles[str(x)+'_'+str(y)+'_left']=plt.Polygon([[x-0.25,y-0.2], [x-0.25,y+0.2], [x-0.45,y+0]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)
                triangles[str(x)+'_'+str(y)+'_right']=plt.Polygon([[x+0.25,y-0.2], [x+0.25,y+0.2], [x+0.45,y+0]],fc=cm[cm_empty],ec=ec,alpha=1,linestyle=ls)

                ax.add_patch(triangles[str(x)+'_'+str(y)+'_stay'])
                #ax.add_patch(triangles[str(x)+'_'+str(y)+'_down'])
                ax.add_patch(triangles[str(x)+'_'+str(y)+'_left'])
                ax.add_patch(triangles[str(x)+'_'+str(y)+'_right'])

                if add_labels:
                    labels[str(x)+'_'+str(y)+'_stay'] = ax.annotate('',[x-0.1,y-0.3],fontsize=fs)
                    #labels[str(x)+'_'+str(y)+'_down'] = ax.annotate('',[x-.1,y+0.35],fontsize=fs)
                    labels[str(x)+'_'+str(y)+'_left'] = ax.annotate('',[x-0.45,y],fontsize=fs)
                    labels[str(x)+'_'+str(y)+'_right'] = ax.annotate('',[x+0.25,y],fontsize=fs)

    return(triangles,labels)



def action_offset(action):
    '''Relative to the state value coordinates. Where are the actions?'''
    # up,down,left,right
    if action==0:
        offset = [-0.25,0]
    elif action==1:
        offset = [0.25, 0]
    elif action==2:
        offset = [0,0.25]
    elif action==3:
        offset = [0,-0.25]
    return(offset)

def check_for_lowest_color(color,new_q,cm,n_colors,middle_or_end='middle',roundoff=3):
    '''Middle or end is whether white is in the middle or left end of the color array'''

    if np.isnan(new_q):
        # return white (middle color)
        return(cm[int(n_colors/2)])

    # if np.round(new_q,2)==0.09:
    #     import pdb; pdb.set_trace()

    if middle_or_end=='middle': # white is middle color
        # check for non-zero q's; give them lowest color
        if new_q<0 and np.round(new_q,roundoff)==0:
            color = cm[int(n_colors/2)-1]
        elif new_q>0 and np.round(new_q,roundoff)==0:
            color = cm[int(n_colors/2)+1]
    # else:
    #     if new_q<0 and np.round(new_q,1)==0:
    #         color = cm[1]
    #     elif new_q>0 and np.round(new_q,1)==0:
    #         color = cm[1]

    return(color)



def plot_all_qvalues(Q_table,trianglelist,maze,cm,Qrange_discrete,labellist=None,q_or_pi='q',roundoff=3,term_states_for_plotting=[],num_actions=4):
    '''Updates a trianglelist and labellist with Q-values
    '''

    n_colors = len(cm)-1
    statelist = np.arange(Q_table.shape[0])
    actionlist = np.arange(Q_table.shape[1])
    for s in statelist:
        for a in actionlist:

            if s not in term_states_for_plotting:
                #import pdb; pdb.set_trace()
                si = state2idcs(s,maze)
                new_q = Q_table[s,a]

                #color = cm[int(n_colors/2)+int(np.round(new_q,1)*int(n_colors/2))] # scale value by max and find index
                color = cm[np.argmin(np.abs(new_q-Qrange_discrete))]
                #print(color)
                if q_or_pi=='q':
                    color = check_for_lowest_color(color,new_q,cm,n_colors,roundoff=roundoff)
                #print(new_q)
                #print(color)
                try:
                    trianglelist[0]['_'.join(str(x) for x in si[::-1])+'_'+action2str(a,num_actions=num_actions)].set_fc(color)
                except:
                    import pdb; pdb.set_trace()
                trianglelist[0]['_'.join(str(x) for x in si[::-1])+'_'+action2str(a,num_actions=num_actions)].set_linestyle('None')
                if labellist is not None:
                    labellist[0]['_'.join(str(x) for x in si[::-1])+'_'+action2str(a,num_actions=num_actions)].set_text(str(np.round(new_q,roundoff)))
                    labellist[0]['_'.join(str(x) for x in si[::-1])+'_'+action2str(a,num_actions=num_actions)].set_color('k')


def plot_q_or_pi(Q,V,title,ax,maze,q_or_pi='q',Qrange=None,roundoff=3,annot_value=False,
    value_fontsize=8,n_colors = 20,
    inc_triangles=True,tri_ls='-',tri_ec = (0.4,0.4,0.4,0.1),tri_fs=8,tri_add_labels=True,
    plot_value=True,term_states_for_plotting=[],pi_color="black",tri_type='all',colorbar=False):
    '''Wrapper to the plot_all_qvalues
       Can be used for pi's or Q-vales.
       Give an axis.
    '''
    #import pdb; pdb.set_trace()

    # color map
    if q_or_pi=='q':

        if Qrange is None:
            minmaxQ = np.max(np.abs(Q))
            minmaxV = np.max(np.abs(V))
            Qrange = [-1*np.max((minmaxQ,minmaxV)),np.max((minmaxQ,minmaxV))]
            if Qrange[0]==Qrange[1]:
                Qrange = [-1,1]

        cm_Q = sns.light_palette("red",int(n_colors/2))[::-1]+[(0.96, 0.96, 0.96)]+sns.light_palette("green",int(n_colors/2))
        # if invert_colors:
        #     cm_Q = cm_Q[::-1]
        #     Qrange = Qrange[::-1]
        Qrange_discrete = list(np.linspace(Qrange[0],-1*Qrange[1]/(n_colors/2),int(n_colors/2)))+\
                          [0]+\
                          list(np.linspace(Qrange[1]/(n_colors/2),
                             Qrange[1],int(n_colors/2)))
        cm_empty=int(n_colors/2)
        #import pdb; pdb.set_trace()


    elif q_or_pi=='pi':
        cm_Q = [(0.96, 0.96, 0.96)]+sns.light_palette(pi_color,int(n_colors)-1)

        if Qrange is None:
            maxV = np.max(np.abs(V))
            Qrange = [0,maxV]
            #print(Qrange)

        Qrange_discrete = list(np.linspace(Qrange[0],Qrange[1],n_colors))
        cm_empty=0



    assert len(cm_Q)==len(Qrange_discrete)

    im_value = None
    if plot_value:
        #value
        im_value = ax.imshow(V.reshape(maze.shape),
                                  interpolation='none',origin='upper',
                                  cmap = matplotlib.colors.ListedColormap(cm_Q),
                                  vmax=Qrange[1],
                                  vmin=Qrange[0],
                                  )

    if annot_value:
        # Loop over data dimensions and create text annotations.
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                value = np.round(V.reshape(maze.shape)[i,j],roundoff)
                if value!=0.0:
                    text = ax.text(j, i, value,fontsize=value_fontsize,
                                   ha="center", va="center", color="k")


        # triangles
    trianglelist = []
    labellist = []
    if inc_triangles:
        if tri_type=='all':
            tri,lab = add_triangles(ax,maze,cm_Q,add_labels=tri_add_labels,cm_empty=cm_empty,ec=tri_ec,ls=tri_ls,fs=tri_fs,term_states_for_plotting=term_states_for_plotting)
        elif tri_type=='stayhoriz':
            tri,lab = add_stayhoriz_triangles(ax,maze,cm_Q,add_labels=tri_add_labels,cm_empty=cm_empty,ec=tri_ec,ls=tri_ls,fs=tri_fs,term_states_for_plotting=term_states_for_plotting)
        elif tri_type=='stay_3':
            tri,lab = add_stay_3_triangles(ax,maze,cm_Q,add_labels=tri_add_labels,cm_empty=cm_empty,ec=tri_ec,ls=tri_ls,fs=tri_fs,term_states_for_plotting=term_states_for_plotting)
        elif tri_type=='horiz':
            tri,lab = add_horiz_triangles(ax,maze,cm_Q,add_labels=tri_add_labels,cm_empty=cm_empty,ec=tri_ec,ls=tri_ls,fs=tri_fs,term_states_for_plotting=term_states_for_plotting)

        trianglelist.append(tri)
        labellist.append(lab)

    if tri_add_labels==False:
        labellist=None

    if inc_triangles:
        # put q-values into triangles
        if tri_type=='all':
            num_actions=4
        elif tri_type=='stayhoriz':
            num_actions=3
        elif tri_type=='stay_3':
            num_actions=3
        elif tri_type=='horiz':
            num_actions=2

        plot_all_qvalues(Q,
                         trianglelist,maze,cm_Q,Qrange_discrete,
                         labellist=labellist,q_or_pi=q_or_pi,roundoff=roundoff,term_states_for_plotting=term_states_for_plotting,num_actions=num_actions)

    ax.set_title(title)
    return(trianglelist,im_value)


def embellish_plot(ax,maze,agent,s0_py,cost,corner_labels,color_agent='b',
    center_rewards=False,r_fontsize=8,add_rewards=True,ec='k',fc='white',alpha=1,outer_lw=3,reward_color='rg'):

    add_walls(ax,maze)

    # add grids
    ax.set_yticks(np.arange(0, maze.shape[0], 1));
    ax.set_yticks(np.arange(-.5, maze.shape[0], 1), minor=True);
    ax.grid(True,which='minor', color='k', linestyle='-', linewidth=0.5,axis='both')
    ax.set_xticks(np.arange(0, maze.shape[1], 1));
    ax.set_xticks(np.arange(-.5, maze.shape[1], 1), minor=True);
    ax.grid(True,which='minor', color='k', linestyle='-', linewidth=0.5,axis='both')

    # add the rewards
    if add_rewards:
        reward_states = np.where(agent.r_params['r_function'][0,:,0,0]!=0)[0]
        for gs in reward_states:
            rs = np.unique(agent.r_params['r_function'][:,gs,:,:])

            if cost:
                rs = -1*rs
            r_str = str(int(rs[0]))

            if rs>0:
                color=sns.color_palette()[2]#'g'
            else:
                color=sns.color_palette()[3] #'r'
            if reward_color=='k':
                color='k'
                if rs[0]>0:
                    r_str = '+'+r_str

            s_idcs = state2idcs(gs,maze)
            if center_rewards==False:
                xoffset = -0.45
            else:
                xoffset = -0.1
            ax.text(s_idcs[1]+xoffset,s_idcs[0]+0.47,'r='+r_str,fontsize=r_fontsize,color=color,
                    bbox=dict(edgecolor=ec,facecolor=fc, alpha=alpha,pad=2))


    # add in start state as dot (or agent)
    start_state = s0_py
    if start_state is not None:
        start_idcs = state2idcs(start_state,maze)
        plt.scatter(start_idcs[1]-0.35,start_idcs[0]-0.35,color=color_agent,marker='D',s=75)

    if corner_labels:
        #add numbers in the upper corner
        for s in range(maze.shape[0]*maze.shape[1]):
            s_idcs = state2idcs(s,maze)
            if s==start_state:
                extra = '=s0'
                color = 'k'
            else:
                extra = ''
                color = 'k'
            ax.text(s_idcs[1]-0.45,s_idcs[0]-0.35,str(int(s))+extra,fontsize=12,color=color)

    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    # darken the outside
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(str(outer_lw))
