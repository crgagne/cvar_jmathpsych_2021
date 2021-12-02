from plotting_basics import plot_q_or_pi,embellish_plot
from plotting_oneD import plot_1D_arrows
from matpy_conversions import convert_mat_s_to_python, convert_SI_function, convert_SIA_function
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

from distorted_probabilities import create_distorted_probability_matrix
from plotting_distorted_probabilities import plot_distorted_probabilities
from scenarios import setup_params, Agent, state2idcs


def plot_xy_replay(example_name,
                    agent,
                    testcase = 4,
                    subcase='aS',
                    x_eval = 4,
                    y_eval = 1,
                    iters = 60,
                    last_step=1,
                    yis = [5,4,3,2,1], # should go in reverse
                    savefolder='prioritized_replay_XY',
                    save=False,
                    extra_loadname='',
                    remove_reward_color=False,
                    title_alpha=0.05):

    fs = 18
    # load data
    savename = '../simulation_results/'+example_name+\
        '_prioritized_VI_pCVaR_'+extra_loadname+'ypr'+str(y_eval)+'_xeval'+str(x_eval)+'_iters'+str(iters)+'.pkl'
    with open(savename,'rb') as f:
        results = pickle.load(f)

    # # Specify Save Folder
    # savefolder = '../figs/'+savefolder+'/testcase'+str(4)+subcase+'_xeval_'+str(x_eval)+'yeval_'+str(y_eval)
    # if not os.path.isdir(savefolder):
    #     os.mkdir(savefolder)
    # print(savefolder)

    # Shared Some Things
    y_set = results['y_set']
    Ny = results['Ny']
    x_eval_py = convert_mat_s_to_python(results['x_eval'],agent,num_actions=3,num_interp_points=Ny)
    y_eval_py = y_eval

    Qrange = [-10,10]
    term_states_for_plotting = [0]

    # Set up plot
    f = plt.figure(figsize=(8,8),dpi=200)
    yis = yis
    L = len(yis)

    # Set up left and right axes
    axes_left = []
    for i in range(len(yis)):
        #axes_left.append(plt.axes([0.07*(L), 1-0.15*i, 0.75,0.15]))
        axes_left.append(plt.axes([0.07*(L), 1-0.15*i, 1,0.20]))
    # axes_right = []
    # for i in range(len(yis)):
    #     axes_right.append(plt.axes([0.8+0.07*(L), 1-0.15*(i-0.2)+0.01, 0.18,0.07]))

    # Get value for current replay iteration
    V_py = convert_SI_function(results['V_storage'][last_step,:,:],agent,neg_term_states=[],num_inter_points=Ny)
    Q_py = convert_SIA_function(results['Q_storage'][last_step,:,:,:],agent,neg_term_states=[],num_inter_points=Ny)
    Pol_py = convert_SIA_function(results['Pol_storage'][last_step,:,:,:],agent,neg_term_states=[],num_inter_points=Ny)

    #import pdb; pdb.set_trace()
    if remove_reward_color:
        for yy in range(V_py.shape[1]):
            for xx in range(V_py.shape[0]):
                if (xx,yy) not in results['replay_storage'][0:(last_step+1)]:
                    if xx!=0:
                        V_py[xx,yy]=0


    # Loop through y's and left axes
    for i,y_ind in enumerate(yis):
        ax = axes_left[i]
        plt.sca(ax)

        y = y_set[y_ind]

        # Values
        v = -1*V_py[:,y_ind].reshape(agent.maze.shape)
        q = -1*Q_py[:,y_ind,:]

        plot_q_or_pi(q,v,'',ax,agent.maze,q_or_pi='q',
                    roundoff=1,inc_triangles=False,tri_type='stay_3',tri_add_labels=False,annot_value=False,Qrange=Qrange,
                     term_states_for_plotting=term_states_for_plotting,n_colors=100)

        # Policy as Arrows
        plot_1D_arrows(Pol_py[:,y_ind,:],
                       y_ind,
                       agent.maze,
                       term_states_for_plotting,
                       question=True,
                       box=True)

        # Embellish
        embellish_plot(ax,agent.maze,
            agent,
            None,
            cost=True,
            corner_labels=False,
            color_agent='b',
            center_rewards=False,
            r_fontsize=10,
            add_rewards=True,ec='white',fc='white',alpha=0.0,outer_lw=1,reward_color='k')

        trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
        ann = ax.annotate(str(np.round(y,2)), xy=(-1.26,0.5), xycoords=trans,fontsize=fs)

        # Add lava pit
        arr_img = mpimg.imread('../mscl/lava_pit.png')
        imagebox = OffsetImage(arr_img , zoom=.014)
        ab = AnnotationBbox(imagebox, (0, .0),pad=0)
        ax.add_artist(ab)

    # y-axis
    x0=0.3; y0=0.32
    x1=0.3; y1=1.12
    plt.arrow(x1,y1,x0-x1,y0-y1,transform=plt.gcf().transFigure,
             color='k',clip_on=False,head_width=.01)
    plt.arrow(x0,y0,x1-x0,y1-y0,transform=plt.gcf().transFigure,
             color='k',clip_on=False,head_width=.01)
    #if static_or_dynamic=='static':
    #yaxislabel=r'static risk preference ($\alpha$)'
    #elif static_or_dynamic=='dynamic':
    yaxislabel=r'dynamic risk preference ($\alpha$)'
    plt.text(x0-0.05, # offset text by 0.05 horizontally
             y0+0.2*(y1-y0), # center vertically
                 yaxislabel,
                 transform=plt.gcf().transFigure,clip_on=False,
                 rotation=90,fontsize=fs)

    # x-axis
    x0=0.35; y0=0.2
    x1=1.3; y1=0.2
    xoffset = 0.09
    plt.arrow(x1+xoffset,y1,x0-x1,y0-y1,transform=plt.gcf().transFigure,
             color='k',clip_on=False,head_width=.01)
    plt.arrow(x0+xoffset,y0,x1-x0,y1-y0,transform=plt.gcf().transFigure,
              color='k',clip_on=False,head_width=.01)
    plt.text((x0+x1)/2,
             y0-0.05,r'state ($x$)',
                 transform=plt.gcf().transFigure,clip_on=False,
                 rotation=0,fontsize=fs)
    for i in range(1,10):
        xoffset = 0.11
        plt.text(x0+xoffset*i,y0+0.03,str(i),
                 transform=plt.gcf().transFigure,clip_on=False,
                 rotation=0,fontsize=fs)

    # start
    if y_eval==1:
        ytext=0.46
    if y_eval==2:
        ytext=0.61
    if y_eval==3:
        ytext=0.76
    if y_eval==4:
        ytext=0.91
    if y_eval==5:
        ytext=1.06
    plt.text(0.92,ytext,'start',
         transform=plt.gcf().transFigure,clip_on=False,
         rotation=0,fontsize=fs-4,ha='center', va='center')

    # Add replay numnbers
    #import pdb; pdb.set_trace()
    for step in range(0,last_step+1):
        replay = results['replay_storage'][step]
        axis_index = yis.index(replay[1])
        ax= axes_left[axis_index]
        ax.text(replay[0]+0.3,
                 0-0.32,str(int(step+1)),fontsize=fs-4,color='k')


    # Add title
    plt.sca(axes_left[0])
    plt.title(r'$\alpha_0$='+str(np.round(title_alpha,2)),fontsize=fs)

    if save==True:
        plt.savefig(savefolder+'/iter'+str(step)+'.png',dpi=200,bbox_inches='tight')
        plt.clf()
        plt.close()


def plot_nested_replay(
                    example_name,
                    agent,
                    testcase = 4,
                    subcase='aS',
                    x_eval = 4,
                    y_eval = 1,
                    iters = 60,
                    last_step=1,
                    y_set = [0.  , 0.05, 0.1 , 0.3 , 0.6 , 1.  ],
                    yis = [5,4,3,2,1], # should go in reverse
                    savefolder='prioritized_replay_XY',
                    save=False,
                    extra_loadname=''):

    fs = 18
    # number of planning steps to plot

    f = plt.figure(figsize=(8,8),dpi=200)
    L = len(yis)

    # Set up left and right axes
    axes = []
    for i in range(len(yis)):
        axes.append(plt.axes([0.07*(L), 1-0.15*i, 1,0.20]))

    Qrange = [-10,10]

    term_states_for_plotting = []

    for i,y_ind in enumerate(yis):

        # load data
        savename = '../simulation_results/'+example_name+\
            '_prioritized_VI_nCVaR_ypr'+str(y_ind)+'_xeval'+str(x_eval)+'_iters'+str(iters)+'.pkl'
        with open(savename,'rb') as f:
            results = pickle.load(f)

        x_eval_py = convert_mat_s_to_python(results['x_eval'],agent)

        ## Get Values for Last Step
        V_py = convert_SI_function(results['V_storage'][last_step-1,:,:],agent,neg_term_states=[])
        Q_py = convert_SIA_function(results['Q_storage'][last_step-1,:,:,:],agent,neg_term_states=[])
        Pol_py = convert_SIA_function(results['Pol_storage'][last_step-1,:,:,:],agent,neg_term_states=[])

        ax = axes[i]
        plt.sca(ax)

        y = y_set[y_ind]

        # Values
        v = -1*V_py[:,y_ind].reshape(agent.maze.shape)

        plot_q_or_pi(np.zeros_like(Pol_py),v,'',ax,agent.maze,q_or_pi='q',
                    roundoff=1,inc_triangles=False,tri_add_labels=False,annot_value=False,Qrange=Qrange,
                     term_states_for_plotting=term_states_for_plotting,n_colors=100)

        # Policy as Arrows
        plot_1D_arrows(Pol_py[:,y_ind,:],
                       y_ind,
                       agent.maze,
                       term_states_for_plotting,
                       question=True,
                       box=True)

        # Embellish
        embellish_plot(ax,agent.maze,
            agent,
            None,
            cost=True,
            corner_labels=False,
            color_agent='b',
            center_rewards=False,
            r_fontsize=10,
            add_rewards=True,ec='white',fc='white',alpha=0.0,outer_lw=1,reward_color='k')


        trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
        ann = ax.annotate(str(np.round(y,2)), xy=(-1.26,0.5), xycoords=trans,fontsize=fs)


        if y!=0:
            for step in range(last_step):
                # convert replayed states to python
                state_order_mat = np.arange(agent.num_states).reshape(agent.maze.shape).flatten(order='F')
                replayed_state = state_order_mat[int(results['replay_storage'][step])]
                replayed_state_idx = state2idcs(replayed_state,agent.maze)

                ax.text(replayed_state_idx[1]+0.3,
                        replayed_state_idx[0]-0.32,str(int(step+1)),fontsize=fs-4,color='k')

        arr_img = mpimg.imread('../mscl/lava_pit.png')
        imagebox = OffsetImage(arr_img , zoom=.014)
        ab = AnnotationBbox(imagebox, (0, -.1),pad=0)
        ax.add_artist(ab)

    # y-axis
    x0=0.3; y0=0.32
    x1=0.3; y1=1.12
    plt.arrow(x1,y1,x0-x1,y0-y1,transform=plt.gcf().transFigure,
             color='k',clip_on=False,head_width=.01)
    plt.arrow(x0,y0,x1-x0,y1-y0,transform=plt.gcf().transFigure,
             color='k',clip_on=False,head_width=.01)
    #if static_or_dynamic=='static':
    yaxislabel=r'static risk preference ($\bar{\alpha}$)'
    #elif static_or_dynamic=='dynamic':
    #yaxislabel=r'dynamic risk preference ($y$)'
    plt.text(x0-0.05, # offset text by 0.05 horizontally
             y0+0.2*(y1-y0), # center vertically
                 yaxislabel,
                 transform=plt.gcf().transFigure,clip_on=False,
                 rotation=90,fontsize=fs)

    # x-axis
    x0=0.35; y0=0.2
    x1=1.3; y1=0.2
    xoffset = 0.09
    plt.arrow(x1+xoffset,y1,x0-x1,y0-y1,transform=plt.gcf().transFigure,
             color='k',clip_on=False,head_width=.01)
    plt.arrow(x0+xoffset,y0,x1-x0,y1-y0,transform=plt.gcf().transFigure,
              color='k',clip_on=False,head_width=.01)
    plt.text((x0+x1)/2,
             y0-0.05,r'state ($x$)',
                 transform=plt.gcf().transFigure,clip_on=False,
                 rotation=0,fontsize=fs)
    for i in range(1,10):
        xoffset = 0.11
        plt.text(x0+xoffset*i,y0+0.03,str(i),
                 transform=plt.gcf().transFigure,clip_on=False,
                 rotation=0,fontsize=fs)



    for y_eval in range(0,6):
        # start
        if y_eval==0:
            ytext=0.31
        if y_eval==1:
            ytext=0.46
        if y_eval==2:
            ytext=0.61
        if y_eval==3:
            ytext=0.76
        if y_eval==4:
            ytext=0.91
        if y_eval==5:
            ytext=1.06

        plt.text(0.92,ytext,'start',
             transform=plt.gcf().transFigure,clip_on=False,
             rotation=0,fontsize=fs-4,ha='center', va='center')

    if save==True:
        plt.savefig(savefolder+'/iter'+str(step)+'.png',dpi=200,bbox_inches='tight')
        plt.clf()
        plt.close()
