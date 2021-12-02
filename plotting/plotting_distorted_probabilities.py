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
import matplotlib.patches as patches

def plot_distorted_probabilities(P_xy_xpyp_alpha,
                                    Xi_xy_xpyp,
                                    Pol,
                                    agent,
                                    Ns,Ny,Na,
                                    y_set,
                                    yis = [5,4,3,2,1],
                                    maxXi=None
                                 ):

    # create figure
    fig = plt.figure(figsize=(8,8),dpi=500)
    L = len(yis)

    # set up axes
    axes = []
    for i in range(len(yis)):
        axes.append(plt.axes([0.07*(L), 1-0.15*i, 1,0.20]))

    Qrange = [-10,10]
    term_states_for_plotting = [0,9]

    # Loop through y-levels
    for i,yi in enumerate(yis):

        ax = axes[i]
        plt.sca(ax)

        y = y_set[yi]

        # Plot Grid and Colors
        plot_q_or_pi(np.zeros((Ns-1,Na)),
                    np.zeros((1,Ns-1)),
                    '',ax,
                    agent.maze,
                    q_or_pi='pi',
                    roundoff=1,
                    inc_triangles=False,
                    tri_add_labels=True,
                    annot_value=False,
                    Qrange=[0,1],
                    term_states_for_plotting=term_states_for_plotting,
                    pi_color=sns.color_palette()[7])

        embellish_plot(ax,agent.maze,
            agent,None,
            cost=True,
            corner_labels=False,
            color_agent='b',
            center_rewards=True,
            r_fontsize=10,
            add_rewards=True)

        # Policy as Arrow 33$
        plot_1D_arrows(Pol[:,yi,:],
                       yi,
                       agent.maze,
                       term_states_for_plotting,
                       box=True)

        # Plot y along the edge
        trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
        ann = ax.annotate(r'$y$='+str(np.round(y,2)), xy=(-1.7,0.5), xycoords=trans,fontsize=14)

        # Plot Lava Pit
        arr_img = mpimg.imread('../figs/lava_pit.png')
        imagebox = OffsetImage(arr_img , zoom=.014)
        ab = AnnotationBbox(imagebox, (0, .0),pad=0)
        ax.add_artist(ab)


        # Plotting Arrows
        style = "Simple, tail_width=0.5, head_width=4, head_length=8"

        cmap = [plt.get_cmap('brg',100)(i) for i in range(0,40)] # blue purple red colormap

        if maxXi is None:
            maxXi = np.max(Xi_xy_xpyp) # find max weight

        # Draw Arrows for Each State
        for x in range(1,9):

            (xps,yps) = np.where(P_xy_xpyp_alpha[x,i,:,:]!=0)
            ps = P_xy_xpyp_alpha[x,i,xps,yps]
            xis = Xi_xy_xpyp[x,i,xps,yps]


            for xp,yp,p,xi in zip(xps,yps,ps,xis):

                # dealing with y's
                ystart = 0
                if yp==i: # if you stay the same y-level
                    yoffset=0.0
                if np.abs(yp-i)==1: # if you transition 1 y-level
                    yoffset=np.sign((yp-i))*0.45
                if np.abs(yp-i)>=2: # if you transition 2 y-levels
                    yoffset=np.sign((yp-i))*0.45

                if x==xp:
                    # arrow goes left
                    #xstart = x+0.15
                    #xoffset = -0.30
                    #rad = '1.5'

                    # arrow goes right
                    xstart = x-0.13
                    xoffset = +0.27
                    rad = '-1.5'
                    ystart = -0.05
                    yoffset= -0.05  # manually remove...
                    zorder = 0.5
                else:
                    xstart=x
                    xoffset = (xp-x)/2.1
                    rad = '0'
                    zorder = 1

                if xi<maxXi:
                    c = cmap[int(xi*(len(cmap)-1)/maxXi)] # weight * number of colors / max weight
                else:
                    c = cmap[-1]
                kw = dict(arrowstyle=style, color=c)
                patch = patches.FancyArrowPatch((xstart,ystart), # start in center of square
                                        (xstart+xoffset, # point towards difference
                                         ystart+yoffset),
                                        connectionstyle="arc3,rad="+rad,
                                            linestyle='-', linewidth=p*3,**kw,zorder=zorder)
                plt.gca().add_patch(patch)


        x0=0.2; y0=0.45
        x1=0.2; y1=1.12
        plt.arrow(x1,y1,x0-x1,y0-y1,transform=plt.gcf().transFigure,
                 color='k',clip_on=False,head_width=.01)
        plt.arrow(x0,y0,x1-x0,y1-y0,transform=plt.gcf().transFigure,
                 color='k',clip_on=False,head_width=.01)
        plt.text(x0-0.05,y0+0.3*(y1-y0),'Risk Preference (y)',
                     transform=plt.gcf().transFigure,clip_on=False,
                     rotation=90,fontsize=18)

        x0=0.3; y0=0.35
        x1=1.1; y1=0.35
        plt.arrow(x1,y1,x0-x1,y0-y1,transform=plt.gcf().transFigure,
                 color='k',clip_on=False,head_width=.01)
        plt.arrow(x0,y0,x1-x0,y1-y0,transform=plt.gcf().transFigure,
                 color='k',clip_on=False,head_width=.01)
        plt.text(0.6,0.37,'Location (x)',
                     transform=plt.gcf().transFigure,clip_on=False,
                     rotation=0,fontsize=18)

    #plt.show()
    return(fig,axes)
