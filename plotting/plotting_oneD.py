import numpy as np
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import Image,display
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap

from scenarios import state2idcs
from plotting_basics import plot_q_or_pi, embellish_plot


def plot_1D_arrows(Pol,alpha_ind,maze,term_states_for_plotting,
                    question=False,box=False,box_size_adjustement=False,box_ec='k',jointbox=False,staybox='bottom'):

    #if staybox=='bottom':
    arrow_markers = np.array(['$\u21BB$','$\u2192$','$\u2190$'])
    #else:
    #    arrow_markers = np.array(['$\u21BA$','$\u2192$','$\u2190$'])

    # loop over states
    for s in range(Pol.shape[0]):
        if s not in term_states_for_plotting:
            #print(s)

            s_idcs = state2idcs(s,maze)
            pi = Pol[s,:] # single state policy

            if box_size_adjustement:
                markers = arrow_markers[np.where(pi!=0)[0]] # select all markers
                mults = np.ones(len(markers))*pi*1.2+.4 # make multiplier relative to random.
            else:
                markers = arrow_markers[np.where(pi==np.max(pi))[0]] # allows for ties.
                mults=np.ones(len(markers))

            # multiple directions
            if len(markers)>1:

                # replay plots with question mark
                if np.all(pi==1/len(pi)) and question:
                    xoffset = 0
                    yoffset = 0
                    plt.plot(s_idcs[1]+xoffset,s_idcs[0]+yoffset
                             ,marker='$\u003F$',color='k',alpha=0.1,
                             linewidth=0.25,linestyle='--',ms=10)

                # for 0.5 use 2 arrows
                else:

                    if jointbox:
                        box=False

                    for marker,mult in zip(markers,mults):
                        yoffset=0
                        xoffset=0
                        if marker==arrow_markers[0]:
                            if staybox=='bottom':
                                yoffset = 0.21 #-0.08
                                angle = 0
                            elif staybox=='top':
                                yoffset = -0.2
                                angle=195
                        elif marker==arrow_markers[1]:
                            xoffset = 0.14#+0.07
                            angle=0
                        elif marker==arrow_markers[2]:
                            xoffset=-0.14#-0.07
                            angle=0


                        # plt.plot(s_idcs[1]+xoffset,s_idcs[0]+yoffset,marker=marker,
                        #     color='k',linewidth=0.25,linestyle='--',ms=10*np.sqrt(mult))
                        #xoffset = xoffset*np.sqrt(mult)
                        #yoffset = yoffset*np.sqrt(mult)
                        t = mpl.markers.MarkerStyle(marker=marker)
                        t._transform = t.get_transform().rotate_deg(angle)
                        plt.scatter(s_idcs[1]+xoffset,s_idcs[0]+yoffset,marker=t,s=100,color='k',zorder=10)#,ms=10*np.sqrt(mult))


                        if box:
                            ax = plt.gca()
                            ax.add_patch(
                                patches.Rectangle(
                                    xy=(s_idcs[1]+xoffset-0.12,s_idcs[0]+yoffset-0.1),  # point of origin.
                                    width=0.27*np.sqrt(mult), #.25,
                                    height=.2*np.sqrt(mult),
                                    linewidth=1,
                                    facecolor='white',
                                    edgecolor=box_ec,
                                    fill=True,
                                    zorder=2,
                                )
                            )

                    # Add box around all 3
                    if jointbox ==True:

                        # stay box on bottom
                        if staybox=='bottom':
                            xcenter = s_idcs[1]
                            ycenter = s_idcs[0]+0.1
                            xleft = xcenter-0.28
                            xright = xcenter+0.28
                            ytop = ycenter-0.2
                            ybottom = ycenter+0.2
                        elif staybox=='top':
                            xcenter = s_idcs[1]
                            ycenter = s_idcs[0]-0.1
                            xleft = xcenter-0.28
                            xright = xcenter+0.28
                            ytop = ycenter+0.2
                            ybottom = ycenter-0.2

                        #print(xleft,xcenter,xright)
                        #print(ybottom,ycenter,ytop)
                        xy = [[xleft,ytop],
                              [xright,ytop], # top
                              [xright,ycenter],
                              [xcenter+(xright-xcenter)*2/5,ycenter], #
                              [xcenter+(xright-xcenter)*2/5,ybottom], # right leg
                              [xcenter+(xleft-xcenter)*2/5,ybottom],
                              [xcenter+(xleft-xcenter)*2/5,ycenter],
                              [xleft,ycenter]
                             ]
                        xy =np.array(xy)
                        ax = plt.gca()
                        ax.add_patch(
                                    patches.Polygon(
                                        xy=xy,  # point of origin.
                                        linewidth=1,
                                        facecolor='white',
                                        edgecolor='k',
                                        fill=True,
                                        zorder=2,
                                        closed=True,
                                    )
                        )

            # single direction
            else:
                marker = markers[0]
                plt.plot(s_idcs[1],s_idcs[0],marker=marker,color='k',linewidth=0.25,linestyle='--',ms=15)
                if box:
                        ax = plt.gca()
                        ax.add_patch(
                            patches.Rectangle(
                                xy=(s_idcs[1]-0.13,s_idcs[0]-0.1),  # point of origin.
                                width=.3,
                                height=.2,
                                linewidth=1,
                                facecolor='white',
                                edgecolor=box_ec,
                                fill=True,
                                zorder=2,
                            )
                        )


        else:
            pass


def plot_1D_values_policy_distortions(Pol,
                                      V_py,
                                      agent,
                                      P_xy_xpyp_alpha,
                                      Xi_xy_xpyp,
                                      static_or_dynamic='dynamic',
                                      eval_or_opt='eval',
                                      y_set = [0,0.05,0.1,0.3,0.6,1],
                                      yis = [5,4,3,2,1,0],
                                      Qrange = [-10,10],
                                      term_states_for_plotting = [0,9],
                                      example_start=None,
                                      staybox='bottom',
                                      no_vert_space=False):

    plt.figure(figsize=(8,8),dpi=500)
    fs = 18

    L = len(yis)

    axes = []
    for i in range(len(yis)):
        if no_vert_space:
            axes.append(plt.axes([0.07*(L), 1-0.111*i, 1,0.20])) # higher number pushes them further apart; .111 is good
        else:
            axes.append(plt.axes([0.07*(L), 1-0.15*i, 1,0.20]))

    for i,yi in enumerate(yis):

        ax = axes[i]
        plt.sca(ax)

        y = y_set[yi]

        plot_q_or_pi(np.zeros_like(Pol),
                    -1*V_py[:,yi].reshape(agent.maze.shape),
                    '',
                    ax,
                    agent.maze,
                    q_or_pi='q',
                    roundoff=1,
                    inc_triangles=False,
                    tri_add_labels=False,
                    annot_value=False,
                    Qrange=Qrange,
                    term_states_for_plotting=term_states_for_plotting)

        embellish_plot(ax,agent.maze,
            agent,None,
            cost=True,
            corner_labels=False,
            color_agent='b',
            center_rewards=False,
            r_fontsize=10,
            add_rewards=True,ec='white',fc='white',alpha=0.0,outer_lw=1,reward_color='k')

        # Policy as Arrow 33$
        if eval_or_opt=='eval':
            plot_1D_arrows(Pol[:,yi,:],
                           yi,
                           agent.maze,
                           term_states_for_plotting,
                           box=False,box_ec=(0.9,0.9,0.9,1),jointbox=True,staybox=staybox)
        elif eval_or_opt=='opt':
            plot_1D_arrows(Pol[:,yi,:],
               yi,
               agent.maze,
               term_states_for_plotting,
               box=True,
               box_size_adjustement=False,
               jointbox=True,
               staybox=staybox)

        # Plot y along the edge
        trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
        ann = ax.annotate(str(np.round(y,2)), xy=(-1.26,0.5), xycoords=trans,fontsize=fs)

        # Plot Lava Pit
        arr_img = mpimg.imread('../mscl/lava_pit.png')
        imagebox = OffsetImage(arr_img , zoom=.014)
        ab = AnnotationBbox(imagebox, (0, .0),pad=0)
        ax.add_artist(ab)

        # Plotting Arrows
        style = "Simple, tail_width=0.5, head_width=4, head_length=8"

        basic_cols=[(0.0, 0.0, 0.8, 1.0), (0.3, 0.3, 0.3, 1),(0.8, 0.0, 0.8, 1.0)]
        cmap_bkr=LinearSegmentedColormap.from_list('bkr', basic_cols,100)
        cmap_neg = [cmap_bkr(i) for i in range(100)][0:50][::-1]
        cmap_pos = [cmap_bkr(i) for i in range(100)][50::]
        cmap_neg = cmap_pos

        # Find max log weight for coloring; not used anymore
        xistmp = Xi_xy_xpyp.flatten()
        xistmp = xistmp[~np.isnan(xistmp)]
        xistmp = xistmp[xistmp!=0]
        xistmp_log = np.log(xistmp)
        maxXi_log = np.max(xistmp_log)
        minXi_log = np.min(xistmp_log)

        # for each physical location (and each risk preference y; looped through above)
        for x in range(1,9):

            # get probs and weights for next states and risk preferences x',y'
            (xps,yps) = np.where(P_xy_xpyp_alpha[x,i,:,:]!=0)
            ps = P_xy_xpyp_alpha[x,i,xps,yps]
            xis = Xi_xy_xpyp[x,i,xps,yps]

            # loop through next states and plot
            for xp,yp,p,xi in zip(xps,yps,ps,xis):

                # dealing with y's
                ystart = 0
                if yp==i: # if you stay the same y-level
                    yoffset=0.0
                if np.abs(yp-i)==1: # if you transition 1 y-level
                    yoffset=np.sign((yp-i))*0.45
                if np.abs(yp-i)>=2: # if you transition 2 y-levels
                    yoffset=np.sign((yp-i))*0.45

                # deal with x's
                if x==xp: # if in the same location (self arrows)
                    if staybox=='bottom':
                        if yp==i:  # staying in the same y-level
                            xstart = x-0.13
                            xoffset = +0.27
                            rad = '-1.5'
                            ystart = -0.05
                            yoffset= -0.05
                            zorder = 0.5
                        elif yp>i: # moving up in y-level
                            xstart = x
                            xoffset = 0
                            rad = '0'
                            ystart = +0.05
                            yoffset= +0.45
                            zorder = 0.5
                        elif yp<i: # moving down in y-level
                            xstart = x
                            xoffset = 0
                            rad = '0'
                            ystart = -0.05
                            yoffset= -0.45
                            zorder = 0.5
                    elif staybox=='top':
                        if yp==i:
                            xstart = x+0.13
                            xoffset = -0.27
                            rad = '-1.5'
                            ystart = +0.05
                            yoffset= +0.05
                            zorder = 0.5
                        elif yp>i:
                            xstart = x
                            xoffset = 0
                            rad = '0'
                            ystart = +0.05
                            yoffset= +0.4
                            zorder = 0.5
                        elif yp<i:
                            import pdb; pdb.set_trace()
                            xstart = x
                            xoffset = 0
                            rad = '0'
                            ystart = -0.05
                            yoffset= -0.3
                            zorder = 0.5
                else: # non-self location arrows
                    xstart=x
                    xoffset = (xp-x)/2.1
                    rad = '0'
                    zorder = 1

                # get color range for arrow
                rangeXi_log = -minXi_log+maxXi_log

                # set color for arrow
                xi_log = np.log(xi)
                if xi_log>0:
                    c = cmap_pos[int(xi_log/maxXi_log*(len(cmap_pos)-1))]
                elif xi_log<0:
                    c = cmap_neg[int(xi_log/minXi_log*(len(cmap_neg)-1))]
                elif xi==0:
                    c = 'k'

                # plot distortion weight arrow
                kw = dict(arrowstyle=style, color=(0.3,0.3,0.3,1)) #P_xy_xpyp_alpha[x,i,xps,yps]
                patch = patches.FancyArrowPatch((xstart,ystart), # start in center of square
                                        (xstart+xoffset, # point towards difference
                                         ystart+yoffset),
                                        connectionstyle="arc3,rad="+rad,
                                            linestyle='-', linewidth=p*4, # thickness is 4 times the probability
                                            **kw,zorder=zorder)
                plt.gca().add_patch(patch)

    # y-axis
    if no_vert_space:
        x0=0.3; y0=0.48
        x1=0.3; y1=1.12
    else:
        x0=0.3; y0=0.32
        x1=0.3; y1=1.12
    plt.arrow(x1,y1,x0-x1,y0-y1,transform=plt.gcf().transFigure,
             color='k',clip_on=False,head_width=.01)
    plt.arrow(x0,y0,x1-x0,y1-y0,transform=plt.gcf().transFigure,
             color='k',clip_on=False,head_width=.01)
    if static_or_dynamic=='static':
        yaxislabel=r'static risk preference ($\bar{\alpha}$)'
    elif static_or_dynamic=='dynamic':
        yaxislabel=r'dynamic risk preference ($\alpha$)'
    plt.text(x0-0.05, # offset text by 0.05 horizontally
             y0+0.2*(y1-y0), # center vertically
                 yaxislabel,
                 transform=plt.gcf().transFigure,clip_on=False,
                 rotation=90,fontsize=fs)

    # x-axis
    if no_vert_space:
        x0=0.35; y0=0.40
        x1=1.3; y1=0.40
    else:
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

    if example_start=='opt':
        # not using anymore
        if no_vert_space:
            plt.text(0.92,0.83,'example \n start',
                 transform=plt.gcf().transFigure,clip_on=False,
                 rotation=0,fontsize=10,ha='center', va='center',color='b')
            plt.text(0.95,0.92,'t=0',
                 transform=plt.gcf().transFigure,clip_on=False,
                 rotation=0,fontsize=10,ha='center', va='center',color='b')
            plt.text(0.85,0.98,'t=1',
                 transform=plt.gcf().transFigure,clip_on=False,
                 rotation=0,fontsize=10,ha='center', va='center',color='b')

        else:
            plt.text(0.92,0.765,'example \n start',
                 transform=plt.gcf().transFigure,clip_on=False,
                 rotation=0,fontsize=10,ha='center', va='center',color='b')

    #plt.text(.5,1.2,r'$r(x)$=-10',fontsize=fs,zorder=3,va='center',ha='center',color='r',transform=plt.gcf().transFigure)
    #plt.text(.8,1.2,'$r(x)$=2',fontsize=fs,zorder=3,va='center',ha='center',color='g',transform=plt.gcf().transFigure)
    #plt.text(1.2,1.2,'$r(x)$=1',fontsize=fs,zorder=3,va='center',ha='center',color='g',transform=plt.gcf().transFigure)

    plt.show()
