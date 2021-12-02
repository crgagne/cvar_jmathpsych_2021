import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib as mpl
from scenarios import state2idcs

def plot_2D_arrows(Pol,alpha_ind,maze,term_states_for_plotting,question=False,box=True):
    arrow_markers = np.array(['$\u2191$','$\u2193$','$\u2192$','$\u2190$'])
    for s in range(Pol.shape[0]):
        if s not in term_states_for_plotting:
            markers = arrow_markers[np.where(Pol[s,:]==np.max(Pol[s,:]))[0]]
            s_idcs = state2idcs(s,maze)

            # multiple directions
            if len(markers)>1:


                # for 0.33 different arrows use a ?
                if np.all(Pol[s,:]==1/len(Pol[s,:])):
                    xoffset = 0
                    yoffset = 0
                    if question:
                        plt.plot(s_idcs[1]+xoffset,s_idcs[0]+yoffset
                                 ,marker='$\u003F$',color='k',alpha=0.1,
                                 linewidth=0.25,linestyle='--',ms=10)

                # for multiple arrows
                else:
                    #print(markers)
                    for marker in markers:
                        yoffset=0
                        xoffset=0
                        if marker==arrow_markers[0]:
                            yoffset = -0.1
                            xoffset = -0.015
                        elif marker==arrow_markers[1]:
                            yoffset = 0.1
                            xoffset = -0.015
                        elif marker==arrow_markers[2]:
                            xoffset= 0.11
                        elif marker==arrow_markers[3]:
                            xoffset= -0.11 # -0.08
                        plt.plot(s_idcs[1]+xoffset,s_idcs[0]+yoffset,marker=marker,color='k',linewidth=0.25,linestyle='--',ms=10)

                    if box:
                        rotate = 0.0
                        if len(markers)==4:
                                xcenter = s_idcs[1]
                                ycenter = s_idcs[0]+0.09
                                xleft = xcenter-0.22
                                xright = xcenter+0.22
                                ytop = ycenter-0.18
                                ybottom = ycenter+0.12
                                ytop2 = ytop-0.12

                                xy = [[xleft,ytop],
                                      [xcenter+(xleft-xcenter)*2/5,ytop], #
                                      [xcenter+(xleft-xcenter)*2/5,ytop2],
                                      [xcenter+(xright-xcenter)*2/5,ytop2],
                                      [xcenter+(xright-xcenter)*2/5,ytop],
                                      [xright,ytop], # top
                                      [xright,ycenter],
                                      [xcenter+(xright-xcenter)*2/5,ycenter], #
                                      [xcenter+(xright-xcenter)*2/5,ybottom],
                                      [xcenter+(xleft-xcenter)*2/5,ybottom],
                                      [xcenter+(xleft-xcenter)*2/5,ycenter],

                                      [xleft,ycenter]
                                     ]


                        if len(markers)==3:
                            # left, right, up
                            if np.all(markers == np.array(['$\u2191$','$\u2192$','$\u2190$'])):
                                xcenter = s_idcs[1]
                                ycenter = s_idcs[0]+0.09
                                xleft = xcenter-0.22
                                xright = xcenter+0.22
                                ytop = ycenter-0.18
                                ybottom = ycenter+0.12
                                ytop2 = ytop-0.12

                                xy = [[xleft,ytop],
                                      [xcenter+(xleft-xcenter)*2/5,ytop], #
                                      [xcenter+(xleft-xcenter)*2/5,ytop2],
                                      [xcenter+(xright-xcenter)*2/5,ytop2],
                                      [xcenter+(xright-xcenter)*2/5,ytop],
                                      [xright,ytop], # top
                                      [xright,ycenter],
                                      [xleft,ycenter]
                                     ]

                            # up, down, left
                            if np.all(markers == np.array(['$\u2191$','$\u2193$','$\u2190$'])):
                                xcenter = s_idcs[1]-0.07
                                ycenter = s_idcs[0]+0.0
                                xleft = xcenter-0.18
                                xright = xcenter+0.15
                                ytop = ycenter-0.22
                                ybottom = ycenter+0.22
                                xy = [[xleft,ycenter+(ytop-ycenter)*2/5],
                                      [xcenter,ycenter+(ytop-ycenter)*2/5],
                                      [xcenter,ytop],
                                      [xright,ytop],
                                      [xright,ybottom],
                                      [xcenter,ybottom],
                                      [xcenter,ycenter+(ybottom-ycenter)*2/5],
                                      [xleft,ycenter+(ybottom-ycenter)*2/5]
                                ]
                            # up, down, right
                            if np.all(markers == np.array(['$\u2191$','$\u2193$','$\u2192$'])):
                                xcenter = s_idcs[1]+0.07
                                ycenter = s_idcs[0]+0.0
                                xleft = xcenter+0.18
                                xright = xcenter-0.15
                                ytop = ycenter-0.22
                                ybottom = ycenter+0.22
                                xy = [[xleft,ycenter+(ytop-ycenter)*2/5],
                                      [xcenter,ycenter+(ytop-ycenter)*2/5],
                                      [xcenter,ytop],
                                      [xright,ytop],
                                      [xright,ybottom],
                                      [xcenter,ybottom],
                                      [xcenter,ycenter+(ybottom-ycenter)*2/5],
                                      [xleft,ycenter+(ybottom-ycenter)*2/5]
                                ]



                        elif len(markers)==2:
                            # up right
                            if np.all(markers == np.array(['$\u2191$','$\u2192$'])):
                                xcenter = s_idcs[1]+0.12
                                ycenter = s_idcs[0]-0.1
                                xleft = xcenter-0.18
                                xright = xcenter+0.12
                                ytop = ycenter-0.12
                                ybottom = ycenter+0.18
                                xy = [[xleft,ytop],
                                      [xcenter,ytop], # top
                                      [xcenter,ycenter],
                                      [xright,ycenter], #
                                      [xright,ybottom], # right leg
                                      [xleft,ybottom],
                                     ]
                            # up left
                            elif np.all(markers == np.array(['$\u2191$','$\u2190$'])):
                                xcenter = s_idcs[1]+0.12
                                ycenter = s_idcs[0]-0.1
                                xleft = xcenter-0.18
                                xright = xcenter+0.12
                                ytop = ycenter-0.12
                                ybottom = ycenter+0.18
                                xy = [[xleft,ytop],
                                      [xcenter,ytop], # top
                                      [xcenter,ycenter],
                                      [xright,ycenter], #
                                      [xright,ybottom], # right leg
                                      [xleft,ybottom],
                                     ]
                            # up down
                            elif np.all(markers == np.array(['$\u2191$','$\u2193$'])):
                                xcenter = s_idcs[1]+0.12
                                ycenter = s_idcs[0]-0.1
                                xleft = xcenter-0.18
                                xright = xcenter+0.12
                                ytop = ycenter-0.12
                                ybottom = ycenter+0.18
                                xy = [[xleft,ytop],
                                      [xcenter,ytop], # top
                                      [xcenter,ycenter],
                                      [xright,ycenter], #
                                      [xright,ybottom], # right leg
                                      [xleft,ybottom],
                                     ]
                            # left right
                            elif np.all(markers == np.array(['$\u2192$','$\u2190$'])):
                                xcenter = s_idcs[1]+0.12
                                ycenter = s_idcs[0]-0.1
                                xleft = xcenter-0.18
                                xright = xcenter+0.12
                                ytop = ycenter-0.12
                                ybottom = ycenter+0.18
                                xy = [[xleft,ytop],
                                      [xcenter,ytop], # top
                                      [xcenter,ycenter],
                                      [xright,ycenter], #
                                      [xright,ybottom], # right leg
                                      [xleft,ybottom],
                                     ]

                        xy =np.array(xy)
                        ax = plt.gca()
                        poly = patches.Polygon(
                            xy=xy,  # point of origin.
                            linewidth=1,
                            facecolor='white',
                            edgecolor='k',
                            fill=True,
                            zorder=2,
                            closed=True,
                        )
                        ax.add_patch(poly)


            # for single arrow
            else:
                #print('single')
                marker = markers[0]
                plt.plot(s_idcs[1],s_idcs[0],marker=marker,color='k',linewidth=0.25,linestyle='--',ms=15)

                if box:
                    if marker=='$\u2191$' or marker=='$\u2193$':
                        xy=(s_idcs[1]-0.05,s_idcs[0]-0.16)
                        width = 0.22
                        height = .34
                    else:
                        xy=(s_idcs[1]-0.16,s_idcs[0]-0.12)
                        width = 0.34
                        height = .22
                    ax = plt.gca()
                    ax.add_patch(
                        patches.Rectangle(
                            xy=xy,  # point of origin.
                            width=width,#np.sqrt(mult), #.25,
                            height=height,#np.sqrt(mult),
                            linewidth=1,
                            facecolor='white',
                            edgecolor='k',
                            fill=True,
                            zorder=2,
                        )
                    )
