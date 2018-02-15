#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
    Generate inline animations in Jupyter Notebooks
    
    Author: Moshir Harsh, moshir.harsh@ens.fr
    Date: Feb 12, 2018
    
    TODO:
    Add saving option
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation, rc
from IPython.display import HTML

# initialization function: plot the background of each frame
#def init():
#    line.set_data([], [])
#    return (line,)

# animation function. This is called sequentially
def animate(i,Ydata,Xdata,lines):
    for lnum,line in enumerate(lines):
        line.set_data(Xdata, Ydata[i,lnum]) # set data for each line separately. 
    return lines

def generatePlot(Ydata,Xdata=None,xlim=None,ylim=None,figsize=(8,4),lineWidth=2,frames=None,interval=20,blit=True,multiMode=1):
    
    """
        Generate inline animations in Jupyter notebook
        Ydata    = input Y data where the first dimension i.e rows are the different time frames and the second dimension are the values
        Xdata    = The same as Y but for Xdata. If not specified we take the index number of Y as the X points
        xlim     = (xmin,xmax). If not specified, we take the maximum and the minimum of the given data
        frames   = number of frames to use. Should be less than the size of first dimension of Ydata. If None, use the size of the first                      dimension of Ydata. 
        interval = the time interval between frames
        blit     = True = only replot the parts that have changed and not everything. Much faster.
        multiMdoe= The dimension of the array along which the different lines are present.
    
    """
    num_lines = Ydata.shape[multiMode]
    
    #If the Xdata is not explicitly mentioned, use the index as the Xvalues
    
    if Xdata is None:
        Xdata = np.arange(0,Ydata.shape[multiMode+1])

    if xlim is None:
        xlim = (np.min(Xdata),np.max(Xdata))

    if ylim is None:
        ylim = (np.min(Ydata),np.max(Ydata))
    
    if frames is None:
        frames = Ydata.shape[0]
    
    # First set up the figure, the axis, and the plot element we want to animate
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    line, = ax.plot([],[],lw=lineWidth)

    lines = []
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, num_lines))
        
    for index in range(num_lines):
        lobj = ax.plot([],[],lw=2,color=colors[index])[0]
        lines.append(lobj)        

    anim = animation.FuncAnimation(fig, animate, fargs=(Ydata,Xdata,lines),
                               frames=frames, interval=interval, blit=blit)
    
    
    # equivalent to rcParams['animation.html'] = 'html5'
    rc('animation', html='html5') # the default parameter is set to None, so we won't get any animation.
    
    #Close figure and reset before returning.
    plt.close(anim._fig)
    
    return HTML(anim.to_html5_video())
    