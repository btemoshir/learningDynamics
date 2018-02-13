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
def animate(i,Ydata,Xdata,line):
    x = Xdata
    y = Ydata[i]
    line.set_data(x, y)
    return (line,)

def generatePlot(Ydata,Xdata=None,xlim=None,ylim=None,figsize=(8,4),lineWidth=2,frames=100,interval=20,blit=True):
    """
        Generate inline animations in Jupyter notebook
        Ydata    = input Y data where the first dimension i.e rows are the different time frames and the second dimension are the values
        Xdata    = The same as Y but for Xdata. If not specified we take the index number of Y as the X points
        xlim     = (xmin,xmax). If not specified, we take the maximum and the minimum of the given data
        frames   = number of frames to use. Should be less than the size of second dimension of Ydata
        interval = the time interval between frames
        blit     =  True = only replot the parts that have changed and not everything. Much faster.
    """
    #If the Xdata is not explicitly mentioned, use the index as the Xvalues
    if Xdata is None:
        Xdata = np.arange(0,Ydata.shape[1])
        
    #If the xlim and ylim are not specified, take the max and the minimum of x and y to be the limits
    if xlim is None:
        xlim = (np.min(Xdata),np.max(Xdata))
        
    if ylim is None:
        ylim = (np.min(Ydata),np.max(Ydata))
    
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    line, = ax.plot([], [], lw=lineWidth)
    
    anim = animation.FuncAnimation(fig, animate, fargs=(Ydata,Xdata,line),
                                   frames=frames, interval=interval, blit=blit)
    
    # equivalent to rcParams['animation.html'] = 'html5'
    rc('animation', html='html5') # the default parameter is set to None, so we won't get any animation.
    
    #Close figure and reset before returning.
    plt.close(anim._fig)
    
    return HTML(anim.to_html5_video())
    