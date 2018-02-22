#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
    Calculate the Width (FWHM) of the Peak by multiple methods: 
    1. Fitting spline to the peak and finiding the roots.
    2. Calculating sum of squared weights by sum of weights squared. 
    
    Author: Moshir Harsh, moshir.harsh@ens.fr
    Date: Feb 19, 2018
    
    TODO:
    use fit peak function
    ADD AVERAGING OVER LAST TIME POINTS
    
"""

from scipy.interpolate import UnivariateSpline
import pylab as pl
import numpy as np

def peakWidth(weight,plot=False,method='Spline'):
    
    """
        Takes in a single peak channel in weight and calculates the width as a function of time. if Plot is true, it also returns the plot of the last time point, displaying the roots.
        
    """
    if method is 'Spline':
        weightsAligned = []
        spline = []
        roots  = []
        FWHM   = []

        try:
            weight.shape[1]

            for i in range(weight.shape[0]):
                #Roll the weight matrix to put it in the centre
                roll_val = -(np.argmax(abs(weight[i])) - int(len(weight[i])/2))
                weightsAligned.append(np.roll(weight[i],roll_val))
                spline.append(UnivariateSpline(np.arange(len(weight[i])),abs(weightsAligned[i])-np.max(abs(weightsAligned[i]))/2,s=0))
                roots.append(spline[i].roots())
                FWHM.append(abs(roots[i][0]-roots[i][1]))

            if plot is True:
                #Show the roots for visual verification
                pl.plot(np.arange(len(weight[-1])),weightsAligned[-1])
                pl.axvspan(roots[-1][0], roots[-1][1], facecolor='g', alpha=0.5)
                pl.show()

            return np.array(FWHM)

        except:
            weight = np.array([weight])

            for i in range(weight.shape[0]):
                #Roll the weight matrix to put it in the centre
                roll_val = -(np.argmax(abs(weight[i])) - int(len(weight[i])/2))
                weightsAligned.append(np.roll(weight[i],roll_val))
                spline.append(UnivariateSpline(np.arange(len(weight[i])),abs(weightsAligned[i])-np.max(abs(weightsAligned[i]))/2,s=0))
                roots.append(spline[i].roots())
                FWHM = (abs(roots[i][0]-roots[i][1]))

            if plot is True:
                #Show the roots for visual verification
                pl.plot(np.arange(len(weight[-1])),weightsAligned[-1])
                pl.axvspan(roots[-1][0], roots[-1][1], facecolor='g', alpha=0.5)
                pl.show()

            return FWHM
        
    else if method is 'Squared':
            
        FWHM   = []

        try:
            weight.shape[1]

            for i in range(weight.shape[0]):
                sq[i] = np.sum(weight[i]**2)/(np.sum(weight[i])**2)
                FWHM.append(1./sq)

            if plot is True:
                #Show the roots for visual verification
                pl.plot(np.arange(len(weight[-1])),weight[-1])
                pl.axvspan(np.argmax(abs(weight[-1]))-FWHM[-1]/2,np.argmax(abs(weight[-1]))+FWHM[-1]/2, facecolor='g', alpha=0.5)
                pl.show()

            return np.array(FWHM)

        except:
            weight = np.array([weight])

            for i in range(weight.shape[0]):
                sq[i] = np.sum(weight[i]**2)/(np.sum(weight[i])**2)
                FWHM.append(1./sq)

            if plot is True:
                #Show the roots for visual verification
                pl.plot(np.arange(len(weight[-1])),weight[-1])
                pl.axvspan(np.argmax(abs(weight[-1]))-FWHM[-1]/2,np.argmax(abs(weight[-1]))+FWHM[-1]/2, facecolor='g', alpha=0.5)
                pl.show()


            return FWHM
