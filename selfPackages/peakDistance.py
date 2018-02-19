#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
    Calculate interpeak distance in a circular 1D system
    
    Author: Moshir Harsh, moshir.harsh@ens.fr
    Date: Feb 19, 2018
    
    TODO:
    Support more than 1D
"""

def peakDistance(pos1,pos2,systemSize):
    if abs(pos1-pos2) > systemSize/2:
        return pos1+pos2-systemSize
    else:
        return abs(pos1-pos2)