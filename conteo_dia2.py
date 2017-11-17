# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import numpy as np
import matplotlib.pyplot as plt
from oscilloscope import Oscilloscope

osci = Oscilloscope(resource='USB0::0x0699::0x0363::C108011::INSTR', 
                    backend='')


def pantallas(n=1500, umbral=-0.08):

    osci.setup_curve('CH2')
    osci.get_waveform_preamble()
    
    fotones = np.zeros(n, dtype=int)
    x = np.zeros(2500, dtype=float)
    y = np.zeros(2500, dtype=float)
    
    for i in range(n):
        x, y = osci.get_curve(auto_wfmpre=False)
        y = y * (y < umbral)
        dy = np.diff(y)
        yp = y[1:] * (dy<0)
        fotones[i] = len(dy[dy<0])
        
    return fotones


def threshold(n=1000, umbral= -0.08, bins=100, 
              bins_range=(-2.0, 0.0), 
              plot=False, ax=None):
    osci.setup_curve('CH2')
    osci.get_waveform_preamble()
    
    cuentas = np.zeros(bins, dtype=int)
    bin_edges = np.zeros(bins+1, dtype=float)
    x = np.zeros(2500, dtype=float)
    y = np.zeros(2500, dtype=float)
    
    for i in range(n):
        x, y = osci.get_curve(auto_wfmpre=False)
        y = y * (y < umbral)
        dy = np.diff(y)
        yp = y[1:] * (dy<0)
        hist, bin_edges = np.histogram(yp[yp<0], bins=bins, range=bins_range)
        cuentas += hist
        
    #width = 1.5 * (bin_edges[1] - bin_edges[0])
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.bar(centers, cuentas, align='center', width=0.016)
        
    return cuentas, centers

    