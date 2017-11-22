# -*- coding: utf-8 -*-
"""
Herramientas para la práctica de Conteo de Fotones
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from instruments.oscilloscope import Oscilloscope

plt.ion()

rsc_osci = 'USB0::1689::867::C102223::0::INSTR'
osci = Oscilloscope(resource=rsc_osci, backend='@py')


def picos(y):
    """ Encuentra los picos y devuelve sus alturas """
    derivada = np.diff(y)
    puntas = y[1:] * (derivada < 0)
    alturas = puntas[puntas < 0]
    return alturas


def medir_pantallas(n_pantallas=1000, umbral=-0.002,
                    n_bins=100, nombre='medicion',
                    bins_range=(-0.035, 0.0)):

    hdf5 = h5py.File('./datos/' + nombre + '.hdf5', 'x')
    f_alturas = hdf5.create_group('alturas')
    f_histogramas = hdf5.create_group('histogramas')
    f_ventanas = hdf5.create_group('ventanas')

    osci.setup_curve('CH2')
    osci.get_waveform_preamble(log=False)

    cantidad = np.zeros(n_pantallas, dtype=int)
    hist_voltajes = np.zeros(n_bins, dtype=int)
    eje_y = np.zeros(2500, dtype=float)
    eje_x = osci.get_x()

    for i in range(n_pantallas):

        print('\r{:05d}/{:05d}'.format(i, n_pantallas), end='')

        # Adquiere el eje Y
        eje_y = osci.get_y()
        # Busca la intensidad del fotón
        alturas = picos(eje_y * (eje_y < umbral))
        # Cuenta la cantidad de fotones
        cantidad[i] = len(alturas)
        # Hace un histograma de alturas con bins fijos
        hist, bin_edges = np.histogram(alturas, bins=n_bins,
                                       range=bins_range)
        # Acumula los histogramas
        hist_voltajes += hist

        # Guarda las alturas por pantalla
        f_alturas['{:05d}'.format(i)] = alturas
        # Guarda los histogramas individules
        f_histogramas['{:05d}'.format(i)] = hist
        # Guarda las pantallas
        f_pantallas['{:05d}'.format(i)] = eje_y

    hdf5['/hist_general'] = hist_voltajes
    hdf5['/eje_x'] = eje_x
    hdf5['/cantidad'] = cantidad

    return cantidad, hist_voltajes, bin_edges, hdf5
