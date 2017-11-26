# -*- coding: utf-8 -*-
"""
Archivo para el análisis de las mediciones
"""

import numpy as np
import matplotlib.pyplot as plt

from adquisicion import detecta_picos
from recupera_datos import recorrer_ejes, ejes_largos


def histograma(medicion, tamaño_ventana=2500, umbral=0.0,
               bins=10, ax=None, **kwargs):
    """ Devuelve el histograma """

    x, y = ejes_largos(medicion)
    n_ventanas = len(y) // tamaño_ventana
    duracion = (x[0] - x[tamaño_ventana]) * tamaño_ventana * 1000

    cantidad = np.zeros(n_ventanas, dtype=int)

    for i in range(n_ventanas):

        ventana = y[i * tamaño_ventana:(i + 1) * tamaño_ventana]
        picos = detecta_picos(ventana, umbral=0.0)
        alturas = ventana[picos]
        cantidad[i] = sum(alturas < umbral)

    if ax is None:
        fig, ax = plt.subplots(1)

    hist_y, hist_x = np.histogram(cantidad, bins=bins)
    hist_x += ((hist_x[1] - hist_x[0]) / 2)
    hist_x = hist_x[0:-1]

    ax.hist(cantidad,
            bins=bins,
            label=r'$tc={:.2f}\;ms$'.format(duracion),
            **kwargs)

    return hist_x, hist_y


plt.ion()

fig, ax1 = plt.subplots(4)

histograma('poisson', tamaño_ventana=500, umbral=-0.001, ax=ax1[0])
histograma('poisson', tamaño_ventana=1000, umbral=-0.001, ax=ax1[1])
histograma('poisson', tamaño_ventana=2000, umbral=-0.001, ax=ax1[2])
histograma('poisson', tamaño_ventana=5000, umbral=-0.001, ax=ax1[3])

ax1[0].set_title('Poisson')
ax1[0].legend()
ax1[1].legend()
ax1[2].legend()
ax1[3].legend()

fig, ax2 = plt.subplots(4)

histograma('bose', tamaño_ventana=500, umbral=-0.001, ax=ax2[0])
histograma('bose', tamaño_ventana=1000, umbral=-0.001, ax=ax2[1])
histograma('bose', tamaño_ventana=2000, umbral=-0.001, ax=ax2[2])
histograma('bose', tamaño_ventana=5000, umbral=-0.001, ax=ax2[3])

ax2[0].set_title('Bose')
ax2[0].legend()
ax2[1].legend()
ax2[2].legend()
ax2[3].legend()
