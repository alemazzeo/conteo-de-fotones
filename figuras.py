# -*- coding: utf-8 -*-
"""
Archivo para el análisis de las mediciones
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from adquisicion import detecta_picos
from recupera_datos import ejes_largos
from analisis import histograma, graficar_ventanas, autocorrelacion


# CONFIGURACIONES POR DEFECTO

# Figura (tamaño)
plt.rc('figure', figsize=(8, 6))

# Ticks (tamaño de la fuente)
plt.rc(('xtick', 'ytick'), labelsize=14)

# Bordes de la figura (visibles o no)
plt.rc('axes.spines', left=True, bottom=True, top=False, right=False)

# Leyenda (tamaño de la fuenta y ubicación)
plt.rc('legend', fontsize=14, loc='best')

# Ejes (tamaño de la fuente)
plt.rc('axes', labelsize=15)


def plot_umbral(medicion, ax=None, n_bins=50):
    """ Grafica el historial de una medición """

    x, y = ejes_largos(medicion)
    picos = detecta_picos(y)
    if ax is None:
        fig, ax = plt.subplots(1)

    valores = -np.arange(n_bins) * 0.0008
    ax.hist(y[picos], bins=valores[::-1])


def plot_ejemplo_conteo(puntos=1000, pantalla=50, prob=0.05):
    """ Grafica un ejemplo de conteo e histograma """

    n = int(puntos / pantalla)
    x = np.arange(puntos) + 1
    datos = np.random.random(puntos) > (1.0 - prob)
    picos = datos > 0
    cantidad = np.zeros(n, dtype=int)

    fig = plt.figure(1)
    gs = gridspec.GridSpec(3, 1)

    ax0 = fig.add_subplot(gs[:1, :])
    ax1 = fig.add_subplot(gs[1:, :])

    ax0.plot(x, datos, color='blue')
    ax0.plot(x[picos], datos[picos], ls='', marker='o', mfc='white')
    ax0.set_ylim(0, 1.3)
    ax0.set_xticks([])
    ax0.set_yticks([])

    for i in range(n):
        ax0.axvline(i * pantalla, ls=':')
        cantidad[i] = sum(datos[i * pantalla:(i + 1) * pantalla] > 0)
        ax0.text(x=i * pantalla + pantalla / 2, y=1.1, s=str(cantidad[i]),
                 fontsize=16, horizontalalignment='center')

    ax1.hist(cantidad, bins=np.arange(8), align='left', edgecolor='white',
             linewidth=2.0)

    ax1.set_xticks(np.arange(10)[:-1])


def plot_poisson(ventanas=(500, 1000, 2000, 5000), umbral=-0.005):
    """ Grafica los resultados de bose para las ventanas especificadas """

    fig, ax = graficar_ventanas(medicion='poisson',
                                ventanas=ventanas,
                                umbral=umbral,
                                ajusta_poisson=True,
                                color='blue')
    plt.tight_layout()


def plot_bose(ventanas=(500, 1000, 2000, 5000), umbral=-0.005):
    """ Grafica los resultados de bose para las ventanas especificadas """

    fig, ax = graficar_ventanas(medicion='bose',
                                ventanas=ventanas,
                                umbral=umbral,
                                ajusta_bose=True,
                                color='orange')
    plt.tight_layout()


def plot_segmento(medicion, largos=[1, 500, 1000, 2000, 5000], inicio=0,
                  color='blue', umbral=-0.0015):

    fig, ax = plt.subplots(1)
    x, y = ejes_largos(medicion)
    x, y = x[inicio:inicio + max(largos)], y[inicio:inicio + max(largos)]
    x = x * 1000.0
    x = x - x[0]
    y = y * 1000.0
    puntas = detecta_picos(y, umbral=umbral * 1000)
    ax.axhline(umbral * 1000, ls=':', color='0.5')

    ax.plot(x, y, color=color, lw=0.5)
    ax.plot(x[puntas], y[puntas], ls='', color='k',
            marker='o', mfc=color, mew=1)

    for largo in largos:
        ax.axvline(x[largo - 1], ls='--', color='k')
        ax.set_xticks(x[np.asarray(largos, dtype=int) - 1])
        ax.set_xticklabels(['{:.2f}'.format(x[l - 1]) for l in largos],
                           fontsize=13)
    ax.set_ylim(min(y) * 1.1, 2)
    ax.set_xlabel('Tiempo (ms)')
    ax.set_ylabel('Tensión (mV)')

    return fig, ax


plt.ion()
