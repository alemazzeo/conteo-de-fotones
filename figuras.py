# -*- coding: utf-8 -*-
"""
Archivo para el análisis de las mediciones
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

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


def plot_umbral(medicion, ax=None, n_bins=20, color='blue',
                ylim=1000, umbral=-2.4):
    """ Grafica el historial de una medición """

    x, y = ejes_largos(medicion)
    y = y * 1000
    picos = detecta_picos(y)
    if ax is None:
        fig, ax = plt.subplots(1)

    valores = -np.arange(n_bins) * 0.8
    valores = valores[::-1]
    ax.hist(y[picos], bins=valores, align='right',
            edgecolor='white', linewidth=2.0, color=color)
    ax.set_xticks(valores[::1])

    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    ax.axvline(umbral, ls='--')

    ax.set_ylim(0, ylim)
    ax.yaxis.grid(ls=':')
    ax.set_xlabel('Tensión ($mV$)')
    plt.tight_layout()


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


def plot_bose_con_poisson(ventanas=(5000, 5000, ), umbral=-0.005):
    """ Grafica los resultados de bose para las ventanas especificadas """

    fig, ax = graficar_ventanas(medicion='bose',
                                ventanas=ventanas,
                                umbral=umbral,
                                ajusta_bose=True,
                                color='orange')
    plt.tight_layout()


def plot_segmento(medicion, largos=[1, 500, 1000, 2000, 5000], inicio=0, color='blue',
                  umbral=-0.0024, mult=1e3, unidad='ms', f=2, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1)
    x, y = ejes_largos(medicion)
    x, y = x[inicio:inicio + max(largos)], y[inicio:inicio + max(largos)]
    x = x * mult
    x = x - x[0]
    y = y * 1000
    puntas = detecta_picos(y, umbral=umbral * 1000)
    ax.axhline(umbral * 1000, ls=':', color='0.5')

    ax.plot(x, y, color=color, lw=0.5)
    ax.plot(x[puntas], y[puntas], ls='', color='k',
            marker='o', mfc=color, mew=1)

    for largo in largos:
        ax.axvline(x[largo - 1], ls='--', color='k')
        ax.set_xticks(x[np.asarray(largos, dtype=int) - 1])
        ax.set_xticklabels(['{:.{f}f}'.format(x[l - 1], f=f) for l in largos],
                           fontsize=13)
    ax.set_ylim(min(y) * 1.1, 2)
    ax.set_xlabel(r'Tiempo (${}$)'.format(unidad))
    ax.set_ylabel('Tensión ($mV$)')


plt.ion()


def plot_segmento_acorr(medicion, largo=200, inicio=0, color='blue', ts=0.5,
                        umbral=-0.0024, mult=1e3, unidad='ms', ax=None,
                        figsize=(5, 3), label=''):

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)

    x, y = ejes_largos(medicion)
    x, y = x[inicio:inicio + largo], y[inicio:inicio + largo]
    x = x * mult
    x = x - x[0]
    y = y * 1000

    ax.plot(x, y, color=color, lw=1.5, label=label)

    ax.set_ylim(min(y) * 1.1, 2)
    ax.set_xlabel(r'Tiempo (${}$)'.format(unidad))
    # ax.set_ylabel('Tensión ($mV$)')
    ax.set_xticks(np.arange(min(x), max(x), ts))
    ax.set_yticks([])
    ax.xaxis.grid(ls='--')

    plt.tight_layout()


def gif_adquisicion(medicion, frames=200, skip=100):

    fig, ax = plt.subplots(1)

    x, y = ejes_largos(medicion)
    cantidad = np.zeros(frames * skip, dtype=int)

    cantidad[0] = 0
    hy, hx = np.histogram(cantidad[0], bins=np.arange(10))
    barras = ax.bar(hx[1:], hy)

    ax.set_xticks(np.arange(10))
    ax.set_ylim(0, 70)

    def update(i):
        for j in range(10):
            k = i + j
            xn = x[k * 2500:(k + 1) * 2500]
            xn = xn - xn[0]
            yn = y[k * 2500:(k + 1) * 2500]
            puntas = detecta_picos(yn, umbral=-0.005)
            cantidad[k] = sum(yn[puntas] < 0)

        hy, hx = np.histogram(cantidad[0:k], bins=np.arange(10))
        for barra, yi in zip(barras, hy):
            barra.set_height(yi)

        return barras

    animacion = animation.FuncAnimation(fig, update, frames,
                                        interval=25, blit=False)

    return animacion


def gif_adquisicion2(medicion, frames=20):

    fig, ax = plt.subplots(1)

    x, y = ejes_largos(medicion)
    y = y * 1000
    x = x * 1000

    ax.axhline(-2, ls='--', c='k')

    x0, y0 = x[0:2500], y[0:2500]
    puntas0 = detecta_picos(y0, umbral=-2)
    plot_curva, = ax.plot(x0, y0)
    plot_puntas, = ax.plot(x0[puntas0], y0[puntas0], marker='o',
                           mfc='white', ls='')

    ax.set_ylim(-15, 2)
    ax.set_xlabel(r'Tiempo ($ms$)')
    ax.set_ylabel(r'Tensión ($mV$)')

    plt.tight_layout()

    def update(i):
        k = i
        xn = x[k * 2500:(k + 1) * 2500]
        xn = xn - xn[0]
        yn = y[k * 2500:(k + 1) * 2500]
        puntas = detecta_picos(yn, umbral=-2)

        plot_curva.set_data(xn, yn)
        plot_puntas.set_data(xn[puntas], yn[puntas])

        return plot_curva, plot_puntas

    animacion = animation.FuncAnimation(fig, update, frames,
                                        interval=250, blit=False)

    return animacion


plt.ion()
