# -*- coding: utf-8 -*-
"""
Archivo para el análisis de las mediciones
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson as pdf_poisson

from adquisicion import detecta_picos
from recupera_datos import recorrer_ejes, ejes_largos


def poisson(x, lamb, A):
    """ Distribución de poisson normalizada  """
    d_poisson = pdf_poisson.pmf(x, lamb)
    return A * d_poisson
#    d_poisson = ((1**x) * np.exp(-lamb)) / (factorial(x))
#    return d_poisson / np.trapz(d_poisson, x)


def bose(x, lamb, A):
    """ Distribución de Bose-Einstein normalizada """
    d_bose = (lamb**x) / ((1 + lamb)**(1 + x))
    return A * d_bose


def ajustar_poisson(hist_x, hist_y):
    """ Ajusta la distribución dada a la de Poisson """
    popt, pcov = curve_fit(poisson, hist_x, hist_y)
    return popt


def ajustar_bose(hist_x, hist_y):
    """ Ajusta la distribución dada a la de Bose-Einstein """
    popt, pcov = curve_fit(bose, hist_x, hist_y)
    return popt


def histograma(medicion, tamaño_ventana=2500,
               umbral=0.0, bins=25, ax=None, color='blue',
               ajusta_poisson=False, ajusta_bose=False,
               plot_histograma=True):
    """ Devuelve el histograma """

    x, y = ejes_largos(medicion)
    n_ventanas = len(y) // tamaño_ventana
    duracion = (x[0] - x[1]) * tamaño_ventana * 1e6

    cantidad = np.zeros(n_ventanas, dtype=int)

    for i in range(n_ventanas):

        ventana = y[i * tamaño_ventana:(i + 1) * tamaño_ventana]
        picos = detecta_picos(ventana, umbral=0.0)
        alturas = ventana[picos]
        cantidad[i] = sum(alturas < umbral)

    if ax is None:
        fig, ax = plt.subplots(1)

    hist_y, hist_x = np.histogram(cantidad, bins=np.arange(bins),
                                  density=True)
    # hist_x += ((hist_x[1] - hist_x[0]) / 2)
    hist_x = hist_x[0:-1]

    if plot_histograma:
        ax.bar(hist_x, hist_y, alpha=0.7, color=color,
               label=r'$T={:.2f}\;\mu s$'.format(abs(duracion)))

    if ajusta_poisson:
        lambda_poisson, A = ajustar_poisson(hist_x, hist_y)
        x_continuo = np.arange(bins)
        ax.plot(x_continuo, poisson(x_continuo, lambda_poisson, A),
                color='k', marker='o', markersize=10, mfc='white', ls='--')

    if ajusta_bose:
        maximo = np.argmax(hist_y)
        lambda_bose, A = ajustar_bose(hist_x[maximo:], hist_y[maximo:])
        x_continuo = np.arange(bins)
        ax.plot(x_continuo, bose(x_continuo, lambda_bose, A),
                color='k', marker='o', markersize=10, mfc='white', ls='--')

    ax.set_xticks(np.arange(bins)[:-1])
    ax.set_yticks([])

    return hist_x, hist_y


def graficar_ventanas(medicion, ventanas, umbral=-0.001, bins=15, color='blue',
                      ajusta_bose=False, ajusta_poisson=False):
    """ Grafica los histogramas para una lista de ventanas """
    fig, ax = plt.subplots(len(ventanas), sharex=True)

    fig.canvas.set_window_title(medicion.title())

    for i, tamaño in enumerate(ventanas):
        histograma(medicion, tamaño_ventana=tamaño, color=color,
                   umbral=umbral, ax=ax[i], bins=bins,
                   ajusta_bose=ajusta_bose,
                   ajusta_poisson=ajusta_poisson)
        ax[i].legend()
        ax[i].xaxis.grid(ls=':')

    return fig, ax


def autocorrelacion(medicion, ax=None, x_inc=0.5, color='blue'):
    """ Grafica la autocorrelación """

    if ax is None:
        fig, ax = plt.subplots(1)

    ax.set_ylim(-0.1, 0.15)
    plot_acorr, = ax.plot([], [], color=color, lw=2)
    ax.grid(ls=':')

    acorr = np.zeros(1250, dtype=float)

    for i, eje_x, eje_y in recorrer_ejes(medicion):
        acorr += fft_autocorrelacion(eje_y)
        plot_acorr.set_data(eje_x[1250:] * 1000, acorr / i)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)

    x = eje_x[1250:] * 1000
    x0 = x[np.argmax(acorr < 0)]
    ax.axvline(x0, ls='--')
    ax.set_ylabel('Autocorrelación')
    ax.set_xlabel('Tiempo ($ms$)')

    ax.set_xticks(np.arange(min(x), max(x), x_inc))

    return eje_x, acorr


def fft_autocorrelacion(x):
    """
    Compute the autocorrelation of the signal, based on the properties of the
    power spectral density of the signal.
    """
    xp = x - np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2 + np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:int(x.size / 2)] / np.sum(xp**2)


plt.ion()

if __name__ == '__main__':
    plt.ion()

    graficar_ventanas(medicion='poisson',
                      ventanas=(500, 1000, 2000, 5000),
                      umbral=-0.004,
                      ajusta_poisson=False,
                      ajusta_bose=True,
                      color='blue')

    plt.tight_layout()

    graficar_ventanas(medicion='bose',
                      ventanas=(500, 1000, 2000, 5000),
                      umbral=-0.004,
                      ajusta_poisson=True,
                      ajusta_bose=False,
                      color='orange')

    plt.tight_layout()
