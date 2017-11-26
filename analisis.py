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


def poisson(x, lamb):
    """ Distribución de poisson normalizada  """
    return pdf_poisson.pmf(x, lamb)
#    d_poisson = ((1**x) * np.exp(-lamb)) / (factorial(x))
#    return d_poisson / np.trapz(d_poisson, x)


def bose(x, lamb):
    """ Distribución de Bose-Einstein normalizada """
    d_bose = (lamb**x) / ((1 + lamb)**(1 + x))
    return d_bose / np.trapz(d_bose, x)


def ajustar_poisson(hist_x, hist_y):
    """ Ajusta la distribución dada a la de Poisson """
    popt, pcov = curve_fit(poisson, hist_x, hist_y)
    return popt[0]


def ajustar_bose(hist_x, hist_y):
    """ Ajusta la distribución dada a la de Bose-Einstein """
    popt, pcov = curve_fit(bose, hist_x, hist_y)
    return popt[0]


def histograma(medicion, tamaño_ventana=2500,
               umbral=0.0, bins=10, ax=None,
               ajusta_poisson=False, ajusta_bose=False):
    """ Devuelve el histograma """

    x, y = ejes_largos(medicion)
    n_ventanas = len(y) // tamaño_ventana
    duracion = (x[0] - x[1]) * tamaño_ventana * 1000

    cantidad = np.zeros(n_ventanas, dtype=int)

    for i in range(n_ventanas):

        ventana = y[i * tamaño_ventana:(i + 1) * tamaño_ventana]
        picos = detecta_picos(ventana, umbral=0.0)
        alturas = ventana[picos]
        cantidad[i] = sum(alturas < umbral)

    if ax is None:
        fig, ax = plt.subplots(1)

    hist_y, hist_x = np.histogram(cantidad, bins=bins, density=True)
    hist_x += ((hist_x[1] - hist_x[0]) / 2)
    hist_x = hist_x[0:-1]

    # ax.hist(cantidad, bins=bins, density=True,
    ax.plot(hist_x, hist_y, 'go',
            label=r'$tc={:.2f}\;ms$'.format(duracion))

    if ajusta_poisson:
        lambda_poisson = ajustar_poisson(hist_x, hist_y)
        x_continuo = np.arange(1, 25)
        ax.plot(x_continuo, poisson(x_continuo, lambda_poisson))

    if ajusta_bose:
        maximo = np.argmax(hist_y)
        lambda_bose = ajustar_bose(hist_x[maximo:], hist_y[maximo:])
        x_continuo = np.arange(1, 25)
        ax.plot(x_continuo, bose(x_continuo, lambda_bose))

    return hist_x, hist_y


def graficar_ventanas(medicion, ventanas, umbral=-0.001,
                      ajusta_bose=False, ajusta_poisson=False):
    """ Grafica los histogramas para una lista de ventanas """
    fig, ax = plt.subplots(len(ventanas))
    ax[0].set_title(medicion.upper())

    for i, tamaño in enumerate(ventanas):
        histograma('poisson', tamaño_ventana=tamaño,
                   umbral=umbral, ax=ax[i],
                   ajusta_bose=ajusta_bose,
                   ajusta_poisson=ajusta_poisson)
        ax[i].legend()


if __name__ == '__main__':
    plt.ion()

    graficar_ventanas(medicion='poisson',
                      ventanas=(500, 1000, 2000, 5000),
                      umbral=-0.001,
                      ajusta_poisson=True)

    graficar_ventanas(medicion='bose',
                      ventanas=(500, 1000, 2000, 5000),
                      umbral=-0.001,
                      ajusta_bose=True)
