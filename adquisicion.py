# -*- coding: utf-8 -*-
"""
Archivo de adquisición complementario (medición adicional)
"""

import argparse
import os
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt

from instruments.oscilloscope import Oscilloscope


def argumentos():
    """ Función para capturar los argumentos desde la consola """

    parser = argparse.ArgumentParser()
    parser.add_argument('-nombre', type=str, default='medicion')
    parser.add_argument('-ruta', type=str, default='./datos/')
    parser.add_argument('-rsc', type=str,
                        default='USB0::1689::867::C102223::0::INSTR')
    parser.add_argument('-canal', type=str, default='CH2')
    parser.add_argument('-backend', type=str, default='@py')
    parser.add_argument('-pantallas', type=int, default=0)
    parser.add_argument('-guardar_todo', action='store_true')
    parser.add_argument('-ejes', action='store_true')
    parser.add_argument('-umbral', type=float, default=-0.002)
    parser.add_argument('-n_bins', type=int, default=100)
    parser.add_argument('-alturas', action='store_true')
    parser.add_argument('-histogramas', action='store_true')
    parser.add_argument('-autocorr', action='store_true')
    parser.add_argument('-i', '--interactivo', action='store_true')

    return parser.parse_args()


def detecta_picos(datos, umbral=0.0):
    """ Encuentra los picos y devuelve sus alturas """
    picos = datos * (datos < umbral)
    derivada = np.diff(picos)
    puntas = picos[1:] * (derivada < 0)
    alturas = puntas[puntas < 0]
    return alturas


def revisa_archivo(params):
    """ Revisa y resuelve los posibles conflictos con el archivo """

    ruta = os.path.expanduser(params.ruta)
    archivo, _ = os.path.splitext(params.medicion)
    ruta_completa = os.path.join(ruta + archivo + '.hdf5')

    hdf5 = h5py.File(ruta_completa, 'x')

    if params.altura or params.guardar_todo:
        hdf5.create_group('alturas')
    if params.histogramas or params.guardar_todo:
        hdf5.create_group('histogramas')
    if params.pantallas or params.guardar_todo:
        hdf5.create_group('ventanas')
    if params.autocorr:
        hdf5.create_group('autocorr')

    return hdf5


def progreso(cantidad):
    """ Generador para informar el avance en pantalla """

    time_start = time.time()
    fmt_avance = '{actual:0{c}d}/{total:0{c}d}'
    str_total = '--:--:--'

    cifras = len(str(cantidad))

    for i in range(cantidad):
        str_avance = fmt_avance.format(actual=i,
                                       total=cantidad,
                                       c=cifras)
        elapsed = time.gmtime(time.time() - time_start)
        str_elapsed = time.strftime('%X', elapsed) + ' / ' + str_total

        yield i, str_avance, str_elapsed

        total = (time.time() - time_start) * (cantidad / (i + 1))
        str_total = time.strftime('%X', time.gmtime(total))


def main(params):
    """ Rutina principal """

    hdf5 = revisa_archivo(params)
    osc = Oscilloscope(resource=params.rsc, backend=params.backend)

    osc.setup_curve(params.canal, start=1, stop=2500)
    osc.get_waveform_preamble()
    rango = (osc.get_y_range()[0], 0.0)

    cantidad = np.zeros(params.pantallas, dtype=int)
    hist_voltajes = np.zeros(params.n_bins, dtype=int)
    eje_y = np.zeros(2500, dtype=float)
    eje_x = osc.get_x()

    if params.interactive:
        plt.ion()

    for i, str_avance, str_elapsed in progreso(params.pantallas):
        print('\r' + str_avance + '-' + str_elapsed, end='')

        # Adquiere el eje Y
        eje_y = osc.get_y()

        if params.alturas or params.guardar_todo:
            # Busca las alturas de los picos y las guarda
            alturas = detecta_picos(eje_y, umbral=params.umbral)
            hdf5['/alturas/{:05d}'.format(i)] = alturas

            # Cuenta la cantidad de fotones
            cantidad[i] = len(alturas)

            if params.histogramas or params.guardar_todo:
                # Hace un histograma de alturas con bins fijos
                hist, bin_edges = np.histogram(alturas,
                                               bins=params.n_bins,
                                               range=rango)
                # Acumula los histogramas
                hist_voltajes += hist

        if params.autocorr:
            autocorr = np.correlate(eje_y, eje_y, 'same')
            hdf5['/autocorr/{:05}'.format(i)] = autocorr

        if params.ejes or params.guardar_todo:
            # Guarda las pantallas
            hdf5['/pantallas/{:05d}'.format(i)] = eje_y

        if params.interactive:
            pass

    if params.alturas or params.guardar_todo:
        hdf5['/cantidad'] = cantidad
        if params.histogramas or params.guardar_todo:
            hdf5['/hist_general'] = hist_voltajes
            hdf5['/bin_edges'] = bin_edges

    if params.ejes or params.guardar_todo:
        hdf5['/eje_x'] = eje_x


if __name__ == '__main__':
    main(argumentos())
