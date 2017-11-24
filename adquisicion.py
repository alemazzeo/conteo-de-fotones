# -*- coding: utf-8 -*-
"""
Archivo de adquisición complementario (medición adicional)
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt

from instruments.oscilloscope import Oscilloscope
from instruments.tools import FileTools


def argumentos():
    """ Función para capturar los argumentos desde la consola """

    parser = argparse.ArgumentParser()
    parser.add_argument('-nombre', type=str, default='med',
                        help='nombre del archivo')
    parser.add_argument('-ruta', type=str, default='./mediciones/',
                        help='ubicación del archivo')
    parser.add_argument('-rsc', type=str,
                        default='USB0::1689::867::C102223::0::INSTR',
                        help='resource')
    parser.add_argument('-canal', type=str, default='CH2',
                        help='canal: "CH1" o "CH2"')
    parser.add_argument('-backend', type=str, default='@py',
                        help='"" para NI, "@py" para pyvisa-py')
    parser.add_argument('-pantallas', type=int, default=0,
                        help='número de pantallas')
    parser.add_argument('-guardar_todo', action='store_true',
                        help='guarda ejes, alturas e histogramas')
    parser.add_argument('-ejes', action='store_true',
                        help='guarda ejes')
    parser.add_argument('-umbral', type=float, default=-0.002,
                        help='pone un umbral, None=automático')
    parser.add_argument('-n_bins', type=int, default=100,
                        help='número de bins para el histograma')
    parser.add_argument('-alturas', action='store_true',
                        help='guarda las alturas de los picos')
    parser.add_argument('-histogramas', action='store_true',
                        help='guarda los histogramas individuales')
    parser.add_argument('-autocorr', action='store_true',
                        help='guarda la autocorrelación')
    parser.add_argument('-i', '--interactivo', action='store_true',
                        help='activa el modo interactivo')
    parser.add_argument('-n_parcial', type=int, default=10,
                        help='intervalo entre gráficos')

    return parser.parse_args()


def detecta_picos(datos, umbral=0.0):
    """ Encuentra los picos y devuelve sus alturas """
    picos = datos * (datos < umbral)
    derivada = np.append(0, np.diff(picos))
    puntas = picos * (derivada < 0)
    return puntas < 0


def crea_archivo(ruta_completa, params):
    """ Crea el archivo para almacenar la medición """

    print('Preparando archivo {}... '.format(ruta_completa), end='')

    hdf5 = h5py.File(ruta_completa, 'w')

    if params.alturas or params.guardar_todo:
        hdf5.create_group('alturas')
    if params.histogramas or params.guardar_todo:
        hdf5.create_group('histogramas')
    if params.pantallas or params.guardar_todo:
        hdf5.create_group('ventanas')
    if params.autocorr:
        hdf5.create_group('autocorr')

    print('Hecho.')

    return hdf5


def revisa_archivo(params):
    """ Revisa y resuelve los posibles conflictos con el archivo """

    ruta = os.path.expanduser(params.ruta)
    archivo, _ = os.path.splitext(params.nombre)
    ruta_completa = os.path.join(ruta + archivo + '.hdf5')

    if os.path.isfile(ruta_completa):
        print('\nYa existe el archivo: "{}"'.format(ruta_completa))
        nueva_ruta_completa = FileTools.newname(fullname=ruta_completa,
                                                default='./medicion.hdf5')
        nuevo_archivo = FileTools.splitname(nueva_ruta_completa)[1]
        print('Puede continuar con {}, '.format(
            nuevo_archivo + '.hdf5'), end='')
        print('reemplazar el archivo existente o salir')
        rta = input('[C]ontinuar, [R]eemplazar, [S]alir: ').upper()
        while rta not in ('C', 'R', 'S'):
            rta = input('[C]ontinuar, [R]eemplazar, [S]alir: ').upper()

        if rta == 'C':
            return crea_archivo(nueva_ruta_completa, params)

        elif rta == 'R':
            msj = 'Esta acción no puede deshacerse. ¿Continuar? (S/N): '
            rta2 = input(msj).upper()
            while rta2 not in ('SI', 'NO', 'S', 'N'):
                rta2 = input(msj).upper()
            if rta2 in ('SI', 'S'):
                return crea_archivo(ruta_completa, params)
            print('...')
            return revisa_archivo(params)

        elif rta == 'S':
            sys.exit(0)

    return crea_archivo(ruta_completa, params)


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
    # osc = Oscilloscope(resource=params.rsc, backend=params.backend)

    # osc.setup_curve(params.canal, start=1, stop=2500)
    # osc.get_waveform_preamble()
    # rango = (osc.get_y_range()[0], 0.0)
    rango = (-1, 0)

    cantidad = np.zeros(params.pantallas, dtype=int)
    hist_voltajes = np.zeros(params.n_bins, dtype=int)
    bin_edges = np.zeros(params.n_bins + 1, dtype=float)
    eje_y = np.zeros(2500, dtype=float)
    puntas = np.zeros(2500, dtype=float)
    # eje_x = osc.get_x()
    eje_x = np.arange(2500)

    if params.interactivo:
        plt.ion()
        fig, ax = plt.subplots(2, sharex=True)
        plot_curva, = ax[0].plot(eje_y, eje_x)
        plot_puntas, = ax[0].plot([], [], ls='', marker='o', mfc='white',
                                  color='k')
        ax[0].set_xlim(rango)
        plot_hist, = ax[1].plot(
            [], [], ls='', marker='o', mfc='white', color='k')

    print('\nAdquisición iniciada. {}'.format(time.strftime('%x - %X')))
    try:
        for i, str_avance, str_elapsed in progreso(params.pantallas):
            print('\r(' + str_avance + ') --- ' + str_elapsed, end='')

            # Adquiere el eje Y
            # eje_y = osc.get_y()
            eje_y = -np.random.random(2500) * (np.random.random(2500) > 0.995)
            # time.sleep(1)

            if params.alturas or params.guardar_todo:
                # Busca las alturas de los picos y las guarda
                puntas = detecta_picos(eje_y, umbral=params.umbral)
                alturas = eje_y[puntas]
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

            if params.interactivo:
                plot_curva.set_data(eje_y, eje_x)
                plot_puntas.set_data(eje_y[puntas], eje_x[puntas])
                if i % params.n_parcial == 0:
                    plot_hist.set_data(bin_edges[0:-1], hist_voltajes)
                    ax[1].relim()
                    ax[1].autoscale_view()
                plt.pause(0.001)

        print('\rAdquisición finalizada. {}'.format(time.strftime('%x - %X')))
    except (KeyboardInterrupt, NameError):
        print('\n\nAdquisición interrumpida. {}'.format(time.strftime('%x - %X')))
        if params.alturas or params.guardar_todo:
            print('Mediciones recortadas en la posición {}.'.format(i - 1))
            cantidad = cantidad[0:i - 1]

    if params.alturas or params.guardar_todo:
        hdf5['/cantidad'] = cantidad
        if params.histogramas or params.guardar_todo:
            hdf5['/hist_general'] = hist_voltajes
            hdf5['/bin_edges'] = bin_edges

    if params.ejes or params.guardar_todo:
        hdf5['/eje_x'] = eje_x

    print('Se almacenaron en {} las siguientes mediciones: \n'.format(hdf5.filename))

    for nombre in hdf5:
        print('   {}'.format(nombre))


if __name__ == '__main__':
    main(argumentos())
