# -*- coding: utf-8 -*-
"""
Archivo para la recuperación de las mediciones
"""

import os
import h5py
import numpy as np


def recorrer_ejes(nombre, rango=None):

    ruta = os.path.expanduser('./mediciones/')
    archivo, _ = os.path.splitext(nombre)
    ruta_completa = os.path.join(ruta + '/' + archivo + '.hdf5')
    hdf5 = h5py.File(ruta_completa, 'r')

    if rango is None:
        rango = range(len(hdf5['/pantallas/']))

    eje_x = np.asarray(hdf5['/eje_x'][:])

    for i in rango:
        eje_y = np.asarray(hdf5['/pantallas/{:05d}'.format(i)][:])
        yield i, eje_x, eje_y


def ejes_largos(nombre):
    ruta = os.path.expanduser('./mediciones/')
    archivo, _ = os.path.splitext(nombre)
    ruta_completa = os.path.join(ruta + '/' + archivo + '.hdf5')
    hdf5 = h5py.File(ruta_completa, 'r')

    n = len((hdf5['/pantallas/']))

    x_inc = hdf5['/eje_x'][1] - hdf5['/eje_x'][0]

    eje_largo_x = np.arange(n * 2500) * x_inc
    eje_largo_y = np.zeros(n * 2500, dtype=float)

    for i, eje_x, eje_y in recorrer_ejes(nombre):
        eje_largo_y[i * 2500:(i + 1) * 2500] = eje_y

    return eje_largo_x, eje_largo_y
