# -*- coding: utf-8 -*-
"""
Herramientas para la práctica de Conteo de Fotones
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os


def nuevo_dataset(datos, nombre='conteo.hdf5', grupo='general', atrib=None):
    hdf5 = h5py.File(nombre, 'a')
    nuevo = len(hdf5[grupo]) + 1)
