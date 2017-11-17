# -*- coding: utf-8 -*-

import numpy as np
import time
from instruments import Instrument


class ITC4001(Instrument):
    def __init__(self, resource=None, sim_mode=False, backend="@py",
                 query='?*::INSTR', name=None, path='./'):

        Instrument.__init__(self, resource, sim_mode,
                            backend, query, name, path)

    def measurement(self, scalar='TEMP', log=True):
        self.query('MEAS:{}'.format(scalar), log=log)

    def current_setpoint(self, current, log=True):
        self.write('SOUR:CURR {}'.format(current))

    def temperature_setpoint(self, temp, log=True):
        self.write('SOUR2:TEMP {}c'.format(temp))

    def sweep_temp(self, Ti=22.5, Tf=23.5, dT=0.01, dtime=4,
                   func=None, func_args=None):

        n = int((Tf - Ti) / dT)
        dT = dT * np.sign(Tf - Ti)
        T = np.zeros(n, dtype=float)
        t = np.zeros(n, dtype=float)

        self._inst.write('SOUR2:TEMP {}'.format(Ti))

        for i in range(n):
            t[i] = time.time()
            T[i] = self._inst('MEAS:TEMP?')

            if func is not None:
                func(i, *func_args)

            time.delay(dtime)
            self._inst.write('SOUR2:TEMP {}'.format(Ti + i * dT))

        t = t - t[0]

        return t, T
