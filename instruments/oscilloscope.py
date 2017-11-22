# -*- coding: utf-8 -*-

import numpy as np
from .instruments import Instrument


class Oscilloscope(Instrument):
    def __init__(self, resource=None, sim_mode=False, backend="@py",
                 query='?*::INSTR', name=None, path='./'):

        Instrument.__init__(self, resource, sim_mode,
                            backend, query, name, path)

        self._xze = None
        self._xin = None
        self._yze = None
        self._ymu = None
        self._yoff = None

        self._start = 1
        self._stop = 2500

        # self.setup_curve()
        # self.get_waveform_preamble()

    def setup_curve(self, source='CH1', mode='RPB',
                    width=1, start=1, stop=2500):

        self._inst.write('DATa:SOUrce {}'.format(source))
        self._inst.write('DATa:ENC {}'.format(mode))
        self._inst.write('DATa:WIDth {}'.format(width))
        self._inst.write('DATa:STARt {}'.format(start))
        self._inst.write('DATa:STOP {}'.format(stop))
        self._start = start
        self._stop = stop

    def get_waveform_preamble(self, log=True):
        query = 'WFMPRE:XZE?;XIN?;YZE?;YMU?;YOFF?;'
        answer = self.query_ascii_values(query, separator=';', log=log)
        self._xze = answer[0]
        self._xin = answer[1]
        self._yze = answer[2]
        self._ymu = answer[3]
        self._yoff = answer[4]

    def get_curve(self, auto_wfmpre=True, log=True, save_temp=False):
        if auto_wfmpre:
            self.get_waveform_preamble(log=log)

        y = self._inst.query_binary_values('CURV?', datatype='B',
                                           container=np.array)
        y = (y - self._yoff) * self._ymu + self._yze
        x = self._xze + np.arange(len(y)) * self._xin

        if save_temp:
            save = self.save([x, y])
            self._log.time_stamp('CURV?', answer=save)

        return x, y

    def get_y(self):
        y = self._inst.query_binary_values('CURV?', datatype='B',
                                           container=np.array)

        return (y - self._yoff) * self._ymu + self._yze

    def get_x(self):
        n = self._stop - self._start - 1
        return self._xze + np.arange(n) * self._xin
