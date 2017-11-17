# -*- coding: utf-8 -*-

from instruments import Instrument

from lockin_commands import _lockin_autofuncs
from lockin_commands import _lockin_auxout
from lockin_commands import _lockin_ch1
from lockin_commands import _lockin_ch2
from lockin_commands import _lockin_input
from lockin_commands import _lockin_interface
from lockin_commands import _lockin_reference
from lockin_commands import _lockin_setup
from lockin_commands import _lockin_adquisition


class Lockin(Instrument):
    '''
    Class for PyVISA control of Lock-in Amplifier SR830.
    '''

    def __init__(self, resource=None, sim_mode=False, backend="@py",
                 query='?*::INSTR', name=None, path='./'):

        Instrument.__init__(self, resource, sim_mode,
                            backend, query, name, path)

        self.adquisition = _lockin_adquisition(self)
        self.input_panel = _lockin_input(self)
        self.ch1_panel = _lockin_ch1(self)
        self.ch2_panel = _lockin_ch2(self)
        self.auto_panel = _lockin_autofuncs(self)
        self.setup_panel = _lockin_setup(self)
        self.interface_panel = _lockin_interface(self)
        self.reference_panel = _lockin_reference(self)
        self.auxiliar_outs = _lockin_auxout(self)
