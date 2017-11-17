# -*- coding: utf-8 -*-

import numpy as np
import time
from instruments import CommandGroup


class _lockin_reference(CommandGroup):

    def __init__(self, inst):
        CommandGroup.__init__(self, inst)

        self._crefMode = ('External', 'Internal')
        self._crefRslp = ('Sine', 'TTL Rising', 'TTL Falling')

    def _get_refPhase(self):
        return float(self._inst.query('PHAS?'))

    def _set_refPhase(self, value):
        f = self._option_limited(value, vmin=-360.00, vmax=729.99, prec=2)
        self._inst.write('PHAS {0:f}'.format(f))

    def _get_refMode(self):
        return self._crefMode[int(self._inst.query('FMOD?'))]

    def _set_refMode(self, value):
        i = self._option_list(value, self._crefMode)
        self._inst.write('FMOD {0}'.format(i))

    def _get_refFreq(self):
        return float(self._inst.query('FREQ?'))

    def _set_refFreq(self, value):
        f = self._option_limited(value, vmin=0.001, vmax=102000, prec=3)
        self._inst.write('FREQ {0:f}'.format(f))

    def _get_refAmpl(self):
        return float(self._inst.query('SLVL?'))

    def _set_refAmpl(self, value):
        f = self._option_limited(value, vmin=0.004, vmax=5.000, prec=3)
        self._inst.write('SLVL {0:f}'.format(f))

    def _get_refHarm(self):
        return int(self._inst.query('HARM?'))

    def _set_refHarm(self, value):
        freq = float(self._inst.query('FREQ?'))
        i = int(self._option_limited(value, vmin=1, vmax=19999, prec=0))
        try:
            self._option_limited(i * freq, vmin=0, vmax=102000.0, prec=4)
        except ValueError:
            raise ValueError('value * freq <= 102000Hz')
        self._inst.write('HARM {0}'.format(i))

    def _get_refRslp(self):
        return self._crefRslp[int(self._inst.query('RSLP?'))]

    def _set_refRslp(self, value):
        i = self._option_list(value, self._crefRslp)
        self._inst.write('RSLP {0}'.format(i))

    Phase = property(_get_refPhase, _set_refPhase)
    Mode = property(_get_refMode, _set_refMode)
    Frequency = property(_get_refFreq, _set_refFreq)
    Amplitude = property(_get_refAmpl, _set_refAmpl)
    Harmonic = property(_get_refHarm, _set_refHarm)
    ExternalTrigger = property(_get_refRslp, _set_refRslp)

    def _list_properties(self):
        lines = list()
        lines.append(self._format_property(name='Source',
                                           value=self.Mode))
        lines.append(self._format_property(name='Frequency',
                                           value=self.Frequency,
                                           unit='Hz'))
        lines.append(self._format_property(name='Phase',
                                           value=self.Phase,
                                           unit='º'))
        lines.append(self._format_property(name='Amplitude',
                                           value=self.Amplitude,
                                           unit='Vrms'))
        lines.append(self._format_property(name='Harmonic',
                                           value=self.Harmonic))
        lines.append(self._format_property(name='External Trigger',
                                           value=self.ExternalTrigger))
        return lines

    def _log(self):
        self._inst._log.underline('Reference and Phase panel:')
        self._inst._log.tabulated_lines(self._list_properties)


class _lockin_input(CommandGroup):

    def __init__(self, inst):
        CommandGroup.__init__(self, inst)

        self._cinpMode = ('A', 'A-B', 'I1', 'I100')
        self._cinpGnd = ('Float', 'Ground')
        self._cinpCoup = ('AC', 'DC')
        self._cinpLine = ('Out', 'Line In', '2xLine In', 'Both In')

        self._cinpRmod = ('High Reserve', 'Normal', 'Low Noise')
        self._cinpOfsl = ('6dB/oct', '12dB/oct', '18dB/oct', '24dB/oct')
        self._cinpSync = ('Off', 'On')

        self._strinpSens = ('2 nV/fA', '5 nV/fA', '10 nV/fA',
                            '20 nV/fA', '50 nV/fA', '100 nV/fA',
                            '200 nV/fA', '500 nV/fA', '1 ¼V/pA',
                            '2 ¼V/pA', '5 ¼V/pA', '10 ¼V/pA',
                            '20 ¼V/pA', '50 ¼V/pA', '100 ¼V/pA',
                            '200 ¼V/pA', '500 ¼V/pA',  '1 mV/nA',
                            '2 mV/nA', '5 mV/nA', '10 mV/nA',
                            '20 mV/nA', '50 mV/nA', '100 mV/nA',
                            '200 mV/nA', '500 mV/nA', '1 V/¼A')

        self._floatinpSens = (2e-9, 5e-9, 10e-9, 20e-9, 50e-9,
                              100e-9, 200e-9, 500e-9, 1e-6, 2e-6,
                              5e-6, 10e-6, 20e-6, 50e-6, 100e-6,
                              200e-6, 500e-6, 1e-3, 2e-3, 5e-3,
                              10e-3, 20e-3, 50e-3, 100e-3, 200e-3,
                              500e-3, 1)

        self._strinpOflt = ('10 us',  '30 us', '100 us',
                            '300 us', '1 ms', '3 ms',
                            '10 ms', '30 ms', '100 ms',
                            '300 ms', '1 s', '3 s',
                            '10 s', '30 s', '100 s',
                            '300 s', '1 ks', '3 ks',
                            '10 ks', '30 ks')

        self._floatinpOflt = (10e-6, 30e-6, 100e-6, 300e-6,
                              1e-3, 3e-3, 10e-3, 30e-3,
                              100e-3, 300e-3, 1.0, 3.0,
                              10.0, 30.0, 100.0,  300.0,
                              1e3, 3e3, 10e3, 30e3)

    def _get_inpMode(self):
        return self._cinpMode[int(self._inst.query('ISRC?'))]

    def _set_inpMode(self, value):
        i = self._option_list(value, self._cinpMode)
        self._inst.write('ISRC {0}'.format(i))

    def _get_inpGnd(self):
        return self._cinpGnd[int(self._inst.query('IGND?'))]

    def _set_inpGnd(self, value):
        i = self._option_list(value, self._cinpGnd)
        self._inst.write('IGND {0}'.format(i))

    def _get_inpCoup(self):
        return self._cinpCoup[int(self._inst.query('ICPL?'))]

    def _set_inpCoup(self, value):
        i = self._option_list(value, self._cinpCoup)
        self._inst.write('ICPL {0}'.format(i))

    def _get_inpLine(self):
        return self._cinpLine[int(self._inst.query('ILIN?'))]

    def _set_inpLine(self, value):
        i = self._option_list(value, self._cinpLine)
        self._inst.write('ILIN {0}'.format(i))

    def _get_inpSens(self):
        return int(self._inst.query('SENS?'))

    def _set_inpSens(self, value):
        i = self._option_list(value, self._strinpSens, self._floatinpSens)
        self._inst.write('SENS {0}'.format(i))

    def _get_inpRmod(self):
        return self._cinpRmod[int(self._inst.query('RMOD?'))]

    def _set_inpRmod(self, value):
        i = self._option_list(value, self._cinpRmod)
        self._inst.write('RMOD {0}'.format(i))

    def _get_inpOflt(self):
        return self._strinpOflt[int(self._inst.query('OFLT?'))]

    def _set_inpOflt(self, value):
        i = self._option_list(value, self._strinpOflt, self._floatinpOflt)
        self._inst.write('OFLT {0}'.format(i))

    def _get_inpOfsl(self):
        return self._cinpOfsl[int(self._inst.query('OFSL?'))]

    def _set_inpOfsl(self, value):
        i = self._option_list(value, self._cinpOfsl)
        self._inst.write('OFSL {0}'.format(i))

    def _get_inpSync(self):
        return self._cinpSync[int(self._inst.query('SYNC?'))]

    def _set_inpSync(self, value):
        i = self._option_list(value, self._cinpSync)
        self._inst.write('SYNC {0}'.format(i))

    Input = property(_get_inpMode, _set_inpMode)
    Ground = property(_get_inpGnd, _set_inpGnd)
    Couple = property(_get_inpCoup, _set_inpCoup)
    Notch = property(_get_inpLine, _set_inpLine)

    Sensitivity = property(_get_inpSens, _set_inpSens)
    Reserve = property(_get_inpRmod, _set_inpRmod)
    TimeConstant = property(_get_inpOflt, _set_inpOflt)
    LowPassFilter = property(_get_inpOfsl, _set_inpOfsl)
    SyncFilter = property(_get_inpSync, _set_inpSync)

    def _list_properties(self):
        lines = list()
        lines.append(self._format_property(name='Source',
                                           value=self.Input))
        lines.append(self._format_property(name='Ground',
                                           value=self.Ground))
        lines.append(self._format_property(name='Coupling',
                                           value=self.Couple))
        lines.append(self._format_property(name='Notch',
                                           value=self.Notch))
        lines.append(self._format_property(name='Sensitivity',
                                           value=self.Sensitivity))
        lines.append(self._format_property(name='Reserve',
                                           value=self.Reserve))
        lines.append(self._format_property(name='Time Constant',
                                           value=self.TimeConstant))
        lines.append(self._format_property(name='Low Pass Filter',
                                           value=self.LowPassFilter))
        lines.append(self._format_property(name='Sync. Filter',
                                           value=self.SyncFilter))
        return lines

    def _log(self):
        self._inst._log.underline('Input and Lock-in system panel:')
        self._inst._log.tabulated_lines(self._list_properties)


class _lockin_auxout(CommandGroup):

    def __init__(self, inst):
        CommandGroup.__init__(self, inst)

    def _get_auxout1(self):
        return float(self._inst.query('AUXV? 1'))

    def _set_auxout1(self, value):
        f = self._option_limited(value, vmin=-10.500, vmax=10.500, prec=3)
        self._inst.write('AUXV 1, {0:f}'.format(f))

    def _get_auxout2(self):
        return float(self._inst.query('AUXV? 2'))

    def _set_auxout2(self, value):
        f = self._option_limited(value, vmin=-10.500, vmax=10.500, prec=3)
        self._inst.write('AUXV 2, {0:f}'.format(f))

    def _get_auxout3(self):
        return float(self._inst.query('AUXV? 3'))

    def _set_auxout3(self, value):
        f = self._option_limited(value, vmin=-10.500, vmax=10.500, prec=3)
        self._inst.write('AUXV 3, {0:f}'.format(f))

    def _get_auxout4(self):
        return float(self._inst.query('AUXV? 4'))

    def _set_auxout4(self, value):
        f = self._option_limited(value, vmin=-10.500, vmax=10.500, prec=3)
        self._inst.write('AUXV 4, {0:f}'.format(f))

    AuxOut1 = property(_get_auxout1, _set_auxout1)
    AuxOut2 = property(_get_auxout2, _set_auxout2)
    AuxOut3 = property(_get_auxout3, _set_auxout3)
    AuxOut4 = property(_get_auxout4, _set_auxout4)

    def _list_properties(self):
        lines = list()
        lines.append(self._format_property(name='Auxiliar Output 1',
                                           value=self.AuxOut1,
                                           unit='V'))
        lines.append(self._format_property(name='Auxiliar Output 2',
                                           value=self.AuxOut2,
                                           unit='V'))
        lines.append(self._format_property(name='Auxiliar Output 3',
                                           value=self.AuxOut3,
                                           unit='V'))
        lines.append(self._format_property(name='Auxiliar Output 4',
                                           value=self.AuxOut4,
                                           unit='V'))
        return lines

    def _log(self):
        self._inst._log.underline('Auxiliar outputs:')
        self._inst._log.tabulated_lines(self._list_properties())


class _lockin_ch1(CommandGroup):

    def __init__(self, inst):
        CommandGroup.__init__(self, inst)

        self._cch1Mode = ('X', 'R', 'Xn', 'Aux1', 'Aux2')
        self._cch1RatS = ('None', 'Aux In 1', 'Aux In 2')
        self._cch1OutS = ('Display', 'X')
        self._cExpand = ('x1', 'x10', 'x100')

    def _get_ch1Mode(self):
        d, r = self._inst.query_ascii_values('DDEF? 1', separator=',')
        return self._cch1Mode[d]

    def _set_ch1Mode(self, value):
        d, r = self._inst.query_ascii_values('DDEF? 1', separator=',')
        d = self._option_list(value, self._cch1Mode)
        self._inst.write('DDEF 1, {d}, {r}'.format(d=d, r=r))

    def _get_ch1RatS(self):
        d, r = self._inst.query_ascii_values('DDEF? 1', separator=',')
        return self._cch1Rats[r]

    def _set_ch1RatS(self, value):
        d, r = self._inst.query_ascii_values('DDEF? 1', separator=',')
        r = self._option_list(value, self._cch1Rats)
        self._inst.write('DDEF 1, {d}, {r}'.format(d=d, r=r))

    def _get_ch1OutS(self):
        return self._cch1OutS[int(self._inst.query('FPOP? 1'))]

    def _set_ch1OutS(self, value):
        i = self._option_list(value, self._cch1OutS)
        self._inst.write('FPOP 1, {}'.format(i))

    def _get_offsetX(self):
        x, j = self._inst.query_ascii_values('OEXP? 1', separator=',')
        return x

    def _set_offsetX(self, value):
        x, j = self._inst.query_ascii_values('OEXP? 1', separator=',')
        x = self._option_limited(value, vmin=-105.00, vmax=105.00, prec=2)
        self._inst.write('OEXP 1, {x}, {j}'.format(x=x, j=j))

    def _get_offsetY(self):
        x, j = self._inst.query_ascii_values('OEXP? 2', separator=',')
        return x

    def _set_offsetY(self, value):
        x, j = self._inst.query_ascii_values('OEXP? 2', separator=',')
        x = self._option_limited(value, vmin=-105.00, vmax=105.00, prec=2)
        self._inst.write('OEXP 2, {x}, {j}'.format(x=x, j=j))

    def _get_expandX(self):
        x, j = self._inst.query_ascii_values('OEXP? 1', separator=',')
        return self._cExpand[j]

    def _set_expandX(self, value):
        x, j = self._inst.query_ascii_values('OEXP? 1', separator=',')
        j = self._option_list(value, self._cExpand)
        self._inst.write('OEXP 1, {x}, {j}'.format(x=x, j=j))

    def _get_expandY(self):
        x, j = self._inst.query_ascii_values('OEXP? 2', separator=',')
        return self._cExpand[j]

    def _set_expandY(self, value):
        x, j = self._inst.query_ascii_values('OEXP? 2', separator=',')
        j = self._option_list(value, self._cExpand)
        self._inst.write('OEXP 2, {x}, {j}'.format(x=x, j=j))

    Display = property(_get_ch1Mode, _set_ch1Mode)
    Ratio = property(_get_ch1RatS, _set_ch1RatS)
    Output = property(_get_ch1OutS, _set_ch1OutS)
    OffsetX = property(_get_offsetX, _set_offsetX)
    OffsetY = property(_get_offsetY, _set_offsetY)
    ExpandX = property(_get_expandX, _set_expandX)
    ExpandY = property(_get_expandY, _set_expandY)

    def _list_properties(self):
        lines = list()
        lines.append(self._format_property(name='Source',
                                           value=self.Display))
        lines.append(self._format_property(name='Ratio',
                                           value=self.Ratio))
        lines.append(self._format_property(name='Output',
                                           value=self.Output))
        lines.append(self._format_property(name='Offset X',
                                           value=self.OffsetX,
                                           unit='%'))
        lines.append(self._format_property(name='Offset Y',
                                           value=self.OffsetY,
                                           unit='%'))
        lines.append(self._format_property(name='Expand X',
                                           value=self.ExpandX,
                                           unit='%'))
        lines.append(self._format_property(name='Expand Y',
                                           value=self.ExpandX,
                                           unit='%'))
        return lines

    def _log(self):
        self._inst._log.underline('Channel 1 panel:')
        self._inst._log.tabulated_lines(self._list_properties)


class _lockin_ch2(CommandGroup):

    def __init__(self, inst):
        CommandGroup.__init__(self, inst)

        self._cch2Mode = ('Y', 'T', 'Yn', 'Aux3', 'Aux4')
        self._cch2RatS = ('None', 'Aux In 3', 'Aux In 4')
        self._cch2OutS = ('Display', 'Y')
        self._cExpand = ('x1', 'x10', 'x100')

    def _get_ch2Mode(self):
        d, r = self._inst.query_ascii_values('DDEF? 2', separator=',')
        return self._cch2Mode[d]

    def _set_ch2Mode(self, value):
        d, r = self._inst.query_ascii_values('DDEF? 2', separator=',')
        d = self._option_list(value, self._cch1Mode)
        self._inst.write('DDEF 1, {d}, {r}'.format(d=d, r=r))

    def _get_ch2RatS(self):
        d, r = self._inst.query_ascii_values('DDEF? 2', separator=',')
        return self._cch2Rats[r]

    def _set_ch2RatS(self, value):
        d, r = self._inst.query_ascii_values('DDEF? 2', separator=',')
        r = self._option_list(value, self._cch1Rats)
        self._inst.write('DDEF 1, {d}, {r}'.format(d=d, r=r))

    def _get_ch2OutS(self):
        return self._cch2OutS[int(self._inst.query('FPOP? 2'))]

    def _set_ch2OutS(self, value):
        i = self._option_list(value, self._cch2OutS)
        self._inst.write('FPOP 2, {}'.format(i))

    def _get_offsetR(self):
        x, j = self._inst.query_ascii_values('OEXP? 3', separator=',')
        return x

    def _set_offsetR(self, value):
        x, j = self._inst.query_ascii_values('OEXP? 3', separator=',')
        x = self._option_limited(value, vmin=-105.00, vmax=105.00, prec=2)
        self._inst.write('OEXP 3, {x}, {j}'.format(x=x, j=j))

    def _get_expandR(self):
        x, j = self._inst.query_ascii_values('OEXP? 3', separator=',')
        return self._cExpand[j]

    def _set_expandR(self, value):
        x, j = self._inst.query_ascii_values('OEXP? 3', separator=',')
        j = self._option_list(value, self._cExpand)
        self._inst.write('OEXP 3, {x}, {j}'.format(x=x, j=j))

    Display = property(_get_ch2Mode, _set_ch2Mode)
    Ratio = property(_get_ch2RatS, _set_ch2RatS)
    Output = property(_get_ch2OutS, _set_ch2OutS)
    OffsetR = property(_get_offsetR, _set_offsetR)
    ExpandR = property(_get_expandR, _set_expandR)

    def _list_properties(self):
        lines = list()
        lines.append(self._format_property(name='Source',
                                           value=self.Display))
        lines.append(self._format_property(name='Ratio',
                                           value=self.Ratio))
        lines.append(self._format_property(name='Output',
                                           value=self.Output))
        lines.append(self._format_property(name='Offset R',
                                           value=self.OffsetX,
                                           unit='%'))
        lines.append(self._format_property(name='Expand R',
                                           value=self.ExpandX,
                                           unit='%'))
        return lines

    def _log(self):
        self._inst._log.underline('Channel 2 panel:')
        self._inst._log.tabulated_lines(self._list_properties)


class _lockin_interface(CommandGroup):

    def __init__(self, inst):
        CommandGroup.__init__(self, inst)

        self._cstpLock = ('Local', 'Remote', 'Local Lockout')

    def _get_stpLock(self):
        return self._cstpLock[int(self._inst.query('LOCL?'))]

    def _set_stpLock(self, value):
        i = self._option_list(value, self._cstpLock)
        self._inst.write('LOCL{0}'.format(i))

    Lock = property(_get_stpLock, _set_stpLock)

    def _list_properties(self):
        lines = list()
        lines.append(self._format_property(name='Local / Remote',
                                           value=self.Lock))
        return lines

    def _log(self):
        self._inst._log.underline('Interface panel:')
        self._inst._log.tabulated_lines(self._list_properties)


class _lockin_autofuncs(CommandGroup):

    def __init__(self, inst):
        CommandGroup.__init__(self, inst)

    def autoGain(self):
        """ Run autoGain function """
        self._inst.write('AGAN')
        print('Running autoGain... ', end='')
        while not bool(self._inst.query('*STB?1')):
            pass
        print('Finalizado')

    def autoReserve(self):
        """ Run autoReserve function """
        self._inst.write('ARSV')
        print('Running autoReserve... ', end='')
        while not bool(self._inst.query('*STB?1')):
            pass
        print('Finalizado')

    def autoPhase(self):
        """ Run autoPhase function """
        self._inst.write('APHS')
        print('Running autoPhase... ', end='')
        time.sleep(2)
        print('Finalizado')

    def autoOffset(self, num):
        """ Run autoOffset function """
        if not (num == 1 or num == 2 or num == 3):
            raise ValueError('X(1), Y(2), R(3)')
        self._inst.write('AOFF {0}'.format(num))
        print('Running autoOffset... ', end='')
        time.sleep(2)
        print('Finalizado')


class _lockin_setup(CommandGroup):

    def __init__(self, inst):
        CommandGroup.__init__(self, inst)

    def Save(self, slot=1):
        """ Save state in slot (1 to 9) """
        if not (isinstance(slot, int) and 1 <= slot <= 9):
            raise ValueError('Expected slot 1 to 9')
        self._inst.write('SSET {0}'.format(slot))
        print('Setup saved in slot {0} '.format(slot))

    def Recall(self, slot=1):
        """ Load state in slot (1 to 9) """
        if not (isinstance(slot, int) and 1 <= slot <= 9):
            raise ValueError('Expected slot 1 to 9')
        self._inst.write('RSET {0}'.format(slot))
        print('Setup loaded from slot {0}'.format(slot))


class _lockin_adquisition(CommandGroup):

    def __init__(self, inst):
        CommandGroup.__init__(self, inst)

    def read_value(self, value='R', log=True):
        value = value.upper()
        options = {'X': 1, 'Y': 2, 'R': 3, 'T': 4}

        if value in options:
            i = options[value]
        elif value in options.values():
            i = value
        else:
            raise ValueError(options)

        return float(self._inst.query('OUTP? {0}'.format(i), log=log))

    def read_display(self, display=1):
        if display in [1, 2]:
            return float(self._inst.query('OUTR? {0}'.format(display)))
        else:
            raise ValueError('CH1=1, CH2=2')

    def read_multiple(self, *args, log=True):
        options = {'X': 1, 'Y': 2, 'R': 3, 'T': 4,
                   'AuxIn1': 5, 'AuxIn2': 6, 'AuxIn3': 7, 'AuxIn4': 8,
                   'Freq': 9, 'Ch1': 10, 'Ch2': 11}

        if 2 <= len(args) <= 6:
            command = 'SNAP? '
            for param in args:
                if param in options:
                    command += str(options[param]) + ', '
                else:
                    raise ValueError(options.keys())
            command = command[0:-2]
        else:
            raise ValueError('Expected 2 to 6 args')

        return self._inst.query_ascii_values(command, separator=",", log=log)

    def read_auxiliar(self, aux=1):
        if aux in [1, 2, 3, 4]:
            return float(self._inst.query('OAUX? {0}'.format(aux)))
        else:
            raise ValueError('Auxiliar inputs: 1 to 4')

    def sweep_freq(self, start, end, step=None, n=200, delay=0,
                   params=['X', 'Y', 'R', 'T'], log=True):

        if step is None:
            step = (end - start) / n
        else:
            if step == 0:
                raise ValueError('Expected step != 0')
            step = abs(step) * np.sign(end - start)
            n = (end - start) / step

        self._option_limited(start, vmin=0.001, vmax=102000, prec=3)
        self._option_limited(end, vmin=0.001, vmax=102000, prec=3)

        try:
            self.read_multiple(*params, log=False)
        except ValueError:
            raise

        freqs = start + np.arange(n, dtype=float) * step
        reads = np.zeros([n, len(params)], dtype=float)

        for i, f in enumerate(freqs):
            self._inst.write('FREQ {0:f}'.format(f), log=False)
            reads[i] = self.read_multiple(*params, log=False)
            time.sleep(delay)

        if log:
            save = self._inst.save([freqs, reads.T])

            command = 'Sweep Frequency {.f} to {.f}Hz '.format(start, end)
            command += 'with step {.f}Hz'.format(step)
            answer = 'Freq, ('
            for param in params:
                answer += str(param) + ', '
            command = command[0:-2] + ') -> '

            self._log.time_stamp(command, answer + save)

        return freqs, reads.T

    @property
    def buffer_stored(self):
        return int(self._inst.query('SPTS?', log=False))

    def buffer_one_shoot(self, sample_rate='64 Hz', points=16383,
                         wait=True):
    
        strFreqs = ('62.5 mHz', '125 mHz', '250 mHz', '500 mHz', '1 Hz',
                    '2 Hz', '4 Hz', '8 Hz', '16 Hz', '32 Hz',
                    '64 Hz', '128 Hz', '256 Hz', '512 Hz')

        floatFreqs = (0.0625, 0.125, 0.25, 0.5, 1.0,
                      2.0, 4.0, 8.0, 16.0, 32.0,
                      64.0, 128.0, 256.0, 512.0)

        i = self._option_list(sample_rate, strFreqs, floatFreqs)

        if not (1 <= points <= 16383):
            raise ValueError('Points out of range (1 to 16383)')

        self._inst.write('SRAT {}'.format(i), log=False)
        self._inst.write('SEND 0', log=False)
        print('Buffer reset', end='')
        self._inst.write('REST', log=False)
        print('. Done')
        self._inst.write('STRT', log=False)
        print('Buffer storage in progress')

        pr = self.buffer_stored

        if wait:
            while pr < points:
                pr = self.buffer_stored
                msg = '\rRemaining: {:5d}/{:5d}'.format(pr, points)
                print(msg, end='\r')
                time.sleep(0.1)
            print(msg + '. Done')

    def buffer_read(self, display=1, start=0, end=16383):

        if not (1 <= end <= 16383) or start > end:
            raise ValueError('Points out of range (0 to 16383)')

        if end > self.buffer_stored:
            raise ValueError('Not enough points in buffer yet')

        if self._inst.query('SEND?', log=False) == 1:
            self._inst.write('PAUS', log=False)

        command = 'TRCA? {}, {}, {}'.format(display, start, end)

        self._inst._inst.write(command)
        data_raw = self._inst._inst.read_raw()
        data = data_raw.decode()[0:-1]
        data = data.split(',')

        return np.asarray(data, dtype=float)
