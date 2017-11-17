# -*- coding: utf-8 -*-

import numpy as np
import visa
import time
from tools import FileTools as FT
from tools import PromptTools as PT
from tools import LogTools as LT

MAX_LOG_ANSWERS = 10


class SimVISA(object):
    '''
    Emulator for test with VISA.

    Replaces write and query functions with prompts in console.
    Only for algorithm testing.
    '''

    def __init__(self):
        print('VISA simulation enabled')

    def write(self, command, termination=None, encoding=None):
        # Print command in console
        print(">> " + command)

    def query(self, command, separator=''):
        # Print command and request simulated answer
        print(">> " + command)
        if command == '*IDN?':
            return "Simulated VISA Device, SimVISA"
        else:
            return input("Answer: ")

    def query_ascii_values(self, command, **args):
        # Print command and generate random curve
        print(">> " + command)
        return np.random.rand(100), np.random.rand(100), 0, 0

    def close(*args, **kwargs):
        pass


class Instrument():
    def __init__(self, resource=None, sim_mode=False, backend="@py",
                 query='?*::INSTR', name=None, path='./'):

        if sim_mode:
            # Create a simulated instrument
            self._resource = 'SimVISA'
            self._inst = SimVISA()

        else:
            rm = visa.ResourceManager(backend)
            # Help to find resource
            if resource is None:

                available = rm.list_resources(query=query)

                # Filter results with the (optional) name provided
                if name is not None:
                    available = {k: v for k, v in available.items()
                                 if v.resource_name == name}

                # Generate a list with results
                if len(available) >= 1:
                    resources = list()
                    options = list()
                    for i, item in enumerate(available):
                        resources.append(item)
                        option = '{:s}'.format(item)
                        options.append(option)

                    # Make a list for multiples values
                    if len(available) > 1:
                        message = "Available devices:\n"
                        number = PT.select_prompt(options, message)
                        self._resource = resources[number]

                    # Or autoselect
                    else:
                        self._resource = resources[0]

                # Raise exception for negative query results
                else:
                    raise(visa.VisaIOError(-1073807343))

            # Use the resource selected
            else:
                self._resource = resource

            # Open resource
            self._inst = rm.open_resource(self._resource)

        self._idn = self._inst.query('*IDN?')
        self._name = self._idn.split(',')[0] + '-' + self._idn.split(',')[1]

        self._fullname = '{}/{}'.format(path, self._name)
        self._path = path
        self._path_data = path + '/data/temp.npy'

        self._log = LT('{}.log'.format(self._fullname))
        self._temp_list = list()

        self._log.block(time.strftime('%x - %X'),
                        'Session started',
                        self._resource,
                        self._name)

    def __del__(self):
        self._log.block(time.strftime('%x - %X'),
                        'Session closed')
        self._inst.close()

    def write(self, command, termination=None, encoding=None, log=True):
        self._inst.write(command, termination, encoding)
        if log:
            self._log.time_stamp(command)

    def query(self, command, delay=None, log=True):
        answer = self._inst.query(command, delay)
        if log:
            self._log.time_stamp(command, answer)
        return answer

    def query_ascii_values(self, command, converter='f', separator=',',
                           container=list, delay=None, log=True):
        answer = self._inst.query_ascii_values(command, converter,
                                               separator, container,
                                               delay)
        if log:
            if len(answer) < MAX_LOG_ANSWERS:
                for value in answer:
                    self._log.time_stamp(answer=value)
            else:
                save = self.save(answer)
                self._log.time_stamp(command, answer=save)
        return answer

    def query_binary_values(self, command, datatype='f', is_big_endian=False,
                            container=list, delay=None, header_fmt='ieee',
                            log=True):

        answer = self._inst.query_binary_values(command, datatype,
                                                is_big_endian, container,
                                                delay, header_fmt)
        if log:
            save = self.save(answer)
            self._log.time_stamp(command, answer=save)
        return answer

    def save(self, data, fullname='temp.npy'):
        fullname = FT.newname(fullname, default=self._path_data)
        np.save(fullname, data)
        self._temp_list.append(fullname)
        return fullname

    def load(self, fullname='temp.npy'):
        fullname = FT.file_exist(fullname, default_path=self._path_data)
        return np.load(fullname)


class CommandGroup(object):

    def __init__(self, inst):
        self._inst = inst

    def _option_list(self, value, options, floats=None):
        error = 'Type option or number (0-{})'.format(len(options))
        upper_options = [opt.upper() for opt in options]

        if isinstance(value, int):
            if 0 <= value <= len(options):
                return value
            else:
                for i, opt in enumerate(options):
                    print('{} - {}'.format(i, opt))
                raise ValueError(error)

        elif isinstance(value, str):
            if value.upper() in upper_options:
                return upper_options.index(value.upper())
            else:
                for i, opt in enumerate(options):
                    print('{} - {}'.format(i, opt))
                raise ValueError(error)

        elif isinstance(value, float):
            if floats is None:
                raise ValueError(error)
            else:
                if value in floats:
                    return floats.index(value)
                else:
                    raise ValueError(error)

        else:
            raise ValueError(error)

    def _option_limited(self, value, vmin, vmax, prec=4):
        error = '{:.{p}f} <= value <= {:.{p}f}'.format(vmin, vmax, p=prec)
        try:
            value = float(value)
        except:
            raise ValueError(error)

        if vmin <= value <= vmax:
            return value
        else:
            raise ValueError(error)

    def _format_property(self, name, value, unit=''):
        return '{:_<22.20} {:15} {:10}{}'.format(name + ':', value)
