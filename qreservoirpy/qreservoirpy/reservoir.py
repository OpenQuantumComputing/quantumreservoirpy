from qreservoirpy import utilities
from .interface import ReservoirCircuit

from qiskit import QuantumRegister, ClassicalRegister
import numpy as np




class QReservoir:
    def __init__(self, qubits, layers, analyze_function=lambda res: res, **kwargs) -> None:
        self.qreg = QuantumRegister(qubits, name='q')
        self.layers = layers

        self.total_runned = 0
        self.states = []

        self.kwargs = kwargs
        self.analyze_fcn = analyze_function

    def run(self, timeseries, shots=10000):
        self.total_runned += len(timeseries)
        while len(timeseries) > 0:
            timeseries = self.add_timeseries(timeseries)
            circ = self.__build()
            mem = utilities.simulate(circ, shots)
            self.states.append(self.analyze_fcn(utilities.memory_to_mean(mem, 1)))
        self.states = np.array(self.states)
        self.states = self.states.reshape((self.total_runned, -1))
        return self.states

    def add_timeseries(self, timeseries):
        for layer in self.layers:
            timeseries = layer.add_timeseries(timeseries)
        return timeseries

    @property
    def circuit(self):
        return self.__build()

    # @property
    # def states(self):
    #     return np.array(self.states)

    def __get_num_measurements(self):
        num_meas = 0
        for layer in self.layers:
            num_meas += layer.get_num_measurements(self.qreg)
        return num_meas

    def __build(self):
        num_meas = self.__get_num_measurements()
        creg = ClassicalRegister(num_meas)
        circ = ReservoirCircuit(self.qreg, creg)

        for layer in self.layers:
            circ = layer.build(circ, **self.kwargs)

        self.creq = ClassicalRegister(circ.RC_measured, name='c')
        newCirc = ReservoirCircuit(self.qreg, self.creq)

        for layer in self.layers:
            newCirc = layer.build(newCirc, **self.kwargs)
        return newCirc





