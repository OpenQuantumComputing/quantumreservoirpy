from qreservoirpy import utilities
from .interface import ReservoirCircuit

from qiskit import QuantumRegister, ClassicalRegister
import numpy as np
from tqdm import tqdm



class QReservoir:
    def __init__(self, qubits, layers, analyze_function=lambda res:res, incrementally=False, M=np.inf, **kwargs) -> None:
        self.qreg = QuantumRegister(qubits, name='q')
        self.layers = layers

        self.incrementally = incrementally
        self.M = M

        self.kwargs = kwargs
        self.analyze_fcn = analyze_function


    def run(self, timeseries, shots=10000, transpile=False, simulator='aer_simulator_statevector'):
        len_timeseries = len(timeseries)

        if self.incrementally:
            timeseries = [
                timeseries[:i+1][-self.M:] for i in range(len_timeseries)
            ]
        else:
            timeseries = [timeseries]

        result = []
        with tqdm(total=len(timeseries)) as pbar:
            pbar.set_description("Simulating")
            for series in timeseries:
                circ = self.__build(series)

                mem = utilities.simulate(circ, shots, transpile, simulator)

                result.append(self.analyze_fcn(utilities.memory_to_mean(mem, 1)))

                pbar.update(1)


        return np.array(result).reshape((len_timeseries, -1))

    @property
    def circuit(self):
        return self.__build([0, 1, 0, 1])


    def __get_num_measurements(self, series):
        num_meas = 0
        for layer in self.layers:
            num_meas += layer.get_num_measurements(self.qreg, series)
        return num_meas


    def __build(self, series):
        ## There are probably more efficient ways of doing this
        num_meas = self.__get_num_measurements(series)
        creg = ClassicalRegister(num_meas)
        circ = ReservoirCircuit(self.qreg, creg)

        for layer in self.layers:
            circ = layer.build(circ, series, **self.kwargs)

        self.creq = ClassicalRegister(circ.RC_measured, name='c')
        newCirc = ReservoirCircuit(self.qreg, self.creq)

        for layer in self.layers:
            newCirc = layer.build(newCirc, series, **self.kwargs)
        return newCirc





