from qreservoirpy import utilities
from .interface import ReservoirCircuit

from qiskit import QuantumRegister, ClassicalRegister
import numpy as np
from tqdm import tqdm



class QReservoir:
    def __init__(self, qubits, layers, analyze_function=lambda res:res,  M=np.inf, **kwargs) -> None:
        self.qreg = QuantumRegister(qubits, name='q')
        self.layers = layers

        self.M = M

        self.kwargs = kwargs
        self.analyze_fcn = analyze_function

        self.n_features = -1


        self.timeseries = []

    def predict(self, num_pred, model, from_series, shots=10000, low=-np.inf, high=np.inf):
        M = min(self.M, len(from_series))
        pred_series = from_series[-M:]

        states = []
        total = int(num_pred * M + num_pred * (num_pred + 1) / 2)
        with tqdm(total=total, desc="Predicting") as pbar:
            for _ in range(num_pred):
                state = self.run(pred_series, incrementally=False, shots=shots, disable_status_bar=True).reshape((-1, self.n_features))
                state = state[-1].reshape((1, -1))
                states.append(state)

                pred = model.predict(state)
                pred = min(pred, high)
                pred = max(pred, low)

                pred_series = np.append(pred_series, pred)
                pbar.update(len(pred_series))

        return np.array(states).reshape((num_pred, -1)), pred_series[-num_pred:]

    def run(self, timeseries, shots=10000, transpile=False, incrementally=False, disable_status_bar=False, simulator='aer_simulator_statevector'):
        len_timeseries = len(timeseries)

        M = min(self.M, len_timeseries)
        if incrementally:
            timeseries = [
                timeseries[:i+1][-M:] for i in range(len_timeseries)
            ]
        else:
            timeseries = [timeseries[-M:]]

        self.timeseries = timeseries[-1]

        result = []
        for series in tqdm(timeseries, desc="Simulating", disable=disable_status_bar):

            circ = self.__build(series)

            mem = utilities.simulate(circ, shots, transpile, simulator)

            result.append(self.analyze_fcn(utilities.memory_to_mean(mem, 1)))

        result = np.array(result)
        if self.n_features < 0:
            res = result.reshape((len_timeseries, -1))
            self.n_features = res.shape[-1]
            return res
        return result.reshape((-1, self.n_features))



    @property
    def circuit(self):
        if len(self.timeseries) == 0:
            return self.__build([0, 1])
        return self.__build(self.timeseries)


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





