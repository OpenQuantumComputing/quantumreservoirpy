import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile, Aer
from qiskit.quantum_info import random_unitary
from abc import ABC, abstractmethod #For the abstract Layer base class

class Layer(ABC):
    @abstractmethod
    def get_num_measurements(self, config):
        pass

    @abstractmethod
    def build(self, circuit, config):
        return circuit


class H(Layer):
    def __init__(self, num_layers=1) -> None:
        super().__init__()
        self.n_layers = num_layers

    def get_num_measurements(self, config):
        return 0

    def build(self, circuit, config):
        for _ in range(self.n_layers):
            for i in range(circuit.num_qubits):
                circuit.h(i)
        return circuit

class Warmup(Layer):
    def __init__(self, num_layers=5) -> None:
        super().__init__()
        self.n_layers = num_layers

    def get_num_measurements(self, config):
        return 0

    def build(self, circuit, config):
        key = 'operator'
        if key not in config:
            raise Exception("Unitary operator must be passed to warmup layer through config, key 'operator'")
        op = config['operator']

        for _ in range(self.n_layers):
            circuit.append(op, [i for i in range(circuit.num_qubits)])
        return circuit

class Timeseries(Layer):
    # measure_qubits are the indeces of the qubits that are measured every timestep.
    # timeseries_qubits are the indeces of the qubits used to encode the series into the circuit.
    # distance controls the numebr of measurements made between encodings.
    def __init__(self, measure_qubits=[0], timeseries_qubits=[], distance=0) -> None:

        super().__init__()

        self.measure_qubits = measure_qubits
        self.timeseries_qubits = timeseries_qubits
        self.distance = distance
        if len(timeseries_qubits) == 0:
            self.timeseries_qubits = measure_qubits

        self.n_features = len(self.measure_qubits)  + len(self.measure_qubits) * self.distance


    def get_num_measurements(self, config):
        config['features'] = self.n_features

        key_timeseries = 'timeseries'
        if key_timeseries not in config:
            raise Exception("Need timeseries in config to calculate the total measurements needed")

        num_measurements = self.n_features * (len(config[key_timeseries]) - 1)
        return num_measurements


    def build(self, circuit, config):

        try:
            timeseries = config['timeseries']
            encoder = config['encoder']
            operator = config['operator']
            bitwidth = config['bitwidth']

        except KeyError:
            raise Exception("Not all config-variables needed are present in config")

        ## Used by prediction layer. Placed here to ensure prediction build AFTER timeseries build
        config['distance'] = self.distance
        config['measure_qubits'] = self.measure_qubits
        ##

        measured = config.get('measured', 0)
        assert(bitwidth == len(self.timeseries_qubits)) #Need to measure more qubits to encode the timeseries")

        for idx, time in enumerate(timeseries):
            circuit.initialize(encoder[time], self.timeseries_qubits)

            if idx < len(timeseries) - 1:
                circuit.append(operator, [i for i in range(circuit.num_qubits)])
                circuit.measure(self.measure_qubits, measured + np.arange(len(self.measure_qubits)))
                measured += len(self.measure_qubits)
                for _ in range(self.distance):
                    circuit.append(operator, [i for i in range(circuit.num_qubits)])
                    circuit.measure(self.measure_qubits, measured + np.arange(len(self.measure_qubits)))
                    measured += len(self.measure_qubits)
        circuit.barrier(np.arange(circuit.num_qubits))
        config['measured'] = measured
        return circuit

class Predict(Layer):
    def __init__(self, num_predictions=1) -> None:
        super().__init__()
        self.n_predictions = num_predictions

    def get_num_measurements(self, config):
        self.n_features = config['features']
        return self.n_features * self.n_predictions

    def build(self, circuit, config):
        try:
            operator = config['operator']
            distance = config['distance']
            measure_qubits = config['measure_qubits']
        except KeyError:
            raise Exception("Not all config-variables needed are present in config")

        measured = config.get('measured', 0)
        config['pred_idx'] = measured // self.n_features

        for _ in range(self.n_predictions):
            circuit.append(operator, [i for i in range(circuit.num_qubits)])
            circuit.measure(measure_qubits, measured + np.arange(len(measure_qubits)))
            measured += len(measure_qubits)
            for _ in range(distance):
                circuit.append(operator, [i for i in range(circuit.num_qubits)])
                circuit.measure(measure_qubits, measured + np.arange(len(measure_qubits)))
                measured += len(measure_qubits)
                circuit.barrier(np.arange(circuit.num_qubits))
        config['measured'] = measured
        return circuit


class Measurement(Layer):
    def __init__(self, measure_qubits) -> None:
        super().__init__()
        self.measure_qubits = measure_qubits

    def get_num_measurements(self, config):
        return len(self.measure_qubits)

    def build(self, circuit, config):
        measured = config.get('measured', 0)
        circuit.measure(self.measure_qubits, measured + np.arange(len(self.measure_qubits)))
        measured += len(self.measure_qubits)
        config['measured'] = measured
        return circuit
