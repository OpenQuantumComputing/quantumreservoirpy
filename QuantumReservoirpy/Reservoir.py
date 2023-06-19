import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary

from QuantumReservoirpy import utilities


class QReservoir:
    def __init__(self, qubits, layers) -> None:
        self.n_qubits = qubits
        self.layers = layers
        self.built = False

        self.config = {
            'operator': random_unitary(2**self.n_qubits),
        }

    def run(self, timeseries, operator=None, shots=10000):
        if operator is not None:
            self.config['operator'] = operator

        if not self.built:
            self.circuit = self.__build_circuit(timeseries)
        self.built = True

        memory = utilities.simulate(self.circuit, shots)
        features = self.config['features']


        result = utilities.memory_to_mean(memory, features)
        pred_idx = self.config.get('pred_idx', -1)
        if pred_idx > 0:
            return result[:pred_idx], result[pred_idx:]

        return result

    def circuit(self, timeseries=[0]):
        timeseries =  self.config.get('timeseries', timeseries)
        return self.__build_circuit(timeseries)

    def __build_circuit(self, timeseries):
        self.config['measured'] = 0
        self.config['timeseries'] = timeseries
        self.config['bitwidth'] = int(np.ceil(np.log2(np.max(timeseries) + 1))) # number of bits needed to encode largest element
        self.config['encoder'] = utilities.generate_encoder(self.config['bitwidth'])

        num_measurements = 0
        for layer in self.layers:
            num_measurements += layer.get_num_measurements(self.config)

        num_measurements = max(num_measurements, 2)

        circuit = QuantumCircuit(self.n_qubits, num_measurements)


        for layer in self.layers:
            circuit = layer.build(circuit, self.config)

        return circuit







