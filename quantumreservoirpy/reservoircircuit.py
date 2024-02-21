import numpy as np

from util import listify

from qiskit import QuantumCircuit, ClassicalRegister


class InterfaceCircuit(QuantumCircuit):
    def measure_all(self):
        return self.measure(self.qubits)


class ReservoirCircuit(InterfaceCircuit):
    def measure(self, qubit, clbit=None):
        if clbit:
            return super().measure(qubit, clbit)
        else:
            cr = ClassicalRegister(len(listify(qubit)))
            self.add_register(cr)
            return super().measure(qubit, cr)


class CountingCircuit(InterfaceCircuit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0
        self.idx = 0

    def measure(self, qubit):
        num_meas = len(listify(qubit))
        instr = super().measure(qubit, self.count + np.arange(num_meas, dtype=int))
        self.count += num_meas
        return instr
