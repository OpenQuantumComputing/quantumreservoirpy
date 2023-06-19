from qreservoirpy import utilities
from qiskit import QuantumCircuit
from qiskit.circuit.instructionset import InstructionSet
from qiskit.circuit.quantumcircuit import QubitSpecifier

class ReservoirCircuit(QuantumCircuit):
    def __init__(self, *args, **kwargs):
        self.RC_measured = 0
        self.num_meas = 0
        super().__init__(*args, **kwargs)

        self.reverse_bits()

    def measure(self, qubit: QubitSpecifier) -> InstructionSet:
        qubit = utilities.listify(qubit)
        self.RC_measured += len(qubit)
        return super().measure(qubit=qubit,cbit=self.clbits[self.RC_measured-len(qubit):self.RC_measured])



