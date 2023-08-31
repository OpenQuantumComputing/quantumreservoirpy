from typing import Optional, Sequence
import matplotlib.pyplot as plt
from qiskit import Aer, QuantumCircuit, ClassicalRegister
from qiskit.circuit.bit import Bit
from qiskit.circuit.instructionset import InstructionSet
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumcircuit import ClbitSpecifier, QuantumCircuit, QubitSpecifier
from qiskit.circuit.register import Register
from qiskit.providers import Backend

from tqdm import tqdm

from .util import listify, memory_to_mean
import numpy as np

class InterfaceCircuit(QuantumCircuit):
    def measure_all(self):
        return self.measure(self.qubits)

class ReservoirCircuit(InterfaceCircuit):
    def measure(self, qubit):
        cr = ClassicalRegister(len(listify(qubit)))
        self.add_register(cr)
        return super().measure(qubit, cr)


class CountingCircuit(InterfaceCircuit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0
        self.idx = 0

    def measure(self, qubit: QubitSpecifier) -> InstructionSet:
        num_meas = len(listify(qubit))
        instr = super().measure(qubit,self.count +  np.arange(num_meas, dtype=int))
        self.count += num_meas
        return instr

from abc import ABC, abstractmethod

class BaseReservoir(ABC):
    """
    Interface for custom reservoirs. Please note that one should NOT
    inherit from this class directly, but rather through one of the
    subclasses in reservoir.py.
    """

    def before(self, circuit):
        pass

    # What to do every timestep
    def during(self, circuit, timestep):
        pass

    # What should happen after the timeseries
    def after(self, circuit):
        pass



class QReservoir(BaseReservoir):
    def __init__(self, n_qubits, backend=Aer.get_backend('aer_simulator'), memory=np.inf) -> None:
        if not issubclass(type(backend), Backend):
            raise TypeError("Argument 'backend' must inherit from qiskits abstract 'Backend'")

        self.n_qubits = n_qubits
        self.backend = backend
        self.memory = memory

    def run(self, timeseries, **kwargs):
        pass

    def predict(self, num_pred, model, from_series, **kwargs):
        pass


    def circuit(self, timeseries, merge_registers=False):
        circ = ReservoirCircuit(self.n_qubits)
        circ = self.__build(circ, timeseries)

        if merge_registers:
            temp_circ = CountingCircuit(self.n_qubits, circ.num_clbits)
            circ = self.__build(temp_circ, timeseries)
        return circ


    @property
    def job(self):
        try:
            return getattr(self, "_job")
        except AttributeError as e:
            raise AttributeError(str(e) +  ". Try runnning the reservoir first")


    def __build(self, circ, timeseries):
        self.before(circ)
        for t in timeseries:
            self.during(circ, t)

        self.after(circ)

        return circ

