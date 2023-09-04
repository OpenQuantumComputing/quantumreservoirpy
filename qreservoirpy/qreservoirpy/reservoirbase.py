import warnings
import numpy as np

from .reservoircircuit import ReservoirCircuit, CountingCircuit

from abc import ABC, abstractmethod
import qiskit as qs
from qiskit import Aer
from qiskit.providers import Backend


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
    def __init__(self, n_qubits,  memory=np.inf, backend=None) -> None:
        try:
            if not issubclass(type(backend), Backend):
                raise TypeError()
            self.backend = backend
        except:
            warnings.warn("Argument 'backend' must inherit from qiskits abstract 'Backend'. Using 'aer_simulator' instead")
            self.backend=Aer.get_backend('aer_simulator')

        self.n_qubits = n_qubits
        self.memory = memory

    @abstractmethod
    def run(self, timeseries, **kwargs):
        pass

    @abstractmethod
    def predict(self, num_pred, model, from_series, **kwargs):
        pass


    def circuit(self, timeseries, merge_registers=False, transpile=True):
        circ = ReservoirCircuit(self.n_qubits)
        circ = self.__build(circ, timeseries)

        if merge_registers:
            temp_circ = CountingCircuit(self.n_qubits, circ.num_clbits)
            circ = self.__build(temp_circ, timeseries)
        if transpile:
            circ = qs.transpile(circ, self.backend)
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





