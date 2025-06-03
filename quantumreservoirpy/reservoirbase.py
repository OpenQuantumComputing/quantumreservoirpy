import warnings
import numpy as np

from quantumreservoirpy.reservoircircuit import ReservoirCircuit, CountingCircuit

from abc import ABC, abstractmethod

from qiskit_aer import AerSimulator
from qiskit.compiler import transpile as qtranspile
from qiskit.providers import Backend


class BaseReservoir(ABC):
    """
    Interface for custom reservoirs. Please note that one should NOT
    inherit from this class directly, but rather through one of the subclasses in reservoir.py.

    The reservoir creation is divided into 3 steps: Before, during and after.
    """
    def before(self, circuit):
        """ Specify what operations should be applied to the circuit before
        the timeseries dependent operators are added.

        The function should take in the circuit, apply the necessary operations,
        and return the circuit again

        Args:
            circuit: qiskit.circuit

        Returns:
            circuit: qiskit.circuit
        """
        pass

    # What to do every timestep
    def during(self, circuit, timestep, reservoir_number):
        """ Apply operators to the circuit at each timestep. Add reservoir number
        to specify multiple different reservoirs.

        The internal engine loops through the timesteps when run or predict is called,
        and appends to the circuit what is specified here for each timestep.

        Args:
            circuit: qiskit.circuit
            timestep (Iterable[int]): Iterable of timesteps
            reservoir_number (int): Useful to specify many different reservoirs

        Returns:
            circuit: qiskit.circuit
        """
        pass

    # What should happen after the timeseries
    def after(self, circuit):
        """ Apply operations after all timeseries dependent operators are added.
        Common usecase is to measure all qubits.

        The function should take in the circuit, apply the necessary operations,
        and return the circuit again

        Args:
            circuit: qiskit.circuit
        Returns:
            circuit: qiskit.circuit
        """
        pass


class QReservoir(BaseReservoir):
    def __init__(
        self, n_qubits, memory=np.inf, backend=None, degree=1, num_reservoirs=1
    ) -> None:
        try:
            if not issubclass(type(backend), Backend):
                raise TypeError()
            self.backend = backend
        except:
            warnings.warn(
                "Argument 'backend' must inherit from qiskits abstract 'Backend'. Using 'aer_simulator' instead"
            )
            self.backend = AerSimulator()

        self.n_qubits = n_qubits
        self.memory = memory
        self.degree = degree
        self.num_reservoirs = num_reservoirs

    @abstractmethod
    def run(self, timeseries, **kwargs):
        """ Runs the timeseries through the reservoir, and returns the reservoir
        state at each timestamp of the timeseries. Kwargs depend on subclass.

        Args:
            timeseries (Iterable[int]): Iterable of timestamps

        Returns:
            np.ndarray: Internal reservoir state at each timestamp
        """

    @abstractmethod
    def predict(self, num_pred, model, from_series, **kwargs):
        """ Use the trained model for timeseries predictions

        Args:
            num_pred (int): Number of timesteps to predict
            model: Ready trained model. Should implement model.predict
            from_series (Iterable[int]): Iterable of timestamps
        """
        pass

    def circuit(
        self, timeseries, merge_registers=False, transpile=True, reservoir_number=1
    ):
        """ Build the qiskit circuit the reservoir represents from a given timeseries.

        Args:
            timeseries: Iterable of timestamps
            merge_registers (bool, optional): Whether or not the measurement registers should be merged
            into one. Defaults to False.
            transpile (bool, optional): Perform qiskit transpilation. Defaults to True.
            reservoir_number (int, optional): Parameter that may be used in the creation process. Defaults to 1.

        Returns:
            qiskit circuit
        """
        circ = ReservoirCircuit(self.n_qubits)
        circ = self.__build(circ, timeseries, reservoir_number)

        if merge_registers:
            temp_circ = CountingCircuit(self.n_qubits, circ.num_clbits)
            circ = self.__build(temp_circ, timeseries, reservoir_number)
        if transpile:
            circ = qtranspile(circ, self.backend)
        return circ

    @property
    def job(self):
        try:
            return getattr(self, "_job")
        except AttributeError as e:
            raise AttributeError(str(e) + ". Try runnning the reservoir first")

    def __build(self, circ, timeseries, reservoirnumber):
        self.before(circ)
        for t in timeseries:
            self.during(circ, t, reservoirnumber)
        self.after(circ)

        return circ
