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


class BaseReservoir:
    """
    Interface for custom reservoirs. Please note that one should NOT
    inherit from this class directly, but rather through the subclass
    QReservoir.
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
    def __init__(self, n_qubits, backend=Aer.get_backend('aer_simulator')) -> None:
        if not issubclass(type(backend), Backend):
            raise TypeError("Argument 'backend' must inherit from qiskits abstract 'Backend'")

        self.n_qubits = n_qubits
        self.backend = backend


    def run(self, timeseries, analyze=True, memory=True, **kwargs):
        circ = self.circuit(timeseries, merge_registers=False).reverse_bits()
        self._job = self.backend.run(circ, memory=memory, **kwargs)

        if not analyze:
            return

        mem = self._job.result().get_memory()
        avg = memory_to_mean(mem)

        idx = len(avg) - self.post_meas
        during_data = avg[:idx]
        after_data = avg[idx:].reshape((1, -1))

        if during_data.size == 0:
            return after_data

        during_data = during_data.reshape((len(timeseries), -1))
        if after_data.size == 0:
            return during_data

        return during_data, after_data

    def predict(self, num_pred, model, from_series, analyze_fcn=lambda v : v[-1], **kwargs):
        predictions = np.zeros(num_pred + len(from_series))
        predictions[:len(from_series)] = from_series

        for i in tqdm(range(num_pred), desc="Predicting..."):
            curidx = len(from_series) + i
            states = self.run(predictions[:curidx], kwargs=kwargs)
            pred_state = analyze_fcn(states).reshape((1, -1))
            predictions[curidx] = model.predict(pred_state)

        return predictions[-num_pred:]



    def run_incrementally(self, timeseries, M=np.inf, analyze=True, **kwargs):

        M = min(len(timeseries), M)
        timeseries_splited = [timeseries[:i+1][-M:] for i in range(len(timeseries))]

        total = 0
        if analyze:
            total = len(timeseries_splited)

        with tqdm(total=total) as pbar:
            pbar.set_description("Creating circuits...")
            circuits = [self.circuit(series, merge_registers=False).reverse_bits() for series in timeseries_splited]

            pbar.set_description("Running job...")
            self._job = self.backend.run(circuits, memory=analyze, **kwargs)

            if not analyze:
                return


            result = self._job.result()
            num_features = memory_to_mean(result.get_memory(0)).size
            states = np.zeros((len(timeseries_splited), num_features))


            pbar.set_description("Analyzing... ")
            for idx, _ in enumerate(timeseries_splited):
                memory = self._job.result().get_memory(idx)
                states[idx] = memory_to_mean(memory)
                pbar.update(1)

            return states


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

        before = circ.num_clbits
        self.after(circ)
        after = circ.num_clbits

        self.post_meas = after - before
        return circ



encoding = {
    0: '00',
    1: '01',
    2: '10',
    3: '11'
}

from qiskit.quantum_info import random_unitary
class BRes(QReservoir):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.op = random_unitary(2 ** self.n_qubits)

    def before(self, circuit):
        for qbit in circuit.qubits:
            circuit.initialize('+', qbit)

    def during(self, circuit, timestep):
        circuit.barrier()
        for i in range(2):
            circuit.initialize('+', 0)
            circuit.cx(0, i+1)
            # circuit.measure([0, 1])
        circuit.append(self.op, circuit.qubits)
        circuit.initialize(encoding[timestep], [0, 1])

    def after(self, circuit):
        circuit.measure_all()





from qiskit_aer import AerSimulator
res = BRes(n_qubits=4, backend=AerSimulator())


# res.circuit([0, 1, 2] * 4, merge_registers=True).draw('mpl')

states = res.run_incrementally([0, 1, 2]*10, M=20, shots=100)

# print(states)
# import matplotlib.pyplot as plt
# res.__build([0, 1, 2, 3]).draw('mpl')

plt.show()

