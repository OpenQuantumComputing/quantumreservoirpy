

def test_incremental():
    import numpy as np
    from qiskit.quantum_info import random_unitary
    from quantumreservoirpy.reservoirs import  Incremental

    encoder = {0: "00", 1: "01", 2: "10", 3: "11"}
    SHOTS = 100


    class RandomUnitary(Incremental):
        def __init__(self, n_qubits, memory=np.inf, backend=None, num_features=8) -> None:
            super().__init__(n_qubits, memory, backend, num_features)
            self.operator = random_unitary(2**n_qubits)

        def before(self, circuit):
            circuit.h(circuit.qubits)

        def during(self, circuit, timestep, reservoirnumber):
            circuit.measure([0, 1])
            circuit.initialize(encoder[timestep], [0, 1])
            circuit.append(self.operator, circuit.qubits)

        def after(self, circuit):
            circuit.measure_all()


    res = RandomUnitary(n_qubits=4, memory=8)

    timestep = [0, 1, 2, 3, 0, 1, 2, 2, 3]
    timeseries = timestep * 10

    res.run(timeseries, shots=SHOTS)