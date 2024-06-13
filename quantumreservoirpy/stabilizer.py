from itertools import combinations, product
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

from quantumreservoirpy.util import randomIsing, get_Ising_circuit
from quantumreservoirpy.reservoirs import Static


class Stabilizer(Static):
    def __init__(
        self,
        n_qubits,
        n_meas,
        memory=np.inf,
        backend=None,
        degree=1,
        num_reservoirs=1,
        standard=False,
        isingparams=None,
    ) -> None:
        super().__init__(
            n_qubits + 1, memory, backend, degree=degree, num_reservoirs=num_reservoirs
        )
        self.n_meas = n_meas

        if not isingparams:
            steps = 1
            dt = 1.645
            top = list(combinations(range(n_qubits), 2))
            self.U = {}
            self.isingparams = {}
            for nr in range(1, num_reservoirs + 1):
                (
                    self.U[nr],
                    self.isingparams[nr],
                ) = randomIsing(n_qubits, top, steps, dt)
        else:
            self.U = {}
            for nr in range(1, num_reservoirs + 1):
                self.U[nr] = get_Ising_circuit(n_qubits, isingparams[nr])

        self.cs = Stabilizer.get_stabilizer_circuits(n_qubits, n_meas, random=False, standard=standard)
        self.decodermap = Stabilizer.build_decoder_map(n_meas + 1, standard=standard)

    def during(self, circuit, timestep, reservoirnumber):
        circuit.barrier()

        # encode
        for k in range(self.n_meas):
            beta = 3**k
            circuit.rx(-beta / 2 * np.pi * timestep, k)
        # circuit.rx(np.pi * timestep, 0)

        # reservoir
        circuit.append(self.U[reservoirnumber], range(self.n_qubits - 1))

        # decode
        cr = ClassicalRegister(self.n_meas)
        circuit.add_register(cr)
        circuit.barrier()
        for j in range(self.n_meas):
            circuit.append(self.cs[j], range(self.n_qubits))
            circuit.measure(circuit.qubits[self.n_qubits - 1], cr[j])
            circuit.barrier()

        Stabilizer.apply_operations_for_integers(circuit, cr, self.decodermap)

    @staticmethod
    def binary_array_to_integer(binary_array):
        binary_string = "".join(map(str, binary_array.astype(int)))
        decimal_integer = int(binary_string, 2)
        return decimal_integer

    @staticmethod
    def indices_of_ones(input_list, n):
        indices = [n-1-index for index, value in enumerate(input_list) if value == 1.0]
        return indices # n-1-index because of Endian encoding of qiskit

    @staticmethod
    def get_parity_measurements(x):
        G_list = []
        for i in range(x.shape[0] - 1):
            tmp = np.zeros(x.shape[0])
            tmp[i] = 1
            tmp[i + 1] = 1
            G_list.append(tmp)
        G = np.array(G_list)
        return np.remainder(G @ x, 2)

    @staticmethod
    def build_decoder_map(n, standard):
        decoder = {}
        if standard:
            for origin in list(product((0, 1), repeat=n-1)):
                p = np.array(origin)
                p = Stabilizer.binary_array_to_integer(p)
                flips = Stabilizer.indices_of_ones(origin, n-1)
                decoder[p] = flips
        else:
            for origin in list(product((0, 1), repeat=n)):
                p = Stabilizer.get_parity_measurements(np.array(origin))
                p = Stabilizer.binary_array_to_integer(p)
                flips = Stabilizer.indices_of_ones(origin, n)
                if p in decoder:
                    if len(decoder[p]) > len(flips):
                        decoder[p] = flips
                else:
                    decoder[p] = flips
        return decoder

    @staticmethod
    def get_stabilizer_circuits(n, m, random=True, standard=False):
        circuits = []
        if random:
            rs = random_clifford(n).stabilizer

            for P in sample(rs.to_labels(), m):
                q = QuantumRegister(n + 1)
                circuit = QuantumCircuit(q)
                circuit.reset([n])
                circuit.h(n)
                for i in range(1, len(P)):
                    if P[i] == "X":
                        circuit.cx(q[n], q[i - 1])
                    elif P[i] == "Y":
                        circuit.cy(q[n], q[i - 1])
                    elif P[i] == "Z":
                        circuit.cz(q[n], q[i - 1])
                circuit.h(n)
                circuits.append(circuit)
        elif standard:
            for j in range(m):
                q = QuantumRegister(n + 1)
                circuit = QuantumCircuit(q)
                circuit.reset([n])
                circuit.h(n)
                circuit.cz(q[n], q[j])
                circuit.h(n)
                circuits.append(circuit)
        else:
            for j in range(m):
                q = QuantumRegister(n + 1)
                circuit = QuantumCircuit(q)
                circuit.reset([n])
                circuit.h(n)
                circuit.cz(q[n], q[j])
                circuit.cz(q[n], q[j + 1])
                circuit.h(n)
                circuits.append(circuit)
        return circuits

    @staticmethod
    def apply_operations_for_integers(circ, c, integer_dict):
        with circ.switch(c) as case:
            for key, value in integer_dict.items():
                if value:
                    with case(key):
                        circ.x(value)
