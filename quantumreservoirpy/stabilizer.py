from itertools import combinations, product
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, AncillaRegister
from qiskit.quantum_info import random_clifford, Clifford, Pauli
from qiskit.circuit.library import PauliEvolutionGate

from quantumreservoirpy.util import randomIsing, get_Ising_circuit
from quantumreservoirpy.reservoirs import Static

from typing import Iterable


class Stabilizer(Static):
    def __init__(
        self,
        n_qubits,
        n_meas, #number of stabilizer generators
        tableau: dict|None=None, #if specified, overrides the tableau generation
        codestate_preparation_circ: Iterable[QuantumCircuit]|None = None, #if None, will generate a random stabilizer code
        memory=np.inf,
        backend=None,
        degree=1,
        num_reservoirs=1,
        standard=False,
        isingparams=None,
        decode=True,# danger zone: this is only for testing
    ) -> None:
        super().__init__(
            n_qubits + 1, memory, backend, degree=degree, num_reservoirs=num_reservoirs
        )
        self.n_meas = n_meas
        self.decode = decode

        if not isingparams:
            steps = 1
            dt = 1.645
            #top = limitrange(list(combinations(range(n_qubits), 2)))
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

        if tableau != None:
            if len(tableau["stabilizer"])!=n_meas or len(tableau["destabilizer"])!=n_meas:
                raise Exception("Error: the number of entries of the tableau does not match the dimension of the stabilizer")
            self.tableau = tableau
        else:
            self.tableau = Stabilizer.generate_tableau(n_qubits, n_meas, codestate_preparation_circ)

        # self.cs = Stabilizer.get_stabilizer_circuits(n_qubits, n_meas, self.tableau, random=False, standard=standard)
        # self.decodermap = Stabilizer.build_decoder_map(n_meas + 1, standard=standard)


    def before(self, circuit):
        if self.decode:
            # circuit.add_register(ClassicalRegister(self.n_meas))
            circuit.add_register(AncillaRegister(self.n_meas))

    def during(self, circuit, timestep, reservoirnumber):
        circuit.barrier()

        # encode
        for k in range(self.n_meas):
            beta = 3**k
            pauliop = Pauli(self.tableau["destabilizer"][k])
            evo = PauliEvolutionGate(pauliop, -beta / 2 * np.pi * timestep)
            circuit.append(evo, range(self.n_qubits - 1))
            # circuit.rx(-beta / 2 * np.pi * timestep, k) #encodes evolution of destabilizer for each bit of the syndrome measurement bitstring
        # circuit.rx(np.pi * timestep, 0)
        # print("encode")
        # print(circuit)

        # reservoir
        circuit.append(self.U[reservoirnumber], range(self.n_qubits - 1))
        # print("reservoir")
        # print(circuit)

        # decode
        cr = ClassicalRegister(self.n_meas)
        circuit.add_register(cr)
        # circuit.barrier()
        # for j in range(self.n_meas):
        #     circuit.append(self.cs[j], range(self.n_qubits))
        #     circuit.measure(circuit.qubits[self.n_qubits - 1], cr[j])
        #     circuit.barrier()

        if self.decode:
            # Stabilizer.apply_operations_for_integers(circuit, cr, self.decodermap)
            # ar = circuit.ancillas
            # circuit.reset(ar)
            Stabilizer.decoder(circuit, self.tableau)
            
    @staticmethod
    def generate_tableau(n_qubits: int, n_meas: int, codestate_preparation_circ: Iterable[QuantumCircuit]|None=None):
        """generates a tableau for a stabilizer code based on 2**k codestate preparation circuits"""

        logical_qubits = n_qubits - n_meas #also called k

        if codestate_preparation_circ==None: #generate random stabilizer code
            tableau = random_clifford(n_qubits).to_dict()

            #turn the last k stabilizers into logical Zs
            # tableau["logical_z"] = tableau["stabilizer"][n_meas:] #these are just for QEC fun, not useful here
            tableau["stabilizer"] = tableau["stabilizer"][:n_meas]

            #turn the last k destabilizers into logical Xs
            # tableau["logical_x"] = tableau["destabilizer"][n_meas:]
            tableau["destabilizer"] = tableau["destabilizer"][:n_meas]

        elif len(codestate_preparation_circ)!=2**logical_qubits:
            print("Error : number of code state preparation circuits does not match the dimension of the codespace")
            return

        else:
            tableau = Clifford(codestate_preparation_circ[0]).to_dict()
            for circ in codestate_preparation_circ[1:]:
                circ_tableau = Clifford(circ).to_dict()
                to_pop = []

                for i in range(len(tableau["stabilizer"])):
                    if tableau["stabilizer"][i] not in circ_tableau["stabilizer"]:
                        to_pop.append(i)

                for i in to_pop:
                    tableau["stabilizer"].pop(i)
                    tableau["destabilizer"].pop(i)
                # tableau["stabilizer"] = [stab for stab in tableau["stabilizer"] if stab in circ_tableau["stabilizer"]]
                # tableau["destabilizer"] = [stab for stab in tableau["destabilizer"] if stab in circ_tableau["destabilizer"]]
            print(tableau)
            
            #check the stabilizer has the right dimension
            if len(tableau["stabilizer"])!=n_meas or len(tableau["destabilizer"])!=n_meas:
                print("Error : something went wrong with tableau generation")
                print(tableau)
        
        return tableau
        

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

    # @staticmethod
    # def get_stabilizer_circuits(n, m, tableau, random=True, standard=False):
    #     circuits = []
    #     # if random:
    #         # rs = random_clifford(n).stabilizer

    #     for P in tableau["stabilizers"]:
    #         q = QuantumRegister(n + 1)
    #         circuit = QuantumCircuit(q)
    #         circuit.reset([n])
    #         circuit.h(n)
    #         for i in range(1, len(P)):
    #             if P[i] == "X":
    #                 circuit.cx(q[n], q[i - 1])
    #             elif P[i] == "Y":
    #                 circuit.cy(q[n], q[i - 1])
    #             elif P[i] == "Z":
    #                 circuit.cz(q[n], q[i - 1])
    #         circuit.h(n)
    #         circuits.append(circuit)
    #     elif standard:
    #         for j in range(m):
    #             q = QuantumRegister(n + 1)
    #             circuit = QuantumCircuit(q)
    #             circuit.reset([n])
    #             circuit.h(n)
    #             circuit.cz(q[n], q[j])
    #             circuit.h(n)
    #             circuits.append(circuit)
    #     else:
    #         for j in range(m):
    #             q = QuantumRegister(n + 1)
    #             circuit = QuantumCircuit(q)
    #             circuit.reset([n])
    #             circuit.h(n)
    #             circuit.cz(q[n], q[j])
    #             circuit.cz(q[n], q[j + 1])
    #             circuit.h(n)
    #             circuits.append(circuit)
    #     return circuits

    @staticmethod
    def apply_operations_for_integers(circ, c, integer_dict):
        with circ.switch(c) as case:
            for key, value in integer_dict.items():
                if value:
                    with case(key):
                        circ.x(value)

    @staticmethod
    def decoder(circuit: QuantumCircuit, code_tableau: dict):
        """
        Given a n-qubit state and a stabilizer code, detects errors and applies the operations
        to project the state back to the codespace.

        Args:
            circuit : state preparation circuit
            code_tableau : dictionary {"stabilizer": [], "destabilizer": []} each list has n-k elements

        Returns:
            circuit : state preparation circuit with syndrome measurement and error correction appended
        """

        n_qubits = circuit.num_qubits
        n_meas = len(code_tableau["stabilizer"])

        qr = circuit.qregs[0]        
        cr = circuit.cregs[-1]
        ar = circuit.ancillas

        circuit.barrier()

        #syndrome measurement operations
        circuit.reset(ar)
        circuit.h(ar)
        
        for j in range(n_meas):
            P = code_tableau["stabilizer"][j]
            for i in range(1, len(P)):
                if P[i] == "X":
                    circuit.cx(ar[j], qr[i - 1])
                elif P[i] == "Y":
                    circuit.cy(ar[j], qr[i - 1])
                elif P[i] == "Z":
                    circuit.cz(ar[j], qr[i - 1])

        circuit.h(ar)

        for j in range(n_meas):
            circuit.measure(ar[j], cr[j])
            circuit.barrier()

        for j in range(n_meas):
            with circuit.if_test((cr[j], 1)):
                circuit.pauli(code_tableau["destabilizer"][j][1:], qr[:-1])
        
        return circuit