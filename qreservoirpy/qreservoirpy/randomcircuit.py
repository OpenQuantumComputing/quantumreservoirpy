# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


# NOTICE: copied and modified during the summer of 2023
# This code is based on qiskit's own implementation of random_circuit,
# but has been modified.

import numpy as np

from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit.circuit.library import standard_gates


def random_circuit(num_qubits, gates, depth, fillgrade=1):
    if num_qubits == 0:
        return QuantumCircuit()

    possible_gates = {
        # Gate class: (number of qubits, number of parameters)
        standard_gates.IGate: (1, 0),
        standard_gates.SXGate: (1, 0),
        standard_gates.XGate: (1, 0),
        standard_gates.RZGate: (1, 1),
        standard_gates.RGate: (1, 2),
        standard_gates.HGate: (1, 0),
        standard_gates.PhaseGate: (1, 1),
        standard_gates.RXGate: (1, 1),
        standard_gates.RYGate: (1, 1),
        standard_gates.SGate: (1, 0),
        standard_gates.SdgGate: (1, 0),
        standard_gates.SXdgGate: (1, 0),
        standard_gates.TGate: (1, 0),
        standard_gates.TdgGate: (1, 0),
        standard_gates.UGate: (1, 3),
        standard_gates.U1Gate: (1, 1),
        standard_gates.U2Gate: (1, 2),
        standard_gates.U3Gate: (1, 3),
        standard_gates.YGate: (1, 0),
        standard_gates.ZGate: (1, 0),
        standard_gates.CXGate: (2, 0),
        standard_gates.DCXGate: (2, 0),
        standard_gates.CHGate: (2, 0),
        standard_gates.CPhaseGate: (2, 1),
        standard_gates.CRXGate: (2, 1),
        standard_gates.CRYGate: (2, 1),
        standard_gates.CRZGate: (2, 1),
        standard_gates.CSXGate: (2, 0),
        standard_gates.CUGate: (2, 4),
        standard_gates.CU1Gate: (2, 1),
        standard_gates.CU3Gate: (2, 3),
        standard_gates.CYGate: (2, 0),
        standard_gates.CZGate: (2, 0),
        standard_gates.RXXGate: (2, 1),
        standard_gates.RYYGate: (2, 1),
        standard_gates.RZZGate: (2, 1),
        standard_gates.RZXGate: (2, 1),
        standard_gates.XXMinusYYGate: (2, 2),
        standard_gates.XXPlusYYGate: (2, 2),
        standard_gates.ECRGate: (2, 0),
        standard_gates.CSGate: (2, 0),
        standard_gates.CSdgGate: (2, 0),
        standard_gates.SwapGate: (2, 0),
        standard_gates.iSwapGate: (2, 0),
        standard_gates.CCXGate: (3, 0),
        standard_gates.CSwapGate: (3, 0),
        standard_gates.CCZGate: (3, 0),
        standard_gates.RCCXGate: (3, 0),
        standard_gates.C3SXGate: (4, 0),
        standard_gates.RC3XGate: (4, 0),
    }

    gates_1q = []
    extended_gates = []
    for gate in gates:
        n_q, n_p = possible_gates[gate]
        if n_q > num_qubits:
            continue

        extended_gates.append(
            (gate, n_q, n_p)
        )

        if n_q == 1:
            gates_1q.append(
                (gate, n_q, n_p)
            )


    gates = np.array(
        extended_gates, dtype=[("class", object), ("num_qubits", np.int64), ("num_params", np.int64)]
    )
    gates_1q = np.array(gates_1q, dtype=gates.dtype)

    qc = QuantumCircuit(num_qubits)

    seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    qubits = np.array(qc.qubits, dtype=object, copy=True)

    filling = np.random.random(size=depth)

    # Apply arbitrary random operations in layers across all qubits.
    for i in range(depth):
        # We generate all the randomness for the layer in one go, to avoid many separate calls to
        # the randomisation routines, which can be fairly slow.

        # This reliably draws too much randomness, but it's less expensive than looping over more
        # calls to the rng. After, trim it down by finding the point when we've used all the qubits.
        gate_specs = rng.choice(gates, size=len(qubits))
        cumulative_qubits = np.cumsum(gate_specs["num_qubits"], dtype=np.int64)
        # Efficiently find the point in the list where the total gates would use as many as
        # possible of, but not more than, the number of qubits in the layer.  If there's slack, fill
        # it with 1q gates.
        max_index = np.searchsorted(cumulative_qubits, num_qubits, side="right")
        gate_specs = gate_specs[:max_index]
        slack = num_qubits - cumulative_qubits[max_index - 1]
        if slack and filling[i] <  fillgrade:
            gate_specs = np.hstack((gate_specs, rng.choice(gates_1q, size=slack)))

        # For efficiency in the Python loop, this uses Numpy vectorisation to pre-calculate the
        # indices into the lists of qubits and parameters for every gate, and then suitably
        # randomises those lists.
        q_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
        p_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
        q_indices[0] = p_indices[0] = 0
        np.cumsum(gate_specs["num_qubits"], out=q_indices[1:])
        np.cumsum(gate_specs["num_params"], out=p_indices[1:])
        parameters = rng.uniform(0, 2 * np.pi, size=p_indices[-1])
        rng.shuffle(qubits)

        # We've now generated everything we're going to need.  Now just to add everything.  The
        # conditional check is outside the two loops to make the more common case of no conditionals
        # faster, since in Python we don't have a compiler to do this for us.

        for gate, q_start, q_end, p_start, p_end in zip(
            gate_specs["class"], q_indices[:-1], q_indices[1:], p_indices[:-1], p_indices[1:]
        ):
            operation = gate(*parameters[p_start:p_end])
            qc._append(CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end]))

    return qc
