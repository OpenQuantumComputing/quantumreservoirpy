

# import numpy as np
# import matplotlib.pyplot as plt
# from qiskit import QuantumCircuit
# from qiskit.quantum_info import random_unitary
# from QuantumReservoirpy.utilities import gen_controlled_unitary, add_controlled_unitary




# for sys in subsystems:
#     sys.draw('mpl')
# plt.show()

# plt.show()

# memory_qubits = [0]
# control_qubits = [1]



# n_mem = len(memory_qubits)
# n_enc = len(control_qubits)
# _, c_op1 = gen_controlled_unitary(n_mem, n_enc, r'$U_1$')
# op2, c_op2 = gen_controlled_unitary(n_mem, n_enc, r'$U_2$')

# circ = QuantumCircuit(n_mem + n_enc + 1)
# print(op2)

# timeseries = [0, 1, 0, 0, 1, 0, 1]
# for u in timeseries:
#     circ.initialize([np.sqrt(u), np.sqrt(1-u)], control_qubits, )
#     circ.append(c_op1, control_qubits + memory_qubits)
#     circ.unitary(op2.conjugate(), memory_qubits, label=r'${U_2}^T$')
#     circ.append(c_op2, control_qubits + memory_qubits)

# circ.draw('mpl')
# plt.show()

