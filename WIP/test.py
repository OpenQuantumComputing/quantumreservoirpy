import numpy as np
# import matplotlib.pyplot as plt
# from qiskit.quantum_info import random_unitary
# from QuantumReservoirpy import QReservoir, Layers


# def add_on(circ, timestep, operator, encoder):
#     circ.measure(circ.qubits[0])
#     circ.initialize(encoder[timestep], [0, 1])
#     circ.unitary(operator, circ.qubits)
#     return circ

# res = QReservoir(qubits=4, layers=[
#     Layers.H(),
#     Layers.Timeseries(build_method=add_on, incrementally=True),
#     Layers.Measurement([0, 1, 2, 3])
# ], operator=random_unitary(2**4), encoder = {
#     0: '00',
#     1: '01',
#     2: '10',
#     3: '11'
# })
# exp = res.run([0, 1, 2] * 3)

# res.circuit.draw('mpl')
# print(exp)
# plt.show()

f = lambda res : res

print(f([0, 1, 2, 3]))

x = np.array([[[0, 1], [2, 3]]])
print(x.shape)
print(x)
print(x.flatten())
x = x.reshape((-1, x.shape[-1]))
print(x)