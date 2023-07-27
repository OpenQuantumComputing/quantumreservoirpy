import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import random_unitary
from qiskit.extensions import HamiltonianGate
from qreservoirpy import QReservoir, Layers, utilities

n_qubits = 6
SHOTS = 1000
H = utilities.random_ising_H(num_qubits=n_qubits, num_terms=10)
T = 1 #s
UT = HamiltonianGate(H, T, label="UT")
ANSWER = [0, 1, 1, 1, 1, 1, 1, 2] + [0, 1, 1, 1, 1, 1, 1,  3]
N_T = 20
timeseries = ANSWER * N_T
M=16



def build_method(circ, timestep, resop, encoder):
    # circ.measure([0, 1])
    circ.initialize(encoder[timestep], [0, 1])
    circ.unitary(resop, circ.qubits, label=resop.label)
    return circ

res = QReservoir(qubits=n_qubits, layers=[
    Layers.Reset(),
    Layers.Timeseries(build_method=build_method),
    Layers.Measurement(range(3))
], resop = UT,
 encoder = {
    0: '00',
    1: '01',
    2: '10',
    3: '11'
}, M=M)

res.circuit.draw('mpl')

states = res.run(timeseries, shots=SHOTS, incrementally=True)
import matplotlib.pyplot as plt
import numpy as np
WARMUP = 0.1

warmup = int(len(states) * WARMUP)
xx = states[:, 0][warmup:]
yy = states[:, 1][warmup:]
zz = states[:, 2][warmup:]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

cmap = plt.get_cmap('jet', len(np.unique(timeseries)))
for i, x, y, z in zip(timeseries, xx, yy, zz):
    ax.scatter(x, y, z, c=cmap(i))
ax.set_xlabel('state 1')
ax.set_ylabel('state 2')
ax.set_zlabel('state 3')
plt.show()