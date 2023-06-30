import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import random_unitary

import QuantumReservoirpy as QRP

from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


mem_qubits = [0, 1]
timeseries = [0, 1, 2, 3] * 20

ops = [random_unitary(2**len(mem_qubits)) for _ in range(len(np.unique(timeseries)))]
print(len(ops))


encoder = QRP.utilities.generate_encoder(len(mem_qubits))

# Backwards horizon (M) needed to cover 95% (1-r) of all cases
M = 10
r = 0.95

# Dissipation
d =  (1 - r)**(1/M)
print("Dissipation: ", d)

# Number of different circuits (different due to randomness)
num_circuits = 5

SHOTS = 1000
num_subsystems = 20

results = np.zeros((len(timeseries), 2**len(mem_qubits)))
for _ in tqdm(range(num_subsystems)):
    for l in range(1, len(timeseries)):
        qr = QuantumRegister(len(mem_qubits))
        cr = ClassicalRegister(len(mem_qubits))
        circ = QuantumCircuit(qr, cr)
        circ.reset(qr)

        start_idx = max(0, l - M)
        for time in timeseries[start_idx:l]:
            circ.unitary(ops[time], qr, label=f'{time}')
            if np.random.random() > d:
                circ.reset(qr)
        circ.measure(qr, cr)
        counts = QRP.utilities._simulate(circ, shots=SHOTS)#.get_counts()
        print(counts)

        # for key, val in counts.items():
        #     bin_to_int = int(key, 2)
        #     results[l][bin_to_int] += val

results /= (SHOTS * num_subsystems)


warmup = M

states = results[:-1]
target = results[1:]

states = states[warmup:]
target = target[warmup:]

model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
train_x, test_x, train_y, test_y = train_test_split(states, target, test_size=0.33)

model.fit(train_x, train_y)
print(model.score(test_x, test_y))

plt.show()

