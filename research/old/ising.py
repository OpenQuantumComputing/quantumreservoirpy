import numpy as np
import matplotlib.pyplot as plt
from OLD import QReservoir, Layers, utilities

from qiskit.extensions import HamiltonianGate

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from ulid import ULID

models = np.array([
    SVC(kernel=ker) for ker in ['linear', 'poly', 'rbf', 'sigmoid']
] + [
    RidgeClassifier(alpha=alph) for alph in np.linspace(0, 1e-3, 20)
] + [
    KNeighborsClassifier(n_neighbors=n) for n in range(1, 10)
], dtype=object)



n_qubits = 4
SHOTS = 10
timeseries = [0, 1, 2, 1, 2] * 10
encoder = {
    0: '00',
    1: '01',
    2: '10',
    3: '11'
}

placement = {
    0: [0, 1],
    1: [1, 2],
    2: [2, 3],
}

H = utilities.random_ising_H(num_qubits=n_qubits, num_terms=160)


tvals = np.linspace(0.1, 5, 10)

experiment_results = np.zeros_like(tvals)
experiment_methods = []


for exp_id in tqdm(range(len(tvals)), desc="EXPERIMENT"):
# T = 0.2#s
    T = tvals[exp_id]

    UT = HamiltonianGate(H, T, label="UT")


    def build_method(circuit, timestep, encoding, reservoir, pos):
        circuit.measure(range(4))
        # circuit.reset(range(4))
        circuit.initialize(encoding[timestep], pos[timestep])
        # circuit.initialize(encoding[timestep], [0, 1])
        # circuit.initialize(encoding[timestep], [2, 3])
        circuit.unitary(reservoir, circuit.qubits)

    qres = QReservoir(qubits=n_qubits, layers=[
        Layers.Reset(),
        Layers.Timeseries(build_method=build_method,
                        encoding = encoder,
                        reservoir = UT,
                        pos = placement)
    ])

    states = qres.run(
        timeseries=timeseries,
        shots=SHOTS,
        disable_status_bar=True
    )


    WARMUP = 0.1 # 10%
    warmup_idx = int(len(states) * WARMUP)

    xstates = states[:-1][warmup_idx:]
    target = timeseries[1:][warmup_idx:]

    results = np.zeros(len(models))

    N = 100
    for _ in range(N):
        X_train, X_test, y_train, y_test = train_test_split(xstates, target, test_size=1/3)
        for i, model in enumerate(models):
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            results[i] += score

    results /= N

    sorting = np.flip(np.argsort(results))

    experiment_methods.append([models[sorting][:5]])
    experiment_results[exp_id] = results[sorting][0]

