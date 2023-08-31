import numpy as np
from .randomcircuit import random_circuit

def listify(elem):
    try:
        return list(elem)
    except:
        return [elem]


def memory_to_mean(memory):
    """ Utility for analyzing qiskit.Result.get_memory() memory.
    Assumes the data is a list indexed by shots.
    """

    states = [" ".join(mem).split() for mem in memory]
    numb = np.array(states, dtype=int)
    return np.average(numb, axis=0)


from qiskit.quantum_info import SparsePauliOp
def random_ising_H(num_qubits, num_terms, low=-0.5, high=0.5, h=0):
    possibles = ["X", "Y", "Z"]

    weights = np.random.uniform(low=low, high=high)

    ops = np.full(shape=(num_terms, num_qubits), fill_value="I")

    ops[:, :2] = np.random.choice(possibles, size=(num_terms, 2))

    pauli_strings = [['IIII']]*num_terms
    for i in range(num_terms):
        pauli_strings[i] = "".join(ops[i][np.random.permutation(num_qubits)])

    return SparsePauliOp(
        data=pauli_strings, coeffs=weights
    ).to_operator()


from sklearn.model_selection import train_test_split
def stress_test_models(X_data, y_data, models, test_to_train_ratio=1/3, N=100):
    results = np.zeros(len(models))
    for _ in range(N):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_to_train_ratio)
        for i, model in enumerate(models):
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            results[i] += score
    return results/N