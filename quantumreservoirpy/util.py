import numpy as np

from sklearn.model_selection import train_test_split

from qiskit import QuantumCircuit, QuantumRegister


def listify(elem):
    try:
        return list(elem)
    except:
        return [elem]


def shift_down(arr, k):
    t, o = arr.shape
    result = np.zeros_like(arr)

    for j in range(o):
        result[k:, j] = arr[: t - k, j]

    return result


def create_shifted_array(a, k):
    t, o = a.shape
    result = np.zeros((t, k * o), dtype=a.dtype)

    for i in range(k):
        result[:, i * o : (i + 1) * o] = shift_down(a, i)

    return result


def memory_to_mean(memory):
    """Utility for analyzing qiskit.Result.get_memory() memory.
    Assumes the data is a list indexed by shots.
    """

    states = [" ".join(mem).split() for mem in memory]
    numb = np.array(states, dtype=int)
    return np.average(numb, axis=0)


# from qiskit.quantum_info import SparsePauliOp
# def random_ising_H(num_qubits, num_terms, low=-0.5, high=0.5, h=0):
#    possibles = ["X", "Y", "Z"]
#
#    weights = np.random.uniform(low=low, high=high)
#
#    ops = np.full(shape=(num_terms, num_qubits), fill_value="I")
#
#    ops[:, :2] = np.random.choice(possibles, size=(num_terms, 2))
#
#    pauli_strings = [['IIII']]*num_terms
#    for i in range(num_terms):
#        pauli_strings[i] = "".join(ops[i][np.random.permutation(num_qubits)])
#
#    return SparsePauliOp(
#        data=pauli_strings, coeffs=weights
#    ).to_operator()


def stress_test_models(X_data, y_data, models, test_to_train_ratio=1 / 3, N=100):
    results = np.zeros(len(models))
    for _ in range(N):
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=test_to_train_ratio
        )
        for i, model in enumerate(models):
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            results[i] += score
    return results / N


def get_weights(k, a, b):
    return (b - a) * np.random.random_sample(k) + a


class IsingParams:
    def __init__(
        self,
        topology=None,
        trottersteps=None,
        t=None,
        Jx=None,
        Jz=None,
        hx=None,
        hy=None,
        hz=None,
    ):
        self.topology = topology
        self.trottersteps = trottersteps
        self.t = t
        self.Jx = Jx
        self.Jz = Jz
        self.hx = hx
        self.hy = hy
        self.hz = hz


def get_Ising_circuit(n, isingparams):
    q = QuantumRegister(n)
    circuit = QuantumCircuit(q)
    for _ in range(isingparams.trottersteps):
        for i in range(n):
            circuit.rx(isingparams.t * isingparams.hx[i], i)
            circuit.ry(isingparams.t * isingparams.hy[i], i)
            circuit.rz(isingparams.t * isingparams.hz[i], i)
        for i, e in enumerate(isingparams.topology):
            circuit.rzz(isingparams.t * isingparams.Jx[i], e[0], e[1])
            circuit.rxx(isingparams.t * isingparams.Jz[i], e[0], e[1])
    return circuit

def limitrange(topology, dist=2):
    ltop = []
    for tmp in topology:
        if abs(tmp[0]-tmp[1])<=2:
            ltop.append(tmp)
    return ltop


def randomIsing(n, topology, trottersteps, t):
    Jx = get_weights(len(topology), -1, 1)
    Jz = get_weights(len(topology), -1, 1)
    hx = get_weights(n, -0.5, 0.5)
    hy = get_weights(n, -0.5, 0.5)
    hz = get_weights(n, -0.5, 0.5)
    isingparams = IsingParams(topology, trottersteps, t, Jx, Jz, hx, hy, hz)

    circuit = get_Ising_circuit(n, isingparams)

    return circuit, isingparams
