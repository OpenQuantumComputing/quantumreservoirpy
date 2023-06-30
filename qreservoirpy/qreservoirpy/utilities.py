from qiskit import Aer
import qiskit as qs
import numpy as np
import matplotlib.pyplot as plt


def listify(elem):
    try:
        return list(elem)
    except:
        return [elem]


def _simulate(circuit, shots, transpile):
    simulator = Aer.get_backend('aer_simulator_statevector')
    if transpile:
        circuit = qs.transpile(circuit, simulator)
    return simulator.run(circuit, shots=shots, memory=True).result()

def simulate(circuit, shots, transpile):
    return _simulate(circuit, shots, transpile).get_memory()

def memory_to_mean(memory, meas_per_timestep):
        # Takes in memory of the form [1010101001, 1010010101, 0010110010 ...]
        # and computes the averages over shots for each specific measurement (measurement performed at each timestep).
        # A chunk of memory, i.e. 101010 may be measurements of single qubits (such that 101010 correspond to
        # measurements of a single qubit 6 times), but may also correspond to 2 measurments of 3 qubits or 3 of 2 etc.
        # This is decided by meas_per_timestep
        shots = len(memory)

        # Res =  shots x    timesteps        x           features
        res = np.zeros((shots, len(memory[0]))).reshape((shots, len(memory[0])//meas_per_timestep,-1))

        for shot, data in enumerate(memory):
            # Split from e.g. 10010101 to [[1, 0], [0, 1] , [0, 1], [0, 1]] depending on meas size (equal to 2 in this example)
            chunks = np.array([list(data[i:i+meas_per_timestep]) for i in range(0, len(data), meas_per_timestep)], dtype=np.int32)
            for timestep, chunk in enumerate(chunks):
                res[shot, timestep] = chunk
        # res = res.reverse()

        # Average over shots. Returned in format (timesteps, features)
        return np.flip(np.average(res, axis=0))

def result_plotter(x, target, warmup=0):

    warmup_len = int(len(target) * warmup)
    x = x[warmup_len:]
    target = target[warmup_len:]
    n_features = len(x[0])
    n_cols = int(np.ceil(np.sqrt(n_features)))
    n_rows = int(np.ceil(n_features/n_cols))

    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(10 * n_cols, 10*n_rows), sharex='col', sharey='row')
    cmap = plt.get_cmap('jet', len(np.unique(target)))

    try:
        axes = axes.flatten()
    except:
        axes = [axes]

    for i, t in enumerate(np.unique(target)):
        mask = t==target
        for idx in range(n_features):
            ax = axes[idx]
            ax.plot(warmup_len + np.arange(len(x))[mask], x[mask][:, idx], marker='o', lw=0, label=f'{t}', c=cmap(i))
            ax.set_title(f'Feature {idx+1}')
            ax.legend(loc='upper right')
    return fig, axes