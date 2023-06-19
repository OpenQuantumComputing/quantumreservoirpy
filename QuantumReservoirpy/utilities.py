import numpy as np
import matplotlib.pyplot as plt
from qiskit import transpile, Aer


def generate_encoder(bitwidth):
    # Creates an encoder in the form of a dictionary,
    # to encode the
    encoders = {
            1: {
                0: '0',
                1: '1'
            },
            2: {
                0: '00',
                1: '01',
                2: '10',
                3: '11'
            }
        }
    return encoders[bitwidth]

def f(x, y, model):
    return model.predict([x, y])

def feature_plot(x, target, model):
    fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    cmap = plt.get_cmap('jet', len(np.unique(target)))


    N = 100
    xvals = np.linspace(0.3, 0.7, N)
    yvals = np.linspace(0.3, 0.7, N)

    # print(xvals)
    # print(yvals)
    # print(x)

    grid = np.zeros((xvals.size-1, yvals.size-1))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            grid[i, j] = model.predict([[xvals[i], yvals[j]]])[0]
    # print(grid.shape)
    print(xvals.shape)
    print(yvals.shape)
    print(grid.shape)
    ax.pcolormesh(xvals, yvals, grid, cmap=cmap)

    print(x)
    print(model.predict(x))
    print(len(x))

    # print(x)
    # print(model.predict(x))
    for t in np.unique(target):
        mask = t==target
        print(t, mask, np.sum(mask))
        ax.scatter(x[mask][0], x[mask][1], marker='o', lw=1, label=f'{t}',edgecolors='k', color=cmap(t))
    return fig, ax

def result_plotter(x, target):
    n_features = len(x[0])
    n_cols = int(np.ceil(np.sqrt(n_features)))
    n_rows = int(np.ceil(n_features/n_cols))

    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(10 * n_cols, 10*n_rows), sharex='col', sharey='row')
    cmap = plt.get_cmap('jet', len(np.unique(target)))
    axes = axes.flatten()

    for i, t in enumerate(np.unique(target)):
        mask = t==target
        for idx in range(n_features):
            ax = axes[idx]
            ax.plot(np.arange(len(x))[mask], x[mask][:, idx], marker='o', lw=0, label=f'{t}', c=cmap(i))
            ax.set_title(f'Feature {idx+1}')
            ax.legend(loc='upper right')
    return fig, ax

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
            # # Comes out in reverse from qiskit if measured more than one qubit
            # if meas_per_timestep > 1:
            #     data = data[::-1]

            # Split from e.g. 10010101 to [[1, 0], [0, 1] , [0, 1], [0, 1]] depending on meas size (equal to 2 in this example)
            chunks = np.array([list(data[i:i+meas_per_timestep]) for i in range(0, len(data), meas_per_timestep)], dtype=np.int32)
            for timestep, chunk in enumerate(chunks):
                res[shot, timestep] = chunk

        # Average over shots. Returned in format (timesteps, features)
        return np.average(res, axis=0)

def simulate(circuit, shots):
    simulator = Aer.get_backend('aer_simulator_statevector')
    circuit = transpile(circuit, simulator)
    return simulator.run(circuit, shots=shots, memory=True).result().get_memory()



