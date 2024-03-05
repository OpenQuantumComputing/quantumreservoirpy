# standard libs
import traceback
import sys
import time
import itertools
import numpy as np
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
import joblib
import tikzplotlib

# qiskit stuff
import qiskit
from qiskit_aer import AerSimulator
from qiskit import *
from qiskit.quantum_info import *
from qiskit.circuit.random import random_circuit
from random import choice, sample

# quantumreservoirpy
from quantumreservoirpy.partialmeasurement import PartialMeasurement
from quantumreservoirpy.stabilizer import Stabilizer
from quantumreservoirpy.util import *

# sklearn
from sklearn.model_selection import TimeSeriesSplit


def get_time_series(t_stop=200):
    G = 9.8  # acceleration due to gravity, in m/s^2
    L1 = 1.0  # length of pendulum 1 in m
    L2 = 1.0  # length of pendulum 2 in m
    L = L1 + L2  # maximal length of the combined pendulum
    M1 = 1.0  # mass of pendulum 1 in kg
    M2 = 1.0  # mass of pendulum 2 in kg
    t_stop = 200  # how many seconds to simulate
    history_len = 500  # how many trajectory points to display

    def derivs(t, state):
        dydx = np.zeros_like(state)

        dydx[0] = state[1]

        delta = state[2] - state[0]
        den1 = (M1 + M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
        dydx[1] = (
            M2 * L1 * state[1] * state[1] * np.sin(delta) * np.cos(delta)
            + M2 * G * np.sin(state[2]) * np.cos(delta)
            + M2 * L2 * state[3] * state[3] * np.sin(delta)
            - (M1 + M2) * G * np.sin(state[0])
        ) / den1

        dydx[2] = state[3]

        den2 = (L2 / L1) * den1
        dydx[3] = (
            -M2 * L2 * state[3] * state[3] * np.sin(delta) * np.cos(delta)
            + (M1 + M2) * G * np.sin(state[0]) * np.cos(delta)
            - (M1 + M2) * L1 * state[1] * state[1] * np.sin(delta)
            - (M1 + M2) * G * np.sin(state[2])
        ) / den2

        return dydx

    # create a time array from 0..t_stop sampled at 0.02 second steps
    dt = 0.01
    t = np.arange(0, t_stop, dt)

    # th1 and th2 are the initial angles (degrees)
    # w10 and w20 are the initial angular velocities (degrees per second)
    th1 = 120.0
    w1 = 0.0
    th2 = -10.0
    w2 = 0.0

    # initial state
    state = np.radians([th1, w1, th2, w2])

    y = np.empty((len(t), 4))
    y[0] = state

    y = scipy.integrate.solve_ivp(derivs, t[[0, -1]], state, t_eval=t).y.T

    x1 = L1 * np.sin(y[:, 0])
    y1 = -L1 * np.cos(y[:, 0])

    x2 = L2 * np.sin(y[:, 2]) + x1
    y2 = -L2 * np.cos(y[:, 2]) + y1
    return x2, y2


class Worker:
    """n = dim(hilbert space)"""

    def __init__(
        self, num_qubits, num_meas, num_reservoirs, degree, typ, noise_model=None
    ):
        self.num_qubits = num_qubits
        self.num_meas = num_meas
        self.num_reservoirs = num_reservoirs
        self.degree = degree
        self.typ = typ
        self.noise_model = noise_model

    def run(self, X_train, num_shots, precision):
        if typ == "standard":
            self.resmodel = PartialMeasurement(
                self.num_qubits,
                self.num_meas,
                backend=AerSimulator(noise_model=self.noise_model),
                degree=self.degree,
                num_reservoirs=self.num_reservoirs,
            )
        else:
            self.resmodel = Stabilizer(
                self.num_qubits,
                self.num_meas,
                backend=AerSimulator(noise_model=self.noise_model),
                degree=self.degree,
                num_reservoirs=self.num_reservoirs,
            )
        states = self.resmodel.run(
            timeseries=X_train, shots=num_shots, precision=precision
        )
        return (
            states,
            self.resmodel.variances,
            self.resmodel.shots_taken,
            self.resmodel.get_params(),
        )


def saveResult(res):
    global states, variances, shots_taken, resmodelparams
    states.append(res[0])
    variances.append(res[1])
    shots_taken.append(res[2])
    resmodelparams.append(res[3])


def main(
    timeseries,
    num_qubits,
    num_meas,
    num_reservoirs,
    degree,
    typ,
    noise_model,
    num_samples,
    num_shots,
    precision,
):
    worker = Worker(num_qubits, num_meas, num_reservoirs, degree, typ, noise_model)

    global states, variances, shots_taken, resmodelparams
    states = []
    variances = []
    shots_taken = []
    resmodelparams = []

    pool = Pool()
    for j in range(num_samples):
        deb = pool.apply_async(
            worker.run,
            args=(timeseries, num_shots, precision),
            callback=saveResult,
        )
        # try:
        #   deb.get()
        # except Exception as e:
        #   print("Exception in worker.run:", e)
        #   traceback.print_exc()
    pool.close()
    pool.join()

    filename = (
        "simulation_"
        + str(num_qubits)
        + "_"
        + str(num_meas)
        + "_"
        + str(num_reservoirs)
        + "_"
        + str(degree)
        + "_"
        + str(typ)
        + "_"
        + str(noise_model)
        + "_"
        + str(num_samples)
        + "_"
        + str(num_shots)
        + "_"
        + str(precision)
        + ".sav"
    )
    joblib.dump([timeseries, states, variances, shots_taken, resmodelparams], filename)


if __name__ == "__main__":
    print("Creating time series...")
    x2, y2 = get_time_series()
    k = 20
    ts = ((x2[::k] + 2) / 4)[200:]

    tscv = TimeSeriesSplit()
    # Iterate through the splits and get the indices for the first fold
    # cou = 0
    for train_index, test_index in tscv.split(ts):
        # if cou < 2:
        #    continue
        # cou += 1
        train_indices_first_fold = train_index
        test_indices_first_fold = test_index
        break  # Stop after the first fold

    timeseries = ts[train_index]
    print("done.")

    # num_samples = 20

    # for num_qubits in [6,5,4,3]:
    #    for num_meas in range(2, num_qubits):
    ##num_qubits = 4
    ##num_meas = 3
    #        num_reservoirs = 10-num_meas

    num_qubits = int(sys.argv[1])
    num_meas = int(sys.argv[2])
    num_reservoirs = int(sys.argv[3])
    num_samples = int(sys.argv[4])

    print("Running:", num_qubits, num_meas, num_reservoirs, num_samples)

    noise_model = None
    num_shots = 10**3
    precision = 1e-2

    degree = num_meas

    for typ in ["standard", "stabilizer"]:
        main(
            timeseries,
            num_qubits,
            num_meas,
            num_reservoirs,
            degree,
            typ,
            noise_model,
            num_samples,
            num_shots,
            precision,
        )
