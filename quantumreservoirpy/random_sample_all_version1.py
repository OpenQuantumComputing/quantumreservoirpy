# standard libs
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
from qiskit_aer import AerSimulator
from qiskit import *
from qiskit.quantum_info import *
from qiskit.circuit.random import random_circuit
from random import choice, sample

# quantumreservoirpy
from partialmeasurement import PartialMeasurement
from stabilizer import Stabilizer
from util import *

# sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


def fit_model(model, res_states, series, WARMUP):
    warmup = int(len(series) * WARMUP)

    X = res_states
    y = series

    X = X[warmup:]
    y = y[warmup:]

    model.fit(X, y)

    return model, X, y


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
        if typ == "standard":
            self.resmodel = PartialMeasurement(
                num_qubits,
                num_meas,
                backend=AerSimulator(noise_model=noise_model),
                degree=degree,
                num_reservoirs=num_reservoirs,
            )
        else:
            self.resmodel = Stabilizer(
                num_qubits,
                num_meas,
                backend=AerSimulator(noise_model=noise_model),
                degree=degree,
                num_reservoirs=num_reservoirs,
            )

    def fit(self, X_train, num_shots, pipe, WARMUP):
        states = self.resmodel.run(timeseries=X_train, shots=num_shots)

        scores = []
        predictions = []
        for j in range(1, states.shape[1] + 1):
            pipe, X, y = fit_model(pipe, states[:, :j], X_train, WARMUP)
            score = pipe.score(X, y)
            prediction = pipe.predict(X)
            scores.append(score)
            predictions.append(prediction)

        return scores, predictions, X, y, pipe, states, self.resmodel.get_params()


def saveResult(res):
    global scores, predictions, X, y, pipe, states, resmodel
    scores.append(res[0])
    predictions.append(res[1])
    X.append(res[2])
    y.append(res[3])
    pipe.append(res[4])
    states.append(res[5])
    resmodel.append(res[6])


# def plot(m_list, mean_cnots, var_cnots, min_cnots, max_cnots, color, col, style, text):
#    plt.plot(m_list, mean_cnots, style+"-"+col, label=r"mean "+text)
#    plt.fill_between(
#        m_list,
#        mean_cnots - np.sqrt(var_cnots),
#        mean_cnots + np.sqrt(var_cnots),
#        color=color,
#        alpha=0.35,
#        label=r"$1\sigma$",
#    )
#    plt.plot(m_list, min_cnots, style+":k")
#    plt.plot(m_list, max_cnots, style+":k", label=r"min/max "+text)
#    # plt.plot(m_list, mean_cnots_opt-np.sqrt(var_cnots_opt), '-g', alpha=.35)
#    # plt.plot(m_list, mean_cnots_opt+np.sqrt(var_cnots_opt), '-g', alpha=.35,label="std dev")
#    plt.fill_between(m_list, min_cnots, max_cnots, color="black", alpha=0.15)
#    # plt.plot(m_list, time_list / max(time_list), label = f"n = {n}", color = "black")
#    plt.legend()
#    plt.xlim(min(m_list), max(m_list))
#    plt.xlabel(r"$|B|$")
#    # plt.ylabel(r"$t / t_{max}$")
#    plt.ylabel("#CNOTS")
#    plt.grid()


def main(
    ts,
    num_qubits,
    num_meas,
    num_reservoirs,
    degree,
    typ,
    noise_model,
    num_samples,
    num_shots,
):
    worker = Worker(num_qubits, num_meas, num_reservoirs, degree, typ, noise_model)

    WARMUP = 0.3
    tscv = TimeSeriesSplit()
    # Iterate through the splits and get the indices for the first fold
    for train_index, test_index in tscv.split(ts):
        train_indices_first_fold = train_index
        test_indices_first_fold = test_index
        break  # Stop after the first fold

    global scores, predictions, X, y, pipe, states, resmodel
    scores = []
    predictions = []
    X = []
    y = []
    pipe = []
    states = []
    resmodel = []
    pipe_ = Pipeline(
        steps=[
            # ('scale', StandardScaler()),
            # ('preprocessor', PolynomialFeatures(degree=1, include_bias=False)),
            ("estimator", Ridge(alpha=1e-8))
        ]
    )

    pool = Pool()
    for j in range(num_samples):
        deb = pool.apply_async(
            worker.fit,
            args=(ts[train_index], num_shots, pipe_, WARMUP),
            callback=saveResult,
        )
        #try:
        #    deb.get()
        #except Exception as e:
        #    print("Exception in worker.fit:", e)
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
        + str()
        + ".sav"
    )
    joblib.dump([scores, predictions, X, y, pipe, states, resmodel], filename)
    print(scores)


if __name__ == "__main__":
    print("Creating time series...")
    x2, y2 = get_time_series()
    k = 20
    ts = ((x2[::k] + 2) / 4)[200:]
    print("done.")

    num_qubits = 4
    num_meas = 3
    num_reservoirs = 3
    degree = 3
    noise_model = None
    num_samples = 10
    num_shots = 10**3
    for typ in ["standard", "stabilizer"]:

        main(
            ts,
            num_qubits,
            num_meas,
            num_reservoirs,
            degree,
            typ,
            noise_model,
            num_samples,
            num_shots,
        )
