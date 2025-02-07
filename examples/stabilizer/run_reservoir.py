import sys
from reservoirpy.datasets import logistic_map, narma
import numpy as np
import reservoirpy as rpy
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from quantumreservoirpy.util import create_shifted_array
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from itertools import combinations
from quantumreservoirpy.util import randomIsing
from reservoirpy.nodes import Reservoir
from quantumreservoirpy.partialmeasurement import PartialMeasurement
from quantumreservoirpy.stabilizer import Stabilizer
from qiskit_aer import AerSimulator

def fit_model(model, res_states, series, WARMUP, timeplex=1):
    warmup = int(len(series) * WARMUP)

    X = res_states[warmup:-1]
    y = series[warmup + 1 :]

    if timeplex > 1:
        X = create_shifted_array(X, timeplex)
    model.fit(X, y)

    return model, X, y


def run_prediction(model, res_states, timeplex=1):

    X = np.copy(res_states)

    if timeplex > 1:
        X = create_shifted_array(X, timeplex)
    X = X[-1,:]
    X = X.reshape((1, -1))
    return model.predict(X)

def henon1d(n, a=1.4, b=0.3):
    ts=[0,0]
    for i in range(2,n+2):
        ts.append(1 - a*ts[i-1]**2 + b*ts[i-2])
    return np.array(ts[2:])


def main(num_qubits, num_meas, num_reservoirs, method, noise, lentrain, decode, casename, tableaunr, timeplex=10, degree=None):

    if not degree:
        degree = num_meas
    #degree = min(degree, 3)

    num_neurons=num_reservoirs*(2**num_meas-1)

    if casename == "henon":
        ts = henon1d(200)
    elif casename == "logistic":
        ts=logistic_map(200, r=3.9, x0=0.5).flatten()
    #ts=narma(200).flatten()

    string_identifier="casename"+str(casename)+"_num_qubits"+str(num_qubits)+"_num_meas"+str(num_meas)
    string_identifier+="_degree"+str(degree)+"_num_reservoirs"+str(num_reservoirs)+"_timeplex"+str(timeplex)
    string_identifier+="_method"+str(method)+"_noise"+str(noise)
    if not decode:
        string_identifier+="_decodeFalse"
    string_identifier+="_tableaunr"+str(tableaunr)

    print(string_identifier, " number of neurons/observables=",num_neurons )



    linreg = Ridge(alpha=1e-7)#LinearRegression()



    if noise == "None":
        noise_model = None
    else:
# Create an empty noise model
        noise_model = NoiseModel()
# Add depolarizing error to all single qubit u1, u2, u3 gates
        error = depolarizing_error(0.01, 1)
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
        error = depolarizing_error(0.1, 2)
        noise_model.add_all_qubit_quantum_error(error, ['cx'])

    WARMUP=0.3

    with open("isingparams_"+"num_qubits"+str(num_qubits)+"_num_reservoirs"+str(num_reservoirs)+".pickle","rb") as f:
        isingparams = pickle.load(f)

    with open("tableau_"+"num_qubits"+str(num_qubits)+"_num_measurements"+str(num_measurements)+"_num_tableaus"+str(num_tableaus)+".pickle","rb") as f:
        tableau = pickle.load(f)

    print(tableau[tableaunr])

    if method == "classical":
        res = Reservoir(num_neurons, lr=0.5, sr=0.9)
    elif method == "quantum_part":
        res = PartialMeasurement(num_qubits, num_meas, backend = AerSimulator(noise_model=noise_model),\
                                 degree=degree, num_reservoirs=num_reservoirs, isingparams=isingparams,decode=decode)
    elif method == "quantum_stab":
        res = Stabilizer(num_qubits, num_meas, backend = AerSimulator(noise_model=noise_model, method="automatic", device="GPU"),\
                         degree=degree, num_reservoirs=num_reservoirs, isingparams=isingparams,decode=decode, tableau = tableau[tableaunr])

    tscv = TimeSeriesSplit()

    for i, (train_index, test_index) in enumerate(tscv.split(ts)):
        if not i == lentrain:
            continue

        X_train=ts[train_index]
        X_test=ts[test_index]
        num_pred = len(test_index)

        with open("X_train"+str(i)+"_"+string_identifier+".pickle","wb") as f:
            pickle.dump(X_train, f)
        with open("X_test"+str(i)+"_"+string_identifier+".pickle","wb") as f:
            pickle.dump(X_test, f)
        
        if method =='classical':
            states = res.run(X_train.reshape(-1, 1), reset=True)
            linreg, X, y = fit_model(linreg, states, X_train, WARMUP, 1)
        else:
            states = res.run(timeseries=X_train, shots=1e3, precision=1e-2)
            linreg, X, y = fit_model(linreg, states, X_train, WARMUP, timeplex)

        score = linreg.score(X, y)
        print("score[",method,"]=", score)
        
        with open("score"+str(i)+"_"+string_identifier+".pickle","wb") as f:
            pickle.dump(score, f)

#        if not method == "classical":
        with open("state"+str(i)+"_"+string_identifier+".pickle","wb") as f:
            pickle.dump(states, f)

        firsttime=True
        prediction = X_train

        for j in range(num_pred):
            print(j,"/",num_pred)
            if method=='classical':
                if firsttime:
                    states = res.state()
                else:
                    states = res.run(prediction[-1])
                tmp = run_prediction(linreg, states, 1)
            else:
                states = res.run(prediction[-2*int(timeplex):], shots=1e3, precision=1e-2)
                tmp = run_prediction(linreg, states, timeplex)
            
            prediction = np.append(prediction,tmp)
            
            firsttime=False

        with open("prediction"+str(i)+"_"+string_identifier+".pickle","wb") as f:
            pickle.dump(prediction, f)



if __name__ == "__main__":
    num_qubits = int(sys.argv[1])
    num_meas = int(sys.argv[2])
    num_reservoirs = int(sys.argv[3])
    method = str(sys.argv[4])
    noise = str(sys.argv[5])
    lentrain = int(sys.argv[6])
    decode = bool(int(sys.argv[7]))
    casename = str(sys.argv[8])
    tableaunr = str(sys.argv[9])

    print("Running:", num_qubits, num_meas, num_reservoirs, method, noise, lentrain, decode, casename, tableaunr)
    main(num_qubits, num_meas, num_reservoirs, method, noise, lentrain, decode, casename, tableaunr)

