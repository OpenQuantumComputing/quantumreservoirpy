import sys

PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, PACKAGE_PATH)
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
import random 
from itertools import islice
from collections import defaultdict
from reservoirpy.datasets import logistic_map, narma

def fit_model(model, res_states, series, WARMUP, timeplex=1):
    warmup = int(len(series) * WARMUP)

    X = res_states[warmup:-1]
    y = series[warmup + 1 :]
    model.fit(X, y)

    return model, X, y


def run_prediction(model, res_states, timeplex=1):

    X = np.copy(res_states)
    X = X[-1,:]
    X = X.reshape((1, -1))
    return model.predict(X)

def henon1d(n, a=1.4, b=0.3):
    ts=[0,0]
    for i in range(2,n+2):
        ts.append(1 - a*ts[i-1]**2 + b*ts[i-2])
    return np.array(ts[2:])


def main(num_qubits, num_meas, num_reservoirs, method, noise, lents, decode, casename, tableaunr, timeplex=10, degree=None, stab_method='random',stab_degree=1):

    if not degree:
        degree = num_meas
    #degree = min(degree, 3)

    num_neurons=num_reservoirs*(2**num_meas-1)

    if casename == "henon":
        ts = henon1d(lents)
    elif casename == "logistic":
        ts=logistic_map(lents, r=3.9, x0=0.5).flatten()
    #ts=narma(200).flatten()
    string_identifier="casename"+str(casename)+"_num_qubits"+str(num_qubits)+"_num_meas"+str(num_meas)
    string_identifier+="_degree"+str(degree)+"_num_reservoirs"+str(num_reservoirs)+"_timeplex"+str(timeplex)
    string_identifier+="_method"+str(method)+"_noise"+str(noise)+"_expconcentration"
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

    with open("data/isingparams_"+"num_qubits"+str(num_qubits)+"_num_reservoirs20"+".pickle","rb") as f:
        isingparams = pickle.load(f)
    isingparams = dict(islice(isingparams.items(), num_reservoirs))
    with open("data/tableau_"+"num_qubits"+str(num_qubits)+"_num_measurements"+str(num_meas)+"degree"+str(degree)+"_num_tableaus100.pickle","rb") as f:
        tableau = pickle.load(f)
    sampled_keys = random.sample(list(tableau.keys()), 1)
    sampled_list = [tableau[key] for key in sampled_keys]

    # merge and wrap as list of dict
    tableau = [{
        'stabilizer': [s for d in sampled_list for s in d['stabilizer']],
        'destabilizer': [s for d in sampled_list for s in d['destabilizer']]
    }][0]
    print(tableau["stabilizer"])
    if method == "classical":
        res = Reservoir(num_neurons, lr=0.5, sr=0.9)
    elif method == "quantum_part":
        res = PartialMeasurement(num_qubits, num_meas, backend = AerSimulator(noise_model=noise_model),\
                                    degree=degree, num_reservoirs=num_reservoirs, isingparams=isingparams,decode=decode)
    elif method == "quantum_stab":
                res = res= Stabilizer(num_qubits, num_meas, tableau=tableau, backend = AerSimulator(noise_model=noise_model),\
                            degree=1,stab_method=stab_method,stab_deg=stab_degree, num_reservoirs=num_reservoirs,decode=True)
    tscv = TimeSeriesSplit(n_splits=2)
    print(ts)
    for i, (train_index, test_index) in enumerate(tscv.split(ts)):
        print(i)
        X_train=ts[train_index]
        X_test=ts[test_index]
        num_pred = len(test_index)

        with open("results/X_train"+str(i)+"_"+string_identifier+".pickle","wb") as f:
            pickle.dump(X_train, f)
        with open("results/X_test"+str(i)+"_"+string_identifier+".pickle","wb") as f:
            pickle.dump(X_test, f)
        
        if method =='classical':
            states = res.run(X_train.reshape(-1, 1), reset=True)
            linreg, X, y = fit_model(linreg, states, X_train, WARMUP, 1)
        else:
            states = res.run(timeseries=X_train, shots=1e3, precision=1e-2, transpile=True)
            linreg, X, y = fit_model(linreg, states, X_train, WARMUP, timeplex)

        score = linreg.score(X, y)
        print("score[",method,"]=", score)
        
        with open("results/score"+str(i)+"_"+string_identifier+".pickle","wb") as f:
            pickle.dump(score, f)

    #        if not method == "classical":
        with open("results/state"+str(i)+"_"+string_identifier+".pickle","wb") as f:
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
                states = res.run(prediction, shots=1e3, precision=1e-2,transpile=True)
                tmp = run_prediction(linreg, states, timeplex)
            
            prediction = np.append(prediction,tmp)
            print(prediction)
            firsttime=False

        with open("results/prediction"+str(i)+"_"+string_identifier+".pickle","wb") as f:
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
    tableaunr = int(sys.argv[9])
    timeplex= int(sys.argv[10])
    degree= int(sys.argv[11])
    stab_method= str(sys.argv[12])
    stab_degree= int(sys.argv[13])

    print("Running:", num_qubits, num_meas, num_reservoirs, method, noise, lentrain, decode, casename, tableaunr,timeplex, degree, stab_method, stab_degree)
    main(num_qubits, num_meas, num_reservoirs, method, noise, lentrain, decode, casename, tableaunr,timeplex,degree,stab_method, stab_degree)

