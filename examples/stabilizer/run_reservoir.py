import sys
import os
# Add the package folder to sys.path relative to this script
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, PACKAGE_PATH)
from reservoirpy.datasets import logistic_map, narma
import numpy as np
import reservoirpy as rpy
import pickle
from qiskit import  transpile
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
from my_util import fit_model, run_prediction, compute_z_expectations, pauli_z_expectation,predict_one_step_ahead
from my_util import narma_task, henon1d,logistic_map
from my_util import compute_expectations_for_all_timesteps, execute_reservoir, generate_ZI_pauli_strings





def main(num_qubits, num_meas, num_reservoirs, method, noise, lents, decode, casename,
        tableaunr,stab_method, stab_degree, timeplex=1, degree=None,shots=10000):
    if not degree:
        degree = num_meas
    #degree = min(degree, 3)
    k=timeplex
    num_neurons=num_reservoirs*num_meas

    if casename == "henon":
        ts = henon1d(lents)
    elif casename == "logistic":
        ts=logistic_map(lents, r=3.9, x0=0.5).flatten()
    #ts=narma(200).flatten()

    string_identifier="casename"+str(casename)+"_num_qubits"+str(num_qubits)+"_num_meas"+str(num_meas)
    string_identifier+="_degree"+str(degree)+"_num_reservoirs"+str(num_reservoirs)+"_timeplex10"
    string_identifier+="_method"+str(method)+"_noise"+str(noise)+"_tableaunr"+str(tableaunr)+"_shots"+str(shots)
    if not decode:
        string_identifier+="_decodeFalse"
    string_identifier+="_tableaunr"+str(tableaunr)

    print(string_identifier, " number of neurons/observables=",num_neurons )



    model= Ridge(alpha=1e-7)#LinearRegression()



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
    warmup=int(WARMUP*len(ts))
    if method == "classical":
        res = Reservoir(num_neurons, lr=0.7, sr=0.99)
    if method != "classical":
        with open("data/isingparams_"+"num_qubits"+str(num_qubits)+"_num_reservoirs20"+".pickle","rb") as f:
            isingparams = pickle.load(f)
        isingparams = dict(islice(isingparams.items(), num_reservoirs))
        with open("data/tableau_"+"num_qubits"+str(num_qubits)+"_num_measurements"+str(num_meas)+"_num_tableaus100.pickle","rb") as f:
            tableau = pickle.load(f)
        sampled_keys = random.sample(list(tableau.keys()), 1)
        sampled_list = [tableau[key] for key in sampled_keys]
        print(isingparams.values())
        # merge and wrap as list of dict
        tableau = [{
            'stabilizer': [s for d in sampled_list for s in d['stabilizer']],
            'destabilizer': [s for d in sampled_list for s in d['destabilizer']]
        }][0]
        print(tableau["stabilizer"])

        if method == "quantum_part":
            res=[]
            for i in range(num_reservoirs):
                res.append(PartialMeasurement(num_qubits, num_meas, backend = AerSimulator(noise_model=noise_model),\
                                        degree=degree, num_reservoirs=1, isingparams=list(isingparams.items())[i],decode=decode))
        if method == "quantum_stab":
            res=[]
            for i in range(num_reservoirs):
                res.append(Stabilizer(num_qubits, num_meas, tableau=tableau, backend = AerSimulator(noise_model=noise_model),\
                                degree=1,stab_method=stab_method,stab_deg=stab_degree, num_reservoirs=1,isingparams= list(isingparams.items())[i],decode=True))
    tscv = TimeSeriesSplit(n_splits=2)
    print(tscv)
    for ep, (train_index, test_index) in enumerate(tscv.split(ts)):

        X_train=ts[train_index]
        y_train_aux=ts[train_index+1]
        X_test=ts[test_index]
        
        num_pred = len(test_index)

        with open("results/X_train"+str(ep)+"_"+string_identifier+".pickle","wb") as f:
            pickle.dump(X_train, f)
        with open("results/X_test"+str(ep)+"_"+string_identifier+".pickle","wb") as f:
            pickle.dump(X_test, f)
        if method =='classical':
            states = res.run(X_train.reshape(-1, 1))
            model, X, y = fit_model(model, states, y_train_aux, WARMUP, 1)
            score = model.score(X, y)
        else:
            num_tot_obs=len(generate_ZI_pauli_strings(num_meas))
            state=[]
            for i,reservoir_circuit in enumerate(res):
                circuit = reservoir_circuit.circuit(X_train.tolist())
                circuit = transpile(circuit, backend= AerSimulator(noise_model=noise_model))
                counts_train = execute_reservoir(X_train,num_meas,shots, circuit)
                obs_train = compute_expectations_for_all_timesteps(counts_train, generate_ZI_pauli_strings(num_meas)[1:num_tot_obs-1])
                state.append(obs_train)  # shape (train-1, num_measurement)
            # Build supervised learning dataset with memory k
            states_train_k = []
            y_train_k = []
            for i in range(0, len(list(y_train_aux))):
                # Build feature vector stacking k past observations
                Xi = []
                Yi = []
                for j in range(k):
                    for r in range(0,num_reservoirs):
                        Xi.extend(state[r][i-j-1])  # notice: i-j-1 because X_train is already delayed by 1
                    Yi.append(y_train_aux[i-j-1])
                states_train_k.append(Xi[::-1])
                y_train_k.append(Yi[::-1])

            # Convert to numpy arrays
            states_train_k=np.array(states_train_k)
            y_train_k = np.array(y_train_k)

            # Train Ridge regression
            states= states_train_k[warmup+k:]
            y= y_train_k[warmup+k:]
            model.fit(states, y)
            score = model.score(states, y)

        print("score[",method,"]=", score)
        
        with open("results/score"+str(ep)+"_"+string_identifier+".pickle","wb") as f:
            pickle.dump(score, f)

        with open("results/state"+str(ep)+"_"+string_identifier+".pickle","wb") as f:
            pickle.dump(states, f)
        # Now autoregressive prediction
        timeseries_aux = np.vstack([np.array(X_train.copy()).reshape(-1, 1), np.array(y_train_aux[-1]).reshape(-1, 1)])
        y_pred = timeseries_aux.copy()
        num_pred = len(X_test)
        firsttime=True
        if method=='classical':
            for j in range(num_pred):
                    print(j,"/",num_pred)
                    print(timeseries_aux[-1])
                    states_aux = res.run(timeseries_aux[-1].reshape(-1, 1))
                    tmp = run_prediction(model, states_aux, 1)
                    
                    timeseries_aux= np.append(timeseries_aux,tmp)
                    firsttime=False
        else:
            for i in range(len(X_test)):
                print(j,"/",num_pred)
                pred_step = predict_one_step_ahead(model, timeseries_aux, num_qubits, num_meas, res, backend= AerSimulator(noise_model=noise_model),method=method,ising_params=isingparams,shots=shots)
                timeseries_aux = np.vstack([timeseries_aux, np.array([[pred_step]])])

        with open("results/prediction"+str(ep)+"_"+string_identifier+".pickle","wb") as f:
            pickle.dump(timeseries_aux, f)
 


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
    stab_method= str(sys.argv[10])
    stab_degree= int(sys.argv[11])
    shots= int(sys.argv[12])

    print("Running:", num_qubits, num_meas, num_reservoirs, method, noise, lentrain, decode, casename, tableaunr,stab_method,stab_degree,shots=shots)
    main(num_qubits, num_meas, num_reservoirs, method, noise, lentrain, decode, casename, tableaunr,stab_method,stab_degree,shots=shots)

