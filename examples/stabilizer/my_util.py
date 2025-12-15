import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict
from sklearn.linear_model import Ridge
from reservoirpy.datasets import narma,logistic_map




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



def compute_z_expectations(memory, num_measurement, num_steps):
    timestep_counts = [defaultdict(int) for _ in range(num_steps)]

    for shot in memory:
        steps = shot.strip().split()  # e.g., ['01', '11', '00', ...] — one per timestep
        for step_idx, bits in enumerate(steps):
            # Reverse bitstring if using little-endian (rightmost qubit is qubit 0)
            bitstring = bits[::-1][:num_measurement]
            timestep_counts[step_idx][bitstring] += 1

    # Convert defaultdicts to regular dicts (optional)
    return [dict(d) for d in timestep_counts]

def pauli_z_expectation(counts, pauli_str):
    total_counts = sum(counts.values())
    expectation = 0.0

    for bitstring, count in counts.items():
        # Compute product of eigenvalues for this bitstring
        val = 1
        # bitstring assumed little-endian (bit 0 is rightmost), adjust if needed
        for i, p in enumerate(pauli_str):
            if p == 'Z':
                # bit at position i in little endian (rightmost = qubit 0)
                bit = bitstring[::-1][i]
                val *= 1 if bit == '0' else -1
            elif p == 'I':
                val *= 1
            else:
                raise ValueError("Pauli string must be 'I' or 'Z' only")
        expectation += val * count

    return expectation / total_counts

def compute_expectations_for_all_timesteps(timestep_counts, pauli_strings):
    all_expectations = []

    for counts in timestep_counts:
        expectations = []
        for p_str in pauli_strings:
            exp_val = pauli_z_expectation(counts, p_str)
            expectations.append(exp_val)
        all_expectations.append(expectations)

    return all_expectations[::-1]

def generate_ZI_pauli_strings(n):
    pauli_strings = [''.join(p) for p in product('ZI', repeat=n)]
    return pauli_strings




def encode_circuit(t,num_qubits):
    qr=QuantumRegister(num_qubits)
    qc=QuantumCircuit(qr)

    for k in range(num_qubits):
        beta = 2**(-(k)/num_qubits)
        #beta=1
        #qc.rx(t *  np.pi * beta, num_qubits-k-1)
        qc.rx(t *  np.pi * beta, k)
    return qc

def quantum_part_circuit(timeseries, num_qubits, num_measurement, reservoir_circuit,type='quantum_part'):
    if type=='quantum_part':
        qr = QuantumRegister(num_qubits, "q")
        crs = [ClassicalRegister(num_measurement, f"c{i}") for i in range(len(timeseries))]
        qc = QuantumCircuit(qr, *crs)

        for step_idx, t in enumerate(timeseries):
            #reservoir = get_Ising_circuit(num_qubits, isingparams)
            qc.compose(encode_circuit(t, num_qubits), qubits=qr, inplace=True)
            qc.compose(reservoir_circuit, qubits=qr, inplace=True)

            # Mid measurement: store each step's result in a different ClassicalRegister
            for i in range(num_measurement):
                qbit = num_qubits - num_measurement + i
                qc.measure(qr[qbit], crs[step_idx][i])
                qc.reset(qr[qbit])  # Reset qubit to |0> after measurement

        return qc
    else:
        #res = Stabilizer(num_qubits, num_measurement, backend = AerSimulator(noise_model=None),\
                           #degree=num_measurement, num_reservoirs=1, isingparams=ising_params,decode=True)
        qc=reservoir_circuit.circuit(timeseries)
        return qc

def execute_reservoir(timeseries,num_measurement, num_shots, circuit):
    backend = AerSimulator()
    

    result = backend.run(circuit, shots=num_shots, memory=True).result()
    
    # Access full memory results (per shot)
    memory = result.get_memory()
    #print(memory)
    z_expectations = compute_z_expectations(
                        memory=memory,
                        num_measurement=num_measurement,
                        num_steps=len(timeseries)
                    )
    return z_expectations


def sine(n):
    ts=[]
    for i in range(1,n):
        ts.append((np.sin(7.35*i*np.pi/n)+1)/2)
        #ts.append(0.5)
    return np.array(ts)

def henon1d(n, a=1.4, b=0.3):
    ts=[0,0]
    for i in range(2,n+2):
        ts.append(1 - a*ts[i-1]**2 + b*ts[i-2])
    return np.array(ts[2:])

def narma_task(n,order):
    rng = np.random.default_rng(seed=2341)
    u = rng.uniform(0, 0.5, size=(n + order, 1))
    y = narma(n, order=order, u=u)
    print(y)
    return np.array(y.flatten()[order:])

def logistic(n):
    y = logistic_map(n_timesteps=n)
    print(y)
    return np.array(y.flatten())



# Prediction function
def predict_one_step_ahead(model, timeseries_aux, num_qubits, num_measurement, reservoirs,backend, method, ising_params=None, shots=10000):
    ts_list = timeseries_aux.flatten().tolist()
    k=5
    obs=[]
    num_tot_obs=len(generate_ZI_pauli_strings(num_measurement))
    for i,reservoir_circuit in enumerate(reservoirs):
        circuit = reservoir_circuit.circuit(ts_list)
        circuit = transpile(circuit, backend)
        counts = execute_reservoir(ts_list,num_measurement,shots, circuit)
        obs.append(compute_expectations_for_all_timesteps(counts, generate_ZI_pauli_strings(num_measurement)[1:num_tot_obs-1]))


    Xi = []
    for j in range(k):
        for r in range (0, len(reservoirs)):
            Xi.extend(obs[r][len(obs[0])-j-1])  # take last k observations
    
    Xi = np.array(Xi[::-1]).reshape(1,-1)
    y_pred = model.predict(Xi)
    if k==1:
        return y_pred[-1]
    else:
        return y_pred[-1][-1]