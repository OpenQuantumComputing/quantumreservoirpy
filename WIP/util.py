from QuantumReservoirpy import QReservoir
from QuantumReservoirpy import Layers
from QuantumReservoirpy import utilities

import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, Lasso, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


M = 100
timeseries = [i for i in range(8)] * 1
reservoir = QReservoir(qubits=5, layers=[
    Layers.OperatorTimeseries(epsilon=1 , M=M)
], num_subcircuits=1)
reservoir.run(timeseries, shots=1)

reservoir.circuit.draw('mpl')
# print(reservoir.states)s
plt.show()

results = reservoir.states[:-1]
temp = results[-1]
indeces = np.triu_indices(len(temp))
states = np.zeros((len(results), len(temp) + len(indeces[0])))
for i, res in enumerate(results):
    states[i] = np.concatenate((res, np.outer(res.T, res)[indeces]))

target = timeseries[1:]

warmup = 2 * M
states = states[warmup:]
target = target[warmup:]

model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1e-2))
model = RidgeClassifier(alpha=1e-2)


train_x, test_x, train_y, test_y = train_test_split(states, target, test_size=0.33)

clf = model.fit(train_x, train_y)
print(f'Score of model: {model.score(test_x, test_y)}')
utilities.result_plotter(states, target)
plt.show()
# plt.savefig("Beautiful85-1000.pdf")

