import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib.pyplot as plt
from QuantumReservoirpy.Reservoir import QReservoir
from QuantumReservoirpy import Layers
from QuantumReservoirpy.utilities import result_plotter

from QuantumReservoirpy.Experiment import Experiment

reservoir = QReservoir(qubits=4, layers=[
    Layers.H(1),
    Layers.Warmup(5),
    Layers.Timeseries(measure_qubits=np.arange(2), timeseries_qubits=[0, 1], distance=1),
    Layers.Predict(num_predictions=10)
])
model = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='linear'))
expr = Experiment(reservoir, model)

timeseries = np.array([0, 1, 2, 3]*40)

expr.run(timeseries, shots=1000, warmup=5)
fig, ax = expr.get_feature_plot()
fig.suptitle(f'Result = {round(100*expr.get_score(), 2)}%')
print(f'Predictions made by model: {expr.get_predictions()}')
plt.show()

# states = reservoir.run(timeseries=[0, 1, 2, 3]*10, shots=10000)
# print(states)