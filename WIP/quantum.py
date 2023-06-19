import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from QuantumReservoirpy.Reservoir import QReservoir
from QuantumReservoirpy import Layers

from QuantumReservoirpy.Experiment import Experiment

score = 0
reservoirs = []
timeseries = np.array([0, 1, 1, 2, 3] * 50)
for i in range(6):
    while True:
        res =  QReservoir(qubits=3, layers=[
                Layers.H(1),
                Layers.Warmup(5),
                Layers.Timeseries(measure_qubits=np.arange(2), timeseries_qubits=[0, 1], distance=0),
                Layers.Predict(num_predictions=25)
        ])

        model = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='linear'))
        expr = Experiment(reservoir=res, model=model)

        expr.run(timeseries, shots=1, warmup=25)
        score = expr.get_score()
        print(f'Result SVC:{i} = {round(100*score, 2)}%')
        if score >= 0.5:
            break
    reservoirs.append(res)


# circ = reservoir1.circuit(timeseries=[1, 2]* 3)
# circ.draw('mpl')
# plt.show()

model = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='linear'))
expr = Experiment(reservoir=reservoirs, model=model)
expr.run(timeseries, shots=1, warmup=25)
score = expr.get_score()
print(f'Result SVC = {round(100*score, 2)}%')

fig, ax = expr.get_feature_plot()
score = f'Result SVC = {round(100*score, 2)}%'
fig.suptitle(score)
print(f'Predictions made by model: {expr.get_predictions()}')
plt.show()

# states = reservoir.run(timeseries=[0, 1, 2, 3]*10, shots=10000)
# print(states)