# from QuantumReservoirpy.Reservoir import QReservoir
from QuantumReservoirpy import Reservoir
from QuantumReservoirpy import Layers
from QuantumReservoirpy import utilities

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np

class Experiment:
    def __init__(self, reservoir, model) -> None:
        self.reservoir = reservoir
        self.model = model

    def run(self, timeseries, shots=10000, warmup=5):
        result = self.reservoir.run(timeseries=timeseries, shots=shots)
        if len(result) == 1:
            self.states = result
        else:
            self.states, self.preds = result

        self.states = self.states[warmup:]
        self.target = timeseries[warmup + 1:]

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.states, self.target, test_size=0.33)
        self.model.fit(self.train_x, self.train_y)

    def get_score(self):
        try:
            self.score = self.model.score(self.test_x, self.test_y)
        except AttributeError:
            raise Exception("Need to run simulation first. Run Experiment.run()")

        return self.score

    def get_predictions(self):
        try:
            self.predictions = self.model.predict(self.preds)
        except AttributeError:
            raise Exception("Need to run simulation first. Run Experiment.run()")

        return self.predictions

    def get_feature_plot(self):
        return utilities.result_plotter(self.states, self.target)

