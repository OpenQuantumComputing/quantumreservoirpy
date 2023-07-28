from sklearn.model_selection import train_test_split
import numpy as np
import json

class Experiment:
    def __init__(self, reservoir, models, timeseries, SHOTS=10000, WARMUP=0.1, num_experiments=1):
        self.reservoir = reservoir
        self.models = models

        self.timeseries = timeseries
        self.shots = SHOTS
        self.WARMUP = WARMUP
        self.num_experiments = num_experiments

    def run(self):

        self.states = self.reservoir.run(
            timeseries = self.timeseries,
            shots = self.shots
        )

        warmup_idx = int(len(self.states) * self.WARMUP)

        xstates = self.states[:-1][warmup_idx:]
        target = self.timeseries[1:][warmup_idx:]

        self.results = np.zeros(len(self.models))
        for _ in range(self.num_experiments):
            X_train, X_test, y_train, y_test = train_test_split(xstates, target, test_size=0.33)
            for i, model in enumerate(self.models):
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                self.results[i] += score
        self.results /= self.num_experiments
        self.save("test.json")

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)

    def save(self, filename):
        with open(filename, "wb") as file:
            file.write(self.toJSON())







