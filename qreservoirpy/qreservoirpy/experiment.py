import numpy as np
import json

class Experiment:
    def __init__(self, reservoirs, models):
        self.reservoirs = reservoirs
        self.models = models

    def run(self,
            timeseries, # list of timeseries to be checked

            warmup_percentage=0,
            repeat=1):

        self.states = self.reservoir.run(
            timeseries = self.timeseries,
            shots = self.shots
        )

        warmup_idx = int(len(self.states) * self.WARMUP)

        xstates = self.states[:-1][warmup_idx:]
        target = self.timeseries[1:][warmup_idx:]

        self.results = np.zeros(len(self.models))

        # Measure performance
        train_test_split_sampling = 100
        for _ in range(train_test_split_sampling):
            X_train, X_test, y_train, y_test = train_test_split(xstates, target, test_size=0.33)
            for i, model in enumerate(self.models):
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                self.results[i] += score
        self.results /= train_test_split_sampling

        self.save("test.json")

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)

    def save(self, filename):
        with open(filename, "wb") as file:
            file.write(self.toJSON())







