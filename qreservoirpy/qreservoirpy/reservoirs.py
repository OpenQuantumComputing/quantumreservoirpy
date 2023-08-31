from .qreservoir import QReservoir
from .util import memory_to_mean

import numpy as np
from tqdm import tqdm

class Static(QReservoir):
    def run(self, timeseries, **kwargs):
        circ = self.circuit(timeseries, merge_registers=False).reverse_bits()
        self._job = self.backend.run(circ, memory=True, **kwargs)
        mem = self._job.result().get_memory()
        avg = memory_to_mean(mem)

        return avg.reshape((len(timeseries), -1))


    def predict(self, num_pred, model, from_series, **kwargs):
        M = min(num_pred + len(from_series), self.memory)

        predictions = np.zeros(num_pred + len(from_series))
        predictions[:len(from_series)] = from_series

        for i in tqdm(range(num_pred), desc="Predicting..."):
            curidx = len(from_series) + i
            states = self.run(predictions[:curidx][-M:], kwargs=kwargs)
            pred_state = states[-1].reshape((1, -1))
            predictions[curidx] = model.predict(pred_state)

        return predictions[-num_pred:]

class Incremental(QReservoir):
    def run(self, timeseries, **kwargs):

        M = min(len(timeseries), self.memory)
        timeseries_splited = [timeseries[:i+1][-M:] for i in range(len(timeseries))]

        total = len(timeseries_splited)

        with tqdm(total=total) as pbar:
            pbar.set_description("Creating circuits...")
            circuits = [self.circuit(series, merge_registers=False).reverse_bits() for series in timeseries_splited]

            pbar.set_description("Running job...")
            self._job = self.backend.run(circuits, memory=True, **kwargs)

            result = self._job.result()
            num_features = memory_to_mean(result.get_memory(0)).size
            states = np.zeros((len(timeseries_splited), num_features))


            pbar.set_description("Analyzing... ")
            for idx, _ in enumerate(timeseries_splited):
                memory = self._job.result().get_memory(idx)
                states[idx] = memory_to_mean(memory)
                pbar.update(1)

            return states

    def __run(self, timeseries, **kwargs):
        circ = self.circuit(timeseries, merge_registers=False).reverse_bits()
        self._job = self.backend.run(circ, memory=True, **kwargs)
        results = self._job.result()
        return memory_to_mean(results.get_memory())

    def predict(self, num_pred, model, from_series, **kwargs):
        M = min(num_pred + len(from_series), self.memory)

        predictions = np.zeros(num_pred + len(from_series))
        predictions[:len(from_series)] = from_series

        for i in tqdm(range(num_pred), desc="Predicting..."):
            curidx = len(from_series) + i
            states = self.__run(predictions[:curidx][-M:], kwargs=kwargs)
            pred_state = states.reshape((1, -1))
            predictions[curidx] = model.predict(pred_state)

        return predictions[-num_pred:]