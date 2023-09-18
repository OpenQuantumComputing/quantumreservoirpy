import numpy as np

from tqdm import tqdm

from .reservoirbase import QReservoir
from .util import memory_to_mean


class Static(QReservoir):
    def run(self, timeseries, **kwargs):
        transpile = kwargs.pop('transpile', True)
        circ = self.circuit(timeseries, merge_registers=False, transpile=transpile).reverse_bits()

        self._job = self.backend.run(circ, memory=True, **kwargs)
        mem = self._job.result().get_memory()
        avg = memory_to_mean(mem)

        states = avg.reshape((len(timeseries), -1))

        self.last_state = states[-1].ravel()

        return states


    def predict(self, num_pred, model, from_series, **kwargs):
        M = min(num_pred + len(from_series), self.memory)

        predictions = np.zeros(num_pred + len(from_series))
        predictions[:len(from_series)] = from_series

        for i in tqdm(range(num_pred), desc="Predicting..."):
            curidx = len(from_series) + i
            states = self.run(predictions[:curidx][-M:],  kwargs=kwargs)

            pred_state = states[-1].reshape((1, -1))
            predictions[curidx] = model.predict(pred_state)
            self.last_state = pred_state.ravel()

        return predictions[-num_pred:]

class Incremental(QReservoir):
    def __init__(self, n_qubits, memory=np.inf, backend=None, num_features=-1) -> None:
        super().__init__(n_qubits, memory, backend)

        if num_features > 0:
            self.num_features = num_features

    def run(self, timeseries, **kwargs):
        transpile = kwargs.pop('transpile', True)
        M = min(len(timeseries), self.memory)
        timeseries_splited = [timeseries[:i+1][-M:] for i in range(len(timeseries))]

        total = len(timeseries_splited)

        with tqdm(total=total) as pbar:
            pbar.set_description("Creating circuits...")
            circuits = [self.circuit(series, merge_registers=False, transpile=transpile).reverse_bits() for series in timeseries_splited]

            pbar.set_description("Running job...")
            self._job = self.backend.run(circuits, memory=True, **kwargs)

            result = self._job.result()

            if not hasattr(self, 'num_features'):
                self.num_features = memory_to_mean(result.get_memory(0)).size

            states = np.zeros((len(timeseries_splited), self.num_features))


            pbar.set_description("Analyzing... ")
            for idx, _ in enumerate(timeseries_splited):
                memory = self._job.result().get_memory(idx)

                avg = memory_to_mean(memory)[-self.num_features:]
                states[idx, self.num_features - len(avg):] = avg
                pbar.update(1)

            return states

    def __run(self, timeseries, **kwargs):
        transpile = kwargs.pop('transpile', True)
        circ = self.circuit(timeseries, merge_registers=False, transpile=transpile).reverse_bits()
        self._job = self.backend.run(circ, memory=True, **kwargs)
        results = self._job.result()

        avg = memory_to_mean(results.get_memory())
        if hasattr(self, 'num_features'):
            avg = avg[-self.num_features:]
            temp = np.zeros(self.num_features)
            temp[self.num_features - len(avg):] = avg
            return temp

        return avg

    def predict(self, num_pred, model, from_series, **kwargs):
        M = min(num_pred + len(from_series), self.memory)

        shape = np.array(np.shape(from_series))
        shape[0] += num_pred
        predictions = np.zeros(shape=shape)
        predictions[:len(from_series)] = from_series

        for i in tqdm(range(num_pred), desc="Predicting..."):
            curidx = len(from_series) + i
            states = self.__run(predictions[:curidx][-M:], kwargs=kwargs)
            pred_state = states.reshape((1, -1))
            predictions[curidx] = model.predict(pred_state)

        return predictions[-num_pred:]