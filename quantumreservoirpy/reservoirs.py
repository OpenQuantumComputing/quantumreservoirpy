import numpy as np

# from multiprocessing import Pool # could be used for parallel execution of reservoirs in multiplexing approach

from itertools import combinations
from tqdm import tqdm

from reservoirbase import QReservoir
from util import memory_to_mean
from statistic import Statistic


class Static(QReservoir):
    def run(self, timeseries, **kwargs):
        transpile = kwargs.pop("transpile", True)
        tqdm_disable = kwargs.pop("tqdm", False)

        for nr in tqdm(
            range(1, self.num_reservoirs + 1),
            desc="Running reservoirs",
            disable=tqdm_disable,
        ):
            circ = self.circuit(
                timeseries,
                merge_registers=False,
                transpile=transpile,
                reservoir_number=nr,
            ).reverse_bits()

            self._job = self.backend.run(circ, **kwargs)
            counts = self._job.result().get_counts()

            num_timesteps = len(timeseries)

            num_measq_pt = int(len(list(counts.keys())[0]) / num_timesteps)

            states_list_this = []

            for t in range(num_timesteps):
                indices = range(num_measq_pt)
                states_t = [self.__getE(O, counts, t) for k in range(1, self.degree + 1) for O in combinations(indices, k)]
                states_list_this.append(np.array(states_t))

            states = np.stack(states_list_this, axis=0) if nr == 1 else np.hstack((states, np.stack(states_list_this, axis=0)))

            #for t in range(num_timesteps):
            #    states_t = []
            #    indices = range(num_measq_pt)
            #    for k in range(1, self.degree + 1):
            #        for O in list(combinations(indices, k)):
            #            states_t.append(self.__getE(O, counts, t))
            #    states_list_this.append(np.array(states_t))
            #if nr == 1:
            #    states = np.stack(states_list_this, axis=0)
            #else:
            #    states = np.hstack((states, np.stack(states_list_this, axis=0)))

        self.last_state = states[-1].ravel()

        return states

    def predict(self, num_pred, model, from_series, **kwargs):
        kwargs["tqdm"] = True
        M = min(num_pred + len(from_series), self.memory)

        predictions = np.zeros(
            num_pred + len(from_series), dtype=np.array(from_series).dtype
        )

        if np.ndim(from_series) == 2:
            prediction_dimension = np.shape(from_series)[1]
            predictions = np.zeros(
                (num_pred + len(from_series), prediction_dimension),
                dtype=np.array(from_series).dtype,
            )

        predictions[: len(from_series)] = from_series

        for i in tqdm(range(num_pred), desc="Predicting..."):
            curidx = len(from_series) + i
            states = self.run(predictions[:curidx][-M:], **kwargs)

            pred_state = states[-1].reshape((1, -1))
            predictions[curidx] = model.predict(pred_state)

        return predictions, pred_state  # [-num_pred:]

    #def __getE(self, Obs, counts, t):
    #    stat = Statistic()
    #    stat.reset()
    #    # E=0
    #    totalcounts = 0
    #    for key in counts:
    #        val = 1
    #        totalcounts += counts[key]
    #        key_t = key.split()[t]
    #        # val = (-1)**(key_t[Obs].count("1"))
    #        for ind_O in Obs:
    #            val *= 1.0 - 2 * int(key_t[ind_O])
    #        stat.add_sample(val, counts[key])
    #        # E+=val*counts[key]
    #    return stat.get_E()  # /totalcounts

    def __getE(self, Obs, counts, t):
        stat = Statistic()
        stat.reset()
        totalcounts = 0

        for key, count in counts.items():
            totalcounts += count
            key_t = key.split()[t]
            val = np.prod(1.0 - 2 * np.array([int(key_t[ind_O]) for ind_O in Obs]))
            stat.add_sample(val, count)

        return stat.get_E()

class Incremental(QReservoir):
    def __init__(self, n_qubits, memory=np.inf, backend=None, num_features=-1) -> None:
        super().__init__(n_qubits, memory, backend)

        if num_features > 0:
            self.num_features = num_features

    def run(self, timeseries, **kwargs):
        transpile = kwargs.pop("transpile", True)
        M = min(len(timeseries), self.memory)
        timeseries_splitted = [timeseries[: i + 1][-M:] for i in range(len(timeseries))]

        total = len(timeseries_splitted)

        with tqdm(total=total) as pbar:
            pbar.set_description("Creating circuits...")
            circuits = [
                self.circuit(
                    series, merge_registers=False, transpile=transpile
                ).reverse_bits()
                for series in timeseries_splitted
            ]

            pbar.set_description("Running job...")
            self._job = self.backend.run(circuits, memory=True, **kwargs)

            result = self._job.result()

            if not hasattr(self, "num_features"):
                self.num_features = memory_to_mean(result.get_memory(0)).size

            states = np.zeros((len(timeseries_splitted), self.num_features))

            pbar.set_description("Analyzing... ")
            for idx, _ in enumerate(timeseries_splitted):
                memory = self._job.result().get_memory(idx)

                avg = memory_to_mean(memory)[-self.num_features :]
                states[idx, self.num_features - len(avg) :] = avg
                pbar.update(1)

            return states

    def __run(self, timeseries, **kwargs):
        transpile = kwargs.pop("transpile", True)
        circ = self.circuit(
            timeseries, merge_registers=False, transpile=transpile
        ).reverse_bits()
        self._job = self.backend.run(circ, memory=True, **kwargs)
        results = self._job.result()

        avg = memory_to_mean(results.get_memory())
        if hasattr(self, "num_features"):
            avg = avg[-self.num_features :]
            temp = np.zeros(self.num_features)
            temp[self.num_features - len(avg) :] = avg
            return temp

        return avg

    def predict(self, num_pred, model, from_series, **kwargs):
        M = min(num_pred + len(from_series), self.memory)

        shape = np.array(np.shape(from_series))
        shape[0] += num_pred
        predictions = np.zeros(shape=shape)
        predictions[: len(from_series)] = from_series

        for i in tqdm(range(num_pred), desc="Predicting..."):
            curidx = len(from_series) + i
            states = self.__run(predictions[:curidx][-M:], kwargs=kwargs)
            pred_state = states.reshape((1, -1))
            predictions[curidx] = model.predict(pred_state)

        return predictions[-num_pred:]
