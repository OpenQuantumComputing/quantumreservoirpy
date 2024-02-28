import numpy as np

# from multiprocessing import Pool # could be used for parallel execution of reservoirs in multiplexing approach

from itertools import combinations
from tqdm import tqdm

from quantumreservoirpy.reservoirbase import QReservoir
from quantumreservoirpy.util import memory_to_mean
from quantumreservoirpy.statistic import Statistic


class Static(QReservoir):
    def run(self, timeseries, **kwargs):
        transpile = kwargs.pop("transpile", True)
        tqdm_disable = kwargs.pop("tqdm", False)
        self.precision = kwargs.pop("precision", None)
        shots = kwargs.pop("shots", 10**3)

        num_timesteps = len(timeseries)
        self.shots_taken = {}

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

            self.stat = {}  # create new statistics
            shots_total = 0
            while True:
                kwargs["shots"] = shots
                self._job = self.backend.run(circ, **kwargs)
                counts = self._job.result().get_counts()

                shots_total += shots
                # print(shots, shots_total)
                states_list, var_list = self.measurementStatistics(
                    counts, num_timesteps
                )
                # print(shots_total, states_list[0])

                if not self.precision:
                    break
                else:
                    v = np.amax(np.concatenate(var_list))
                    # print("max var=", v)
                    shots = int(v / (self.precision) ** 2) - shots_total
                    if shots <= 0:
                        break

            states = (
                np.stack(states_list, axis=0)
                if nr == 1
                else np.hstack((states, np.stack(states_list, axis=0)))
            )
            self.variances = (
                np.stack(var_list, axis=0)
                if nr == 1
                else np.hstack((self.variances, np.stack(var_list, axis=0)))
            )
            self.shots_taken[nr] = shots_total

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

    def measurementStatistics(self, counts, num_timesteps):
        states_list = []
        var_list = []

        # number of measurements per time step
        num_meas_pt = int(len(list(counts.keys())[0]) / num_timesteps)

        for t in range(num_timesteps):
            sl = []
            vl = []
            indices = range(num_meas_pt)
            for k in range(1, self.degree + 1):
                for O in combinations(indices, k):
                    statkey = str(t) + " " + str(O)
                    self.stat.setdefault(statkey, Statistic())
                    Static.__add_counts(self.stat[statkey], O, counts, t)
                    sl.append(self.stat[statkey].get_E())
                    vl.append(self.stat[statkey].get_Variance())
            states_list.append(np.array(sl))
            var_list.append(np.array(vl))

        return states_list, var_list

    @staticmethod
    def __add_counts(stat, Obs, counts, t):
        # tmp_counts=0
        # tmp_vals=[]
        for key, count in counts.items():
            key_t = key.split()[t]
            val = (
                0.5
                - np.prod(1 - 2.0 * np.array([int(key_t[ind_O]) for ind_O in Obs])) / 2
            )
            stat.add_sample(val, count)
            # tmp_counts+=count
            # tmp_vals.append(val*count)
        # print(" -> ", np.var(np.array(tmp_vals)))


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
