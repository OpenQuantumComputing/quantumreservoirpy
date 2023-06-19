# class Layer:
import numpy as np

from abc import ABC, abstractmethod #For the abstract Layer base class
class Layer(ABC):
    @abstractmethod
    def get_num_measurements(self, qreg):
        return 0

    @abstractmethod
    def build(self, circuit, **kwargs):
        return circuit

    @abstractmethod
    def add_timeseries(self, timeseries):
        return timeseries


class Reset(Layer):
    def get_num_measurements(self, qreg):
        return super().get_num_measurements(qreg)
    def build(self, circuit, **kwargs):
        circuit.reset(circuit.qubits)
        return circuit
    def add_timeseries(self, timeseries):
        return super().add_timeseries(timeseries)
class H(Layer):
    def get_num_measurements(self, qreg):
        return super().get_num_measurements(qreg)
    def build(self, circuit, **kwargs):
        circuit.h(circuit.qubits)
        return circuit
    def add_timeseries(self, timeseries):
        return super().add_timeseries(timeseries)

class Measurement(Layer):
    def __init__(self, measure_qubits) -> None:
        super().__init__()
        self.measure_qubits = measure_qubits

    def get_num_measurements(self, qreg):
        return len(self.measure_qubits)
    def build(self, circuit, **kwargs):
        circuit.measure(self.measure_qubits)
        return circuit
    def add_timeseries(self, timeseries):
        return super().add_timeseries(timeseries)

class Timeseries(Layer):
    def __init__(self, build_method, M=np.inf, incrementally=False, **kwargs) -> None:
        super().__init__()
        self.build_method = build_method
        self.timeseries = np.zeros(0)
        self.incrementally = incrementally
        self.M = M
        self.kwargs = kwargs

    def get_num_measurements(self, qreg):
        return len(self.timeseries) * len(qreg)

    def build(self, circuit, **kwargs):
        start_idx = max(0, len(self.timeseries) - 2 * self.M)
        for timestep in self.timeseries[start_idx:]:
            circuit = self.build_method(circuit, timestep, **self.kwargs, **kwargs)

        return circuit


    def add_timeseries(self, timeseries):
        if self.incrementally:
            self.timeseries = np.append(self.timeseries, timeseries[0])
            return timeseries[1:]
        else:
            self.timeseries = np.append(self.timeseries, timeseries)
            return []
