# class Layer:
import numpy as np

from abc import ABC, abstractmethod #For the abstract Layer base class
class Layer(ABC):

    # When this function is called on a layer,
    # it should return an integer overestimating the number
    # of measurements the layer is goind to perform.
    @abstractmethod
    def get_num_measurements(self, qreg, timeseries):
        return 0

    # Main build method for layer. Append operations to the
    # end of the circuit, and return the finished circuit.
    # **kwargs correspond directly to key-word-arguments provided
    # when initializing the reservoir, and are available for use.
    @abstractmethod
    def build(self, circuit, timeseries, **kwargs):
        return circuit


class Reset(Layer):
    def get_num_measurements(self, qreg, timeseries):
        return super().get_num_measurements(qreg, timeseries)
    def build(self, circuit, timeseries, **kwargs):
        circuit.reset(circuit.qubits)
        return circuit


class H(Layer):
    def get_num_measurements(self, qreg, timeseries):
        return super().get_num_measurements(qreg, timeseries)
    def build(self, circuit, timeseries, **kwargs):
        circuit.h(circuit.qubits)
        return circuit



class Measurement(Layer):
    def __init__(self, measure_qubits) -> None:
        super().__init__()
        self.measure_qubits = measure_qubits

    def get_num_measurements(self, qreg, timeseries):
        return len(self.measure_qubits)
    def build(self, circuit, timeseries, **kwargs):
        circuit.measure(self.measure_qubits)
        return circuit



class Timeseries(Layer):
    def __init__(self, build_method, **kwargs) -> None:
        super().__init__()
        self.build_method = build_method
        self.kwargs = kwargs

    def get_num_measurements(self, qreg, timeseries):
        return len(timeseries) * len(qreg)

    def build(self, circuit, timeseries,  **kwargs):
        for timestep in timeseries:
            self.build_method(circuit, timestep, **self.kwargs, **kwargs)

        return circuit