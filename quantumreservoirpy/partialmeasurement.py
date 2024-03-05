from itertools import combinations
import numpy as np

import inspect

from quantumreservoirpy.util import randomIsing
from quantumreservoirpy.reservoirs import Static


class PartialMeasurement(Static):
    def __init__(
        self,
        n_qubits,
        n_meas,
        memory=np.inf,
        backend=None,
        degree=1,
        basis="X",
        num_reservoirs=1,
    ) -> None:
        super().__init__(
            n_qubits, memory, backend, degree=degree, num_reservoirs=num_reservoirs
        )
        self.n_meas = n_meas

        self.steps = 1
        self.dt = 1.645
        self.top = list(combinations(range(n_qubits), 2))
        self.basis = basis
        self.U = {}
        self.Jx = {}
        self.Jz = {}
        self.hx = {}
        self.hy = {}
        self.hz = {}
        for nr in range(1, num_reservoirs + 1):
            (
                self.U[nr],
                self.Jx[nr],
                self.Jz[nr],
                self.hx[nr],
                self.hy[nr],
                self.hz[nr],
            ) = randomIsing(n_qubits, self.top, self.steps, self.dt)

    def get_params(self):
        return (
            self.steps,
            self.dt,
            self.top,
            self.Jx,
            self.Jz,
            self.hx,
            self.hy,
            self.hz,
            self.basis,
        )

    # def __reduce_ex__(self, protocol):
    #    print("reduce_ex called")
    #    current_frame = inspect.currentframe()
    #    calling_frame = inspect.getouterframes(current_frame, 2)[1]
    #    calling_function_name = calling_frame.function
    #    print(f"This function was called by: {calling_function_name}")
    #    return (
    #        self.__class__,
    #        (
    #            self.n_qubits,
    #            self.memory,
    #            None,
    #            self.degree,
    #            self.num_reservoirs,
    #        ),
    #        {
    #            "other_params": (
    #                self.steps,
    #                self.dt,
    #                self.top,
    #                self.basis,
    #                self.Jx,
    #                self.Jz,
    #                self.hx,
    #                self.hy,
    #                self.hz,
    #            )
    #        },
    #    )

    def during(self, circuit, timestep, reservoirnumber):
        # encode
        # for k in range(self.n_meas):
        #    beta = 3**k
        #    circuit.rx(-beta / 2 * np.pi * timestep, k)
        circuit.rx(np.pi * timestep, 0)

        # reservoir
        circuit.append(self.U[reservoirnumber], range(self.n_qubits))

        # decode
        if self.basis == "Z":
            circuit.h(range(self.n_meas))
        elif self.basis == "Y":
            circuit.h(range(self.n_meas))
            circuit.sdg(range(self.n_meas))
        circuit.measure(range(self.n_meas))
        circuit.reset(range(self.n_meas))
