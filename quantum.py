import matplotlib.pyplot as plt
from qreservoirpy import utilities
from qiskit.circuit.library.standard_gates import IGate, XGate, ZGate, CXGate

circ = utilities.random_circuit(4, gates=[
    IGate, XGate, ZGate, CXGate
], depth=5)

circ.draw('mpl')
plt.show()
# states = reservoir.run(timeseries=[0, 1, 2, 3]*10, shots=10000)
# print(states)