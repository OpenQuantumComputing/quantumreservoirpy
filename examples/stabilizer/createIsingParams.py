from itertools import combinations
from quantumreservoirpy.util import randomIsing
import pickle

def createanddumpsising(num_qubits):
    steps = 1
    dt = 1.645
    num_reservoirs=20
    top = list(combinations(range(num_qubits), 2))
    isingparams = {}
    for nr in range(1, num_reservoirs + 1):
        (
            _,
            isingparams[nr],
        ) = randomIsing(num_qubits, top, steps, dt)
    with open("isingparams_"+"num_qubits"+str(num_qubits)+"_num_reservoirs"+str(num_reservoirs)+".pickle","wb") as f:
        pickle.dump(isingparams, f)

createanddumpsising(3)
createanddumpsising(4)
createanddumpsising(5)
