from quantumreservoirpy.stabilizer import Stabilizer
from quantumreservoirpy.fixed_weight_tableau import FixedWeightTableau
import pickle

def createanddumptableaus(num_qubits, num_measurements,degree=None):
    num_tableaus = 100
    tableaus = {}
    for nr in range(1, num_tableaus+ 1):
        if degree is not None:
            tableaus[nr] = Stabilizer.generate_tableau(num_qubits, num_measurements,stab_method='degree', degree=degree)
            with open("tableau_"+"num_qubits"+str(num_qubits)+"_num_measurements"+str(num_measurements)+"degree"+str(degree)+"_num_tableaus"+str(num_tableaus)+".pickle","wb") as f:
                 pickle.dump(tableaus, f)
        else:
            tableaus[nr] = FixedWeightTableau.fixed_weight_tableau(num_qubits, num_meas, degree, XYZ = False)
            with open("tableau_"+"num_qubits"+str(num_qubits)+"_num_measurements"+str(num_measurements)+"_num_tableaus"+str(num_tableaus)+".pickle","wb") as f:
                pickle.dump(tableaus, f)

createanddumptableaus(5,3)
createanddumptableaus(4,3)
createanddumptableaus(3,2)
for num_qubits in range (0,8):
    for num_meas in range (1,num_qubits):
        for deg in range (1,num_qubits):
            createanddumptableaus(num_qubits,num_meas,degree=deg)
