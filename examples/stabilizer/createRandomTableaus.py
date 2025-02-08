from quantumreservoirpy.stabilizer import Stabilizer
import pickle

def createanddumptableaus(num_qubits, num_measurements):
    num_tableaus = 100
    tableaus = {}
    for nr in range(1, num_tableaus+ 1):
        tableaus[nr] = Stabilizer.generate_tableau(num_qubits, num_measurements)
    with open("tableau_"+"num_qubits"+str(num_qubits)+"_num_measurements"+str(num_measurements)+"_num_tableaus"+str(num_tableaus)+".pickle","wb") as f:
        pickle.dump(tableaus, f)

createanddumptableaus(5,3)
createanddumptableaus(4,3)
createanddumptableaus(3,2)

