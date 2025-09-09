import numpy as np
import stim
from bitarray import bitarray


def fixed_weight_tableau(n_qubits, n_meas, weight, XYZ = False):

    """Generates a tableau in dict form, with n_meas pure Z stabilizers of given weight
    and pure X destabilizes of arbitrary weight
    This can be given as the argument "tableau" to the Stabilizer constructor
    
    If XYZ is False, will yield random Z stabilizers only"""

    if n_meas >= n_qubits:
        raise Exception(
            "n_meas should be strictly less than n_qubits"
        )
    if weight >= n_qubits:
        raise Exception(
            "Your Paulis are overweight :( weight should be strictly less than n_qubits"
        )
    
    # bistring generation in lexicographic order: https://stackoverflow.com/a/58072652
    def kbits(n, k):
        limit=1<<n
        val=(1<<k)-1
        while val<limit:
            yield "{0:0{1}b}".format(val,n)
            minbit=val&-val #rightmost 1 bit
            fillbit = (val+minbit)&~val  #rightmost 0 to the left of that bit
            val = val+minbit | (fillbit//(minbit<<1))-1

    Stabz = np.zeros(n_qubits)

    for val in kbits(n_qubits, weight):
        valarray = bitarray(val).tolist()
        Stabz = np.vstack((Stabz, valarray))

    Stabz = np.delete(Stabz, 0, 0)
    Stabz = Stabz.astype(int)
    
    rng = np.random.default_rng()
    rng.shuffle(Stabz)

            
    if XYZ:

        def commutes_with_all(Stabxyz_list, newstab):

            commutes = True
            index = 0

            while commutes & index<len(Stabxyz_list):
                anticommutations = (newstab != Stabxyz_list[index]) #TODO: just use qiskit lmao
                commutes = not sum(anticommutations) % 2
                index += 1

            return commutes

        PauliArray = rng.choice(range(1,4), n_qubits)
        mask = rng.choice(Stabz)
        Stabxyz = np.array([PauliArray*mask])

        while np.linalg.matrix_rank(Stabxyz) < n_meas: # TODO : check linear independence using qiskit symplectic notation or stim
            current_rank = np.linalg.rank(Stabxyz)
            while np.linalg.matrix_rank(Stabxyz) == current_rank:
                PauliArray = rng.choice(range(1,4), n_qubits)
                mask = rng.choice(Stabz)
                Stabxyz = np.vstack(Stabxyz, PauliArray*mask)

                #TODO : randomize signs BUT careful that they're not contradictory like +Z and -Z



    stimStabz = (3*Stabz).tolist() # In Stim, 0=I, 1=X, 2=Y, 3=Z
    stimStabz = [stim.PauliString(stab) for stab in stimStabz]

    tableau = stim.Tableau.from_stabilizers(stimStabz, allow_redundant=True, allow_underconstrained=True)

    # Translate to Qiskit and choose n_meas stabilizers
    stabilizer = [str(tableau.z_output(i)).replace('_','I') for i in range(n_meas)]
    destabilizer = [str(tableau.x_output(i)).replace('_','I') for i in range(n_meas)]
    tableau_dict = {"stabilizer" : stabilizer, "destabilizer" : destabilizer}

    return(tableau_dict)

# print(fixed_weight_tableau(10,5,5,XYZ=True))