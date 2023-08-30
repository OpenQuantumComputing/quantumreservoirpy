import numpy as np
def listify(elem):
    try:
        return list(elem)
    except:
        return [elem]


def memory_to_mean(memory):
    """ Utility for analyzing qiskit.Result.get_memory() memory.
    Assumes the data is a list indexed by shots.
    """

    states = [" ".join(mem).split() for mem in memory]
    numb = np.array(states, dtype=int)
    return np.average(numb, axis=0)