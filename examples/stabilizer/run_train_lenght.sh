#!/bin/bash

for name in {"henon","logistic"}
do
    for i in {0..4}
    do
        sbatch run_reservoir.sh 7 5 20 classical None $i 1 $name 1 random 1 1000
        sbatch run_reservoir.sh 7 5 20 quantum_stab True $i 1 $name 1 random 1 1000
        sbatch run_reservoir.sh 7 5 20 quantum_stab True $i 1 $name 1 random 1 1000
done
