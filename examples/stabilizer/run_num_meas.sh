#!/bin/bash

for name in {"henon","logistic"}
do
    for i in {1..7}
    do
        sbatch run_reservoir.sh 7 $i 20 classical None 4 1 $name 1 random 1 10000
        sbatch run_reservoir.sh 7 $i 20 quantum_part None 4 1 $name 1 random 1 10000
        sbatch run_reservoir.sh 7 $i 20 quantum_stab None 4 1 $name 1 random 1 10000
done
