#!/bin/bash

for name in {"henon","logistic"}
do
    for i in {1..6}
    do
        sbatch run_reservoir_degree.sh 7 6 20 quantum_stab None 4 1 $name $i fixed_stab 1 1000
done
