#!/bin/bash

for name in {"henon","logistic"}
do
    for i in {3..10}
    do
        sbatch run_reservoir.sh $i $(i-1) 20 classical None 4 200 1 $name 2 random 1
        sbatch run_reservoir.sh $i $(i-1) 20 quantum_part None 4 200 1 $name 2 random 1
        sbatch run_reservoir.sh $i $(i-1) 20 quantum_stab None 4 200 1 $name 2 random 1
done
