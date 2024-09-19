#!/bin/bash

for name in {"narma"}
do

    for i in {0..4}
    do
        sbatch run_reservoir_narma.sh 3 2 20 classical None $i 1 $name 3
        sbatch run_reservoir_narma.sh 3 2 20 quantum_part None $i 1 $name 3
        sbatch run_reservoir_narma.sh 3 2 20 quantum_stab None $i 1 $name 3
        sbatch run_reservoir_narma.sh 3 2 20 quantum_part True $i 1 $name 3
        sbatch run_reservoir_narma.sh 3 2 20 quantum_stab True $i 1 $name 3

        for j in {2..3}
        do
            sbatch run_reservoir_narma.sh 4 $j 20 classical None $i 1 $name 3
            sbatch run_reservoir_narma.sh 4 $j 20 quantum_part None $i 1 $name 3
            sbatch run_reservoir_narma.sh 4 $j 20 quantum_stab None $i 1 $name 3 
            sbatch run_reservoir_narma.sh 4 $j 20 quantum_part True $i 1 $name 3
            sbatch run_reservoir_narma.sh 4 $j 20 quantum_stab True $i 1 $name 3
        done


        for j in {2..4}
        do
            sbatch run_reservoir_narma.sh 5 $j 20 classical None $i 1 $name 5
            sbatch run_reservoir_narma.sh 5 $j 20 quantum_part None $i 1 $name 5
            sbatch run_reservoir_narma.sh 5 $j 20 quantum_stab None $i 1 $name 5
            sbatch run_reservoir_narma.sh 5 $j 20 quantum_part True $i 1 $name 5
            sbatch run_reservoir_narma.sh 5 $j 20 quantum_stab True $i 1 $name 5 
        done

    done

    sbatch run_reservoir_narma.sh 3 2 20 quantum_part None $i 0 $name 5
    sbatch run_reservoir_narma.sh 3 2 20 quantum_stab None $i 0 $name 5
    sbatch run_reservoir_narma.sh 4 3 20 quantum_part None $i 0 $name 5
    sbatch run_reservoir_narma.sh 4 3 20 quantum_stab None $i 0 $name 5

done
