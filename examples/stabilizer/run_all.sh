#!/bin/bash

for name in {"henon","logistic"}
do

    for i in {0..4}
    do
        sbatch run_reservoir.sh 3 2 20 classical None $i 1 $name
        sbatch run_reservoir.sh 3 2 20 quantum_part None $i 1 $name
        sbatch run_reservoir.sh 3 2 20 quantum_stab None $i 1 $name
        sbatch run_reservoir.sh 3 2 20 quantum_part True $i 1 $name
        sbatch run_reservoir.sh 3 2 20 quantum_stab True $i 1 $name

        for j in {2..3}
        do
            sbatch run_reservoir.sh 4 $j 20 classical None $i 1 $name
            sbatch run_reservoir.sh 4 $j 20 quantum_part None $i 1 $name
            sbatch run_reservoir.sh 4 $j 20 quantum_stab None $i 1 $name
            sbatch run_reservoir.sh 4 $j 20 quantum_part True $i 1 $name
            sbatch run_reservoir.sh 4 $j 20 quantum_stab True $i 1 $name
        done


        for j in {2..4}
        do
            sbatch run_reservoir.sh 5 $j 20 classical None $i 1 $name
            sbatch run_reservoir.sh 5 $j 20 quantum_part None $i 1 $name
            sbatch run_reservoir.sh 5 $j 20 quantum_stab None $i 1 $name
            sbatch run_reservoir.sh 5 $j 20 quantum_part True $i 1 $name
            sbatch run_reservoir.sh 5 $j 20 quantum_stab True $i 1 $name
        done

    done

    sbatch run_reservoir.sh 3 2 20 quantum_part None $i 0 $name
    sbatch run_reservoir.sh 3 2 20 quantum_stab None $i 0 $name
    sbatch run_reservoir.sh 4 3 20 quantum_part None $i 0 $name
    sbatch run_reservoir.sh 4 3 20 quantum_stab None $i 0 $name

done
