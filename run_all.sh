#!/bin/bash

sbatch sample_logistic_map.sh 3 2 20 classical None
sbatch sample_logistic_map.sh 3 2 20 quantum_part None
sbatch sample_logistic_map.sh 3 2 20 quantum_stab None
sbatch sample_logistic_map.sh 3 2 20 quantum_part True
sbatch sample_logistic_map.sh 3 2 20 quantum_stab True

sbatch sample_logistic_map.sh 4 2 20 classical None
sbatch sample_logistic_map.sh 4 2 20 quantum_part None
sbatch sample_logistic_map.sh 4 2 20 quantum_stab None
sbatch sample_logistic_map.sh 4 2 20 quantum_part True
sbatch sample_logistic_map.sh 4 2 20 quantum_stab True

sbatch sample_logistic_map.sh 4 3 20 classical None
sbatch sample_logistic_map.sh 4 3 20 quantum_part None
sbatch sample_logistic_map.sh 4 3 20 quantum_stab None
sbatch sample_logistic_map.sh 4 3 20 quantum_part True
sbatch sample_logistic_map.sh 4 3 20 quantum_stab True

sbatch sample_logistic_map.sh 7 6 20 classical None
sbatch sample_logistic_map.sh 7 6 20 quantum_part None
sbatch sample_logistic_map.sh 7 6 20 quantum_stab None
sbatch sample_logistic_map.sh 7 6 20 quantum_part True
sbatch sample_logistic_map.sh 7 6 20 quantum_stab True

sbatch sample_logistic_map.sh 7 4 20 classical None
sbatch sample_logistic_map.sh 7 4 20 quantum_part None
sbatch sample_logistic_map.sh 7 4 20 quantum_stab None
sbatch sample_logistic_map.sh 7 4 20 quantum_part True
sbatch sample_logistic_map.sh 7 4 20 quantum_stab True

