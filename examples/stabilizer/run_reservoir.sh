#!/bin/bash
#SBATCH --job-name=random_sample
#              d-hh:mm:ss
#SBATCH --time=30-00:00:00
#SBATCH --output=/home/franzf/quantumreservoirpy_vivaldi/%j.out
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1

source qiskit/bin/activate

python run_reservoir.py "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"

