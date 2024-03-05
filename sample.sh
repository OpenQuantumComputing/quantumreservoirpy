#!/bin/bash
#SBATCH --job-name=random_sample
#              d-hh:mm:ss
#SBATCH --time=30-00:00:00
#SBATCH --output=/home/franzf/quantumreservoirpy/%j.out
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20

source qiskit/bin/activate

python random_sample_all.py "$1" "$2" "$3" "$4"
