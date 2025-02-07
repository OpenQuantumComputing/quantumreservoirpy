#!/bin/bash

for j in {1..100}
do
    bash run_reservoir.sh 5 3 20 quantum_stab None 2 1 logistic $j
done

