#!/bin/bash

for INPUT in data/a.csv data/b.csv data/c.csv
do
    echo "Dataset $INPUT"
    for MODE in 0 1 2
    do
        echo "Mode $MODE"
        for THREADS in 1 2 4
        do
            echo "Threads $THREADS"
            OMP_NUM_THREADS=$THREADS ./kmeans.exe --k 10 --input $INPUT --mode $MODE
        done
    done
done