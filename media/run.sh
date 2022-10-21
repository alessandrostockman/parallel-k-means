#!/bin/bash

./kmeans.exe --k 5 --input data/gaussian.csv --mode 0 --seed 4 --clusters-output out/cl_c_0.csv --centroids-output out/ce_c_0.csv --header
./kmeans.exe --k 5 --input data/gaussian.csv --mode 1 --seed 3 --clusters-output out/cl_c_1.csv --centroids-output out/ce_c_1.csv --header
./kmeans.exe --k 5 --input data/gaussian.csv --mode 2 --seed 2 --clusters-output out/cl_c_2.csv --centroids-output out/ce_c_2.csv --header


python python/plot.py