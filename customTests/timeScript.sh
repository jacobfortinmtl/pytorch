#!/bin/bash
# This program will call the the testtime.py file with different env variables to compare the different times
# The output will be saved in the plots folder
executable="testtime.py"
num_procs=$(nproc)

# First, we loop through the number of cores to see speed changes
for ((i=num_procs; i>=1; i=i-4))
do
    OMP_NUM_THREADS=$i WRITE=1 FILENAME="convolution_cores_$i" python $executable
done

# One extra time for the core = 1 case
OMP_NUM_THREADS=1 WRITE=1 FILENAME="convolution_cores_1" python $executable

# Second we loop through images filled with nans of ratios 0, 0.25, 0.5, 0.75, 1
for i in 0 0.25 0.5 0.75 0.99; do
    WRITE=1 FILENAME="convolution_nan_ratio_$i" FRACTION=$i python $executable
done

# Second we loop through images filled with nans of ratios 0, 0.25, 0.5, 0.75, 1
for i in 0 0.25 0.5 0.75 0.99; do
    WRITE=1 FILENAME="convolution_nan_ratio_no_insert$i" REINSERT=0 FRACTION=$i python $executable
done


# Ensure to use the executable variable when calling python3
WRITE=1 FILENAME=convolution_torch_default DEFAULT=1 python3 $executable
python3 $executable