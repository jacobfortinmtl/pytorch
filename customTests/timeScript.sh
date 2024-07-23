#!/bin/bash
# This program will call the the testtime.py file with different env variables to compare the different times
# The output will be saved in the plots folder
executable="testtime.py"
num_procs=$(nproc)

# First, we loop through the number of cores to see speed changes
# Conversion=1 is to specify that we want IM2COL
for ((i=num_procs; i>=1; i=i-4))
do
    CONVERSION=1 OMP_NUM_THREADS=$i WRITE=1 FILENAME="convolution_cores_$i" python $executable
done

# One extra time for the core = 1 case
CONVERSION=1 OMP_NUM_THREADS=1 WRITE=1 FILENAME="convolution_cores_1" python $executable

# Second we loop through images filled with nans of ratios 0, 0.25, 0.5, 0.75, 1
for i in 0 0.25 0.5 0.75 0.99; do
    CONVERSION=1 WRITE=1 FILENAME="convolution_nan_ratio_$i" FRACTION=$i python $executable
done

# Second we loop through images filled with nans of ratios 0, 0.25, 0.5, 0.75, 1, but no reinsertion
for i in 0.99; do
    CONVERSION=1 WRITE=1 FILENAME="convolution_nan_ratio_no_insert$i" REINSERT=0 FRACTION=$i python $executable
done

# # For plotting of number of rows removed
# for i in 0 0.25 0.5 0.75 0.99; do
#     echo "NaN Fraction in image $i"
#     WRITE=1 FILENAME="convolution_nan_ratio_$i" FRACTION=$i python $executable
# done

# Running default
WRITE=1 FILENAME=convolution_torch_default DEFAULT=1 python3 $executable

# Plotting
python3 $executable