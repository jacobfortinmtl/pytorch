#!/bin/bash
# This program will call the the testtime.py file with different env variables to compare the different times
# The output will be saved in the plots folder
executable="testtime.py"

# Ensure to use the executable variable when calling python3
WRITE=1 FILENAME=convolution_times_parallel python3 $executable
WRITE=1 FILENAME=convolution_times_sequential SEQUENTIAL=1 python3 $executable
WRITE=1 FILENAME=convolution_times_torch DEFAULT=1 python3 $executable
python3 $executable