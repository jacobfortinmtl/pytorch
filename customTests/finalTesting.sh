executable="finaltest.py"
num_procs=$(nproc)

# Loop 10 times calling test.py
# Conversion is to specify im2col, but still use our custom implementation for nans. If we only set default=1, it will use the torch default all the way.
> ../../plots/FINAL_TIMING.txt
for i in {1..10}; do
    python3 $executable >> ../../plots/FINAL_TIMING.txt
done

> ../../plots/FINAL_TIMING_DEFAULT.txt
for i in {1..10}; do
    DEFAULT=1 python3 $executable >> ../../plots/FINAL_TIMING_DEFAULT.txt
done
python finaltest_plotting.py
DEFAULT=1 python finaltest_plotting.py