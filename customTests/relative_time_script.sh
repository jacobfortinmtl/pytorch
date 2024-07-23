sizes=(250 500 750 1000)

# Wipes the file before appending
> ../../plots/relative_increase.txt
# Loop over each size
for size in "${sizes[@]}"; do
  for i in {1..10}; do
    SIZE=$size python relative_increase_speed.py >> ../../plots/relative_increase.txt
  done
done
> ../../plots/relative_increase_default.txt
for size in "${sizes[@]}"; do
  for i in {1..10}; do
    SIZE=$size DEFAULT=1 python relative_increase_speed.py >> ../../plots/relative_increase_default.txt
  done
done
python relative_increase_speed_plotting.py
DEFAULT=1 python relative_increase_speed_plotting.py