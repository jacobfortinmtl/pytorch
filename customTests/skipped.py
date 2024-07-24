import os
import matplotlib.pyplot as plt

def read_data_from_file(file_path):
    data = {}
    current_nan_fraction = None
    initial_windows = []
    convolutions_skipped = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('NaN Fraction in image'):
                if current_nan_fraction is not None:
                    data[current_nan_fraction] = (initial_windows, convolutions_skipped)
                    initial_windows = []
                    convolutions_skipped = []

                current_nan_fraction = float(line.split()[-1])
            elif line.startswith('Number of initial windows'):
                initial_windows.append(int(line.split()[-1]))
            elif line.startswith('Convolutions skipped removed'):
                convolutions_skipped.append(int(line.split()[-1]))

        if current_nan_fraction is not None:
            data[current_nan_fraction] = (initial_windows, convolutions_skipped)
    
    return data

def plot_data(data):
    plt.figure(figsize=(12, 8))
    for nan_fraction, (initial_windows, convolutions_skipped) in data.items():
        plt.plot(initial_windows, convolutions_skipped, 'o-', label=f'NaN Fraction: {nan_fraction}')

    plt.title('Convolutions Skipped vs. Initial Windows for Different NaN Fractions')
    plt.xlabel('Number of Initial Windows')
    plt.ylabel('Convolutions Skipped')
    plt.legend()
    plt.grid(True)
    # Disable scientific notation
    ax = plt.gca()  # Get current axis
    ax.ticklabel_format(style='plain', useOffset=False, axis='x')
    ax.ticklabel_format(style='plain', useOffset=False, axis='y')
    plt.savefig('../../plots/convolutions_skipped_nan_fractions.png')
    plt.show()

def main():
    file_path = '../../plots/rows_removed.txt'
    if os.path.exists(file_path):
        data = read_data_from_file(file_path)
        plot_data(data)
    else:
        print(f"File {file_path} does not exist.")

if __name__ == "__main__":
    main()
