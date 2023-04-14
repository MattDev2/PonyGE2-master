import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def read_floats_from_file(file_path):
    with open(file_path, 'r') as file:
        floats = [float(line.strip()) for line in file]
    return np.array(floats)

def plot_violin(floats):
    sns.set(style='whitegrid')
    sns.violinplot(data=floats)
    plt.show()

# Example usage
file_path = '../results/performances/test_performance.txt'
float_array = read_floats_from_file(file_path)
print(float_array)
plot_violin(float_array)