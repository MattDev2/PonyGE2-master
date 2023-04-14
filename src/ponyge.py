#! /usr/bin/env python

# PonyGE2
# Copyright (c) 2017 Michael Fenton, James McDermott,
#                    David Fagan, Stefan Forstenlechner,
#                    and Erik Hemberg
# Hereby licensed under the GNU GPL v3.
""" Python GE implementation """

from utilities.algorithm.general import check_python_version

check_python_version()

import matplotlib.pyplot as plt

from stats.stats import get_stats
from algorithm.parameters import params, set_params
import sys
import time
import numpy as np
import random
import os

from utilities.stats import trackers

def clear_trackers():
    """Utilities for tracking progress of runs, including time taken per
    generation, fitness plots, fitness caches, etc."""
    trackers.cache = {}
    trackers.runtime_error_cache = []
    trackers.best_fitness_list = []
    trackers.best_test_fitness_list = []
    trackers.first_pareto_list = []
    #trackers.time_list = []
    trackers.stats_list = []
    trackers.best_ever = None
    # Store the best ever individual here.


def append_to_file(file_path, float_value, recreate = False):
    if recreate and os.path.exists(file_path):
        os.remove(file_path)
    # Open or create the file in append mode
    with open(file_path, 'a') as file:
        # Write the float value to the file, converting it to a string first
        file.write(str(float_value) + "\n")  # The "\n" adds a newline character after the float value

def mane():
    """ Run program """
    set_params(sys.argv[1:])  # exclude the ponyge.py arg itself

    #seeds = np.random.randint(1, 4294967295, 3)
    seeds = [2131231231131]
    if len(seeds) == 1:
        random.seed(seeds[0])
        individuals = params['SEARCH_LOOP']()

        # Print final review
        get_stats(individuals, end=True)

        print(((trackers.best_fitness_list)))
        print(((trackers.best_test_fitness_list)))
        
        plt.plot(trackers.best_test_fitness_list)

        # Add title and labels
        plt.title('Line Plot of Values')
        plt.xlabel('Index')
        plt.ylabel('Value')

        # Display the plot
        plt.savefig('line_plot.png')

    else:
        for i, seed in enumerate(seeds):
            params['ITERATION_INDEX'] = i
            random.seed(int(seed))
            # Run evolution
            individuals = params['SEARCH_LOOP']()
            # Print final review
            print(seed)
            
            get_stats(individuals, end=True)
            
            # append_to_file(params['TRAIN_PERFORMANCE_PATH'], trackers.best_ever.training_fitness)
            # append_to_file(params['TEST_PERFORMANCE_PATH'], trackers.best_ever.test_fitness)

            clear_trackers()

if __name__ == "__main__":
    mane()