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
import pickle

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

def load_data_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data['test_results'], data['train_results'], data['test_evolution'], data['train_evolution']

def save_plot(file_path, data):
        plt.plot(data)

        # Add title and labels
        plt.title('Line Plot of Values')
        plt.xlabel('Index')
        plt.ylabel('Value')

        # Display the plot
        plt.savefig(file_path)

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

    test_results = []
    train_results = []

    test_evolution = []
    train_evolution = []

    crossover_rates = np.linspace(0, 1, 3)
    #seeds = np.random.randint(1, 4294967295, 5)
    
    seeds = [74576657456]
    results = {}

    if len(seeds) == 1:
        random.seed(seeds[0])
        params['CROSSOVER_PROBABILITY'] = crossover_rates[0]
        individuals = params['SEARCH_LOOP']()

        # Print final review
        get_stats(individuals, end=True)

        print(((trackers.best_fitness_list)))
        
        print('\n\n\n')

        print(((trackers.best_test_fitness_list)))
        
        save_plot('line_plot.png', trackers.best_test_fitness_list)

    else:
        for c in crossover_rates:
            for i, seed in enumerate(seeds):
                params['ITERATION_INDEX'] = i
                params['CROSSOVER_PROBABILITY'] = c

                random.seed(int(seed))
                # Run evolution
                individuals = params['SEARCH_LOOP']()
                # Print final review
                get_stats(individuals, end=True)

                key = (i, c)
                results[key] = {
                    'test_result': trackers.best_ever.test_fitness,
                    'train_result': trackers.best_ever.training_fitness,
                    'test_evolution': trackers.best_test_fitness_list,
                    'train_evolution': trackers.best_fitness_list
                }

                clear_trackers()
                
        fitness_shape = (len(seeds), len(crossover_rates))
        fitness_list_shapes = (*fitness_shape, params['GENERATIONS'] + 1)

        test_results = np.reshape([results[key]['test_result'] for key in sorted(results)], fitness_shape)
        train_results = np.reshape([results[key]['train_result'] for key in sorted(results)], fitness_shape)
        test_evolution = np.reshape([results[key]['test_evolution'] for key in sorted(results)], fitness_list_shapes)
        train_evolution = np.reshape([results[key]['train_evolution'] for key in sorted(results)], fitness_list_shapes)
        
        file_path = 'data_results1.pkl'
        data = {
            'crossover_rates': crossover_rates,
            'seeds': seeds,
            'test_results': test_results,
            'train_results': train_results,
            'test_evolution': test_evolution,
            'train_evolution': train_evolution
        }

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)


        #save_data_pickle("data_results1.pkl", test_results, train_results, test_evolution, train_evolution)



if __name__ == "__main__":
    mane()
    #data = load_data_pickle("data_results.pkl")
    #print((data[3].shape))
    # Define the file path of the saved data
    file_path = 'data_results1.pkl'

    # Load the data
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)

    # Access the loaded data
    crossover_rates = loaded_data['crossover_rates']
    seeds = loaded_data['seeds']
    test_results = loaded_data['test_results']
    train_results = loaded_data['train_results']
    test_evolution = loaded_data['test_evolution']
    train_evolution = loaded_data['train_evolution']

    #print(test_results[1][2])

    # # Now you can use the loaded data as needed, for example:
    # print("Crossover rates:", crossover_rates)
    # print("Seeds:", seeds)
    # print("Test results shape:", test_results.shape)
    # print("Train results shape:", train_results.shape)
    # print("Test evolution shape:", test_evolution.shape)
    # print("Train evolution shape:", train_evolution.shape)

