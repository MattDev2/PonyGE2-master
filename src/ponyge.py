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

from fitness.supervised_learning.regression import regression

import sys
import time
import numpy as np
import random
import os
import pickle
from tqdm import tqdm

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

def mane(fold, hyperparam, seeds):
    """ Run program """
    fold = str(fold)
    train_set_path = f"Alzheimer/folds/train_data_fold_{fold}.csv"
    test_set_path = f"Alzheimer/folds/test_data_fold_{fold}.csv"

    params['DATASET_TRAIN'] = train_set_path
    params['DATASET_TEST'] = test_set_path

    if fold == '1':
        set_params(sys.argv[1:])  # exclude the ponyge.py arg itself
    else: 
        params['FITNESS_FUNCTION'] = regression()

    test_results = []
    train_results = []

    test_evolution = []
    train_evolution = []

    results = {}

    if len(seeds) == 1:
        random.seed(seeds[0])
        params[hyperparam['name']] = hyperparam['list'][0]
        individuals = params['SEARCH_LOOP']()

        # Print final review
        get_stats(individuals, end=True)

        print(((trackers.best_fitness_list)))
        print('\n\n\n')
        print(((trackers.best_test_fitness_list)))
        
        save_plot('line_plot.png', trackers.best_test_fitness_list)

    else:
        pbar = tqdm(total = len(seeds) * len(hyperparam['list']))
        for h in hyperparam['list']:
            for i, seed in enumerate(seeds):
                params['ITERATION_INDEX'] = i
                params[hyperparam['name']] = h

                random.seed(int(seed))
                # Run evolution
                individuals = params['SEARCH_LOOP']()
                # Print final review

                #print("Seed :",seed, "\n")
                #print(params['DATASET_TRAIN'])
                #print(params['DATASET_TEST'])

                get_stats(individuals, end=True)

                key = (i, h)
                results[key] = {
                    'test_result': trackers.best_ever.test_fitness,
                    'train_result': trackers.best_ever.training_fitness,
                    'test_evolution': trackers.best_test_fitness_list,
                    'train_evolution': trackers.best_fitness_list
                }

                clear_trackers()
                pbar.update(1)
            #pbar.update(len(seeds))
                
        fitness_shape = (len(seeds), len(hyperparam['list']))
        fitness_list_shapes = (*fitness_shape, params['GENERATIONS'] + 1)

        test_results = np.reshape([results[key]['test_result'] for key in sorted(results)], fitness_shape)
        train_results = np.reshape([results[key]['train_result'] for key in sorted(results)], fitness_shape)
        test_evolution = np.reshape([results[key]['test_evolution'] for key in sorted(results)], fitness_list_shapes)
        train_evolution = np.reshape([results[key]['train_evolution'] for key in sorted(results)], fitness_list_shapes)
        
        output_file_path = f"./results_data/results_data_fold_{fold}.pkl"
        
        data = {
            'hyperparam': hyperparam,
            'seeds': seeds,
            'test_results': test_results,
            'train_results': train_results,
            'test_evolution': test_evolution,
            'train_evolution': train_evolution
        }

        with open(output_file_path, 'wb') as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    
    hyperparameter_list = {"name": "MUTATION_PROBABILITY",
                            "list":np.linspace(0, 0.4, 10)}
    seeds = np.random.randint(1, 429467295, 50)

    for i in range(5):
        mane(i+1, hyperparameter_list, seeds)