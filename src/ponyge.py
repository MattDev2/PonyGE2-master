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

from fitness.supervised_learning.classification import classification 
from itertools import product

import sys
import time
import numpy as np
import random
import os
import pickle
import pandas as pd
from tqdm import tqdm

from utilities.stats import trackers

SAVE_CSV = False,

def clear_trackers():
    """Utilities function for clearing the trackers, in case of multiple runs."""
    trackers.cache = {}
    trackers.runtime_error_cache = []
    trackers.best_fitness_list = []
    trackers.best_test_fitness_list = []
    trackers.first_pareto_list = []
    #trackers.time_list = []
    trackers.stats_list = []
    trackers.best_ever = None

    trackers.best_ever_test = None
    # Store the individual with the best test fitness here.

    trackers.best_test_ind_fitness_list = []
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

def append_to_file(file_path, value, recreate = False):
    if recreate and os.path.exists(file_path):
        os.remove(file_path)
    # Open or create the file in append mode
    with open(file_path, 'a') as file:
        # Write the float value to the file, converting it to a string first
        file.write(value + "\n")  # The "\n" adds a newline character after the float value

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
        params['FITNESS_FUNCTION'] = classification()

    test_results = []
    train_results = []

    test_evolution = []
    train_evolution = []

    readable_results = {'Phenotype' : [], 'Genotype': [], 'Train_performance': [], 'Test_performance': []}
    
    results = {}

    if len(seeds) == 1:

        random.seed(int(seeds[0]))
        # params['MUTATION_PROBABILITY'] = 0
        # params['CROSSOVER_PROBABILITY'] = 0.9
        params['TOURNAMENT_SIZE'] = 8
        # params[hyperparam['name']] = hyperparam['list'][0]
        
        
        individuals = params['SEARCH_LOOP']()

        # Print final review
        get_stats(individuals, end=True)

        #print(((trackers.best_fitness_list)))
        #print('\n\n\n')
        #print(((trackers.best_test_fitness_list)))
        
        print(trackers.best_test_fitness_list)
        print(trackers.best_test_ind_fitness_list)
        print(trackers.best_ever_test)

        save_plot('line_plot.png', trackers.best_test_fitness_list)


    else:
        #Generate all combinations of hyperparameter values
        hyperparam_combinations = list(product(*[hyperparam['list'] for hyperparam in hyperparameter_list]))
        total_iterations = len(seeds) * len(hyperparam_combinations)
        pbar = tqdm(total = total_iterations)

        for i, seed in enumerate(seeds):
            params['ITERATION_INDEX'] = i
            
            for h_combination in hyperparam_combinations:
                for hyperparam, h in zip(hyperparameter_list, h_combination):
                    params[hyperparam['name']] = h

                print(seed)
                random.seed(int(seed))
                # Run evolution
                individuals = params['SEARCH_LOOP']()
                # Print final review

                #print("Seed :",seed, "\n")
                #print(params['DATASET_TRAIN'])
                #print(params['DATASET_TEST'])

                get_stats(individuals, end=True)

                key = (i, h_combination)
                results[key] = {
                    'test_result': trackers.best_ever.test_fitness,
                    'train_result': trackers.best_ever.training_fitness,
                    'test_evolution': trackers.best_test_fitness_list,
                    'train_evolution': trackers.best_fitness_list
                }  
                
                if SAVE_CSV: 
                    readable_results['Phenotype'].append(trackers.best_ever.phenotype)
                    readable_results['Genotype'].append(trackers.best_ever.genome)
                    readable_results['Train_performance'].append(trackers.best_ever.training_fitness)
                    readable_results['Test_performance'].append(trackers.best_ever.test_fitness)
                
                clear_trackers()
                pbar.update(1)

        # # Update the fitness_shape and fitness_list_shapes to account for the new hyperparameters
        fitness_shape = (len(seeds), len(hyperparam_combinations))
        fitness_list_shapes = (*fitness_shape, params['GENERATIONS'] + 1)

        test_results = np.reshape([results[key]['test_result'] for key in sorted(results)], fitness_shape)
        train_results = np.reshape([results[key]['train_result'] for key in sorted(results)], fitness_shape)
        test_evolution = np.reshape([results[key]['test_evolution'] for key in sorted(results)], fitness_list_shapes)
        train_evolution = np.reshape([results[key]['train_evolution'] for key in sorted(results)], fitness_list_shapes)

        output_file_path = f"./results_data/results_data_TOURNSIZE_fold_{fold}.pkl"

        data = {
            'hyperparameters': hyperparameter_list,
            'seeds': seeds,
            'test_results': test_results,
            'train_results': train_results,
            'test_evolution': test_evolution,
            'train_evolution': train_evolution
        }

        with open(output_file_path, 'wb') as f:
            pickle.dump(data, f)

        if SAVE_CSV:
            readable_results_df = pd.DataFrame(readable_results)
            readable_results_df.to_csv(f'readable_results_fold{fold}.csv', index=False)

if __name__ == "__main__":
    
    hyperparameter_list = [
        # {"name": "MUTATION_PROBABILITY", "list": np.linspace(0, 0.4, 5)},
        # {"name": "CROSSOVER_PROBABILITY", "list": np.linspace(0.5, 0.9, 5)},
        {"name": "TOURNAMENT_SIZE", "list": [6, 8, 10, 16, 24, 32, 64]
         }
        # Add more hyperparameters as needed
    ]
    
    seeds = np.random.randint(1, 429467295, 50)
    #seeds = [278354876, 263043811, 111391493, 102572394, 232678879, 230819519, 260742289, 413963402, 96987054, 85781585]

    #seeds = [4873487]
    for i in range(5):
        mane(i+1, hyperparameter_list, seeds)

