from algorithm.parameters import params
from utilities.stats import trackers
import numpy as np


def get_best_test(individuals):
    best_individual_fitness = None
    best_individual = None
    for ind in individuals:
        if ind.phenotype is not None:
            curr_fitness = params['FITNESS_FUNCTION'](ind, dist='test')
            if params['FITNESS_FUNCTION'].maximise:
                if best_individual_fitness is None or curr_fitness > best_individual_fitness:
                    best_individual_fitness = curr_fitness
                    best_individual = ind
            else:
                if best_individual_fitness is None or curr_fitness < best_individual_fitness:
                    best_individual_fitness = curr_fitness
                    best_individual = ind
    return best_individual, best_individual_fitness
