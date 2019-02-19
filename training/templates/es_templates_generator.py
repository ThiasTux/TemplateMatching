import random

import numpy as np
import progressbar

from performance_evaluation import fitness_functions as fit_fun
from template_matching.wlcss_cuda_class import WLCSSCudaTemplatesTraining


class ESTemplateGenerator:
    def __init__(self, instances, instances_labels, params, threshold, cls, chromosomes, bit_values, file=None,
                 use_encoding=False,
                 num_processes=1,
                 iterations=500,
                 num_individuals=32,
                 cr_p=0.3, mt_p=0.1, elitism=3, rank=10, maximize=True):
        self.__instances = instances
        self.__instances_labels = np.array(instances_labels).reshape((len(instances), 1))
        self.__params = params
        self.__threshold = threshold
        self.__class = cls
        self.__bit_values = bit_values
        self.__use_encoding = use_encoding
        self.__num_processes = num_processes
        self.__iterations = iterations
        self.__num_individuals = num_individuals
        self.__crossover_probability = cr_p
        self.__mutation_probability = mt_p
        self.__elitism = elitism
        self.__rank = rank
        self.__maximize = maximize
        self.__chromosomes = chromosomes
        self.__m_wlcss_cuda = WLCSSCudaTemplatesTraining(self.__instances, self.__params, self.__chromosomes,
                                                         self.__num_individuals, self.__use_encoding)
        self.__results = list()
        if file is None:
            self.__write_to_file = False
        else:
            self.__write_to_file = True
            self.__output_file = file

    def optimize(self):
        for t in range(self.__num_processes):
            self.__results.append(self.__execute_ga(t))

    def __execute_ga(self, num_test):
        scores = list()
        best_templates = list()
        pop = self.__generate_population()
        bar = progressbar.ProgressBar(max_value=self.__iterations)
        fit_scores = self.__compute_fitness_cuda(pop)
        for i in range(self.__iterations):
            pop_sort_idx = np.argsort(-fit_scores if self.__maximize else fit_scores)
            top_individuals = pop[pop_sort_idx]
            selected_population = self.__selection(top_individuals, self.__rank)
            crossovered_population = self.__crossover(selected_population, self.__crossover_probability)
            pop = self.__mutation(crossovered_population, self.__mutation_probability)
            if self.__elitism > 0:
                pop[0:self.__elitism] = top_individuals[0:self.__elitism]
            fit_scores = self.__compute_fitness_cuda(pop)
            if self.__maximize:
                top_idx = np.argmax(fit_scores)
            else:
                top_idx = np.argmin(fit_scores)
            best_template = pop[top_idx]
            scores.append([np.mean(fit_scores), np.max(fit_scores), np.min(fit_scores), np.std(fit_scores)])
            best_templates.append(best_template)
            bar.update(i)
        bar.finish()
        if self.__maximize:
            top_idx = np.argmax(fit_scores)
        else:
            top_idx = np.argmin(fit_scores)
        top_score = fit_scores[top_idx]
        best_template = pop[top_idx]
        if self.__write_to_file:
            output_scores_path = "{}_{:02d}_{}_scores.txt".format(self.__output_file, num_test, self.__class)
            with open(output_scores_path, 'w') as f:
                for item in scores:
                    f.write("%s\n" % str(item).replace("[", "").replace("]", ""))
            output_templates_path = "{}_{:02d}_{}_templates.txt".format(self.__output_file, num_test, self.__class)
            with open(output_templates_path, 'w') as f:
                for item in best_templates:
                    f.write("%s\n" % str(item).replace("[", "").replace("]", ""))
        self.__m_wlcss_cuda.cuda_freemem()
        return [top_score, best_template]

    def __generate_population(self):
        return np.random.randint(0, self.__bit_values, size=(self.__num_individuals, self.__chromosomes))

    def __selection(self, top_individuals, rnk):
        top_individuals = top_individuals[0:rnk]
        reproduced_individuals = np.array(
            [top_individuals[i % len(top_individuals)] for i in range(self.__num_individuals)])
        np.random.shuffle(reproduced_individuals)
        return reproduced_individuals

    def __crossover(self, pop, cp):
        new_pop = np.empty(pop.shape, dtype=int)
        for i in range(0, len(pop) - 1, 2):
            if np.random.random() < cp:
                crossover_position = random.randint(0, self.__chromosomes - 2)
                new_pop[i] = np.append(pop[i][0:crossover_position], pop[i + 1][crossover_position:])
                new_pop[i + 1] = np.append(pop[i + 1][0:crossover_position], pop[i][crossover_position:])
            else:
                new_pop[i] = pop[i]
                new_pop[i + 1] = pop[i + 1]
        return new_pop

    def __mutation(self, pop, mp):
        sizes = pop.shape
        mask = np.random.rand(pop.shape[0], pop.shape[1]) < mp
        new_pop_mask = np.random.randint(0, self.__bit_values, size=sizes) * mask
        new_pop = np.copy(pop)
        new_pop[mask] = new_pop_mask[mask]
        return new_pop

    def __compute_fitness_cuda(self, pop):
        matching_scores = self.__m_wlcss_cuda.compute_wlcss(pop)
        matching_scores = [np.concatenate((ms, self.__instances_labels), axis=1) for ms in matching_scores]
        fitness_scores = np.array([fit_fun.isolated_fitness_function_templates(matching_scores[0][:, k],
                                                                               matching_scores[0][:, -1],
                                                                               self.__threshold,
                                                                               parameter_to_optimize=4) for k in
                                   range(self.__num_individuals)])
        return fitness_scores

    def get_results(self):
        return self.__results
