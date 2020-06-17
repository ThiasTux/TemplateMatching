import random
from math import ceil, log2

import numpy as np
import progressbar

from performance_evaluation import fitness_functions as fit_fun
from template_matching.wlcss_cuda_class import WLCSSCudaTemplatesTraining


class ESTemplateGenerator:
    def __init__(self, steams, stream_labels, params, threshold, cls, chromosomes, bit_values,
                 chosen_template=None,
                 use_encoding=False,
                 num_processes=1,
                 iterations=500,
                 num_individuals=32,
                 cr_p=0.3, mt_p=0.1,
                 elitism=3, rank=10, fitness_function=7, maximize=True):
        self.__streams = steams
        self.__streams_labels = stream_labels
        self.__params = params
        self.__threshold = threshold
        self.__class = cls
        self.__bit_values = bit_values
        self.__use_encoding = use_encoding
        self.__chosen_template = chosen_template
        self.__num_processes = num_processes
        self.__iterations = iterations
        self.__num_individuals = num_individuals
        self.__crossover_probability = cr_p
        self.__mutation_probability = mt_p
        self.__elitism = elitism
        self.__rank = rank
        self.__fitness_function = fitness_function
        self.__maximize = maximize
        self.__templates_chromosomes = chromosomes
        self.__m_wlcss_cuda = WLCSSCudaTemplatesTraining(self.__streams, self.__params, self.__templates_chromosomes,
                                                         self.__num_individuals, self.__use_encoding)
        self.__results = list()

    def optimize(self):
        self.__execute_ga()

    def __execute_ga(self):
        scores = list()
        best_templates = list()
        templates_pop = self.__generate_population()
        bar = progressbar.ProgressBar(max_value=self.__iterations)
        fit_scores_distances = self.__compute_fitness_cuda(templates_pop)
        fit_scores = fit_scores_distances[:, 0]
        for i in range(self.__iterations):
            pop_sort_idx = np.argsort(-fit_scores if self.__maximize else fit_scores)
            top_templates_individuals = templates_pop[pop_sort_idx]
            templates_selected_population = self.__selection(top_templates_individuals, self.__rank)
            templates_crossovered_population = self.__crossover(templates_selected_population,
                                                                self.__crossover_probability)
            templates_pop = self.__mutation(templates_crossovered_population, self.__mutation_probability)
            if self.__elitism > 0:
                templates_pop[0:self.__elitism] = top_templates_individuals[0:self.__elitism]
            fit_scores_distances = self.__compute_fitness_cuda(templates_pop)
            fit_scores = fit_scores_distances[:, 0]
            good_distances = fit_scores_distances[:, 1]
            bad_distances = fit_scores_distances[:, 2]
            if self.__maximize:
                top_idx = np.argmax(fit_scores)
            else:
                top_idx = np.argmin(fit_scores)
            best_template = templates_pop[top_idx]
            scores.append([np.mean(fit_scores), np.max(fit_scores), np.min(fit_scores), np.std(fit_scores),
                           good_distances[top_idx], bad_distances[top_idx]])
            best_templates.append(best_template)
            bar.update(i)
        bar.finish()
        if self.__maximize:
            top_idx = np.argmax(fit_scores)
        else:
            top_idx = np.argmin(fit_scores)
        top_score = fit_scores[top_idx]
        best_template = templates_pop[top_idx]
        self.__m_wlcss_cuda.cuda_freemem()
        self.__results = [top_score, best_template, scores, best_templates]

    def __generate_population(self):
        templates_pop = np.random.randint(0, self.__bit_values,
                                          size=(self.__num_individuals, self.__templates_chromosomes))
        if self.__chosen_template is not None:
            templates_pop[0] = self.__chosen_template[:, 1]
        return templates_pop

    def __selection(self, top_templates_individuals, rnk):
        top_templates_individuals = top_templates_individuals[0:rnk]
        reproduced_templates_individuals = np.array(
            [top_templates_individuals[i % len(top_templates_individuals)] for i in range(self.__num_individuals)])
        np.random.shuffle(reproduced_templates_individuals)
        return reproduced_templates_individuals

    def __crossover(self, templates_pop, cp):
        new_templates_pop = np.empty(templates_pop.shape, dtype=int)
        for i in range(0, len(templates_pop) - 1, 2):
            if np.random.random() < cp:
                crossover_position = random.randint(0, self.__templates_chromosomes - 2)
                new_templates_pop[i] = np.append(templates_pop[i][0:crossover_position],
                                                 templates_pop[i + 1][crossover_position:])
                new_templates_pop[i + 1] = np.append(templates_pop[i + 1][0:crossover_position],
                                                     templates_pop[i][crossover_position:])
            else:
                new_templates_pop[i] = templates_pop[i]
                new_templates_pop[i + 1] = templates_pop[i + 1]
        return new_templates_pop

    def __mutation(self, templates_pop, mp):
        tmpl_sizes = templates_pop.shape
        mask = np.random.rand(templates_pop.shape[0], templates_pop.shape[1]) < mp
        new_templates_pop_mask = np.random.normal(0, 32, size=tmpl_sizes) * mask
        new_templates_pop = np.remainder(np.copy(templates_pop) + new_templates_pop_mask, self.__bit_values)

        return new_templates_pop.astype(np.int)

    def __compute_fitness_cuda(self, templates_pop):
        matching_scores = self.__m_wlcss_cuda.compute_wlcss(templates_pop)
        fitness_scores = np.array([fit_fun.isolated_fitness_function_templates(matching_scores[0][:, k],
                                                                               self.__streams_labels,
                                                                               self.__threshold,
                                                                               parameter_to_optimize=self.__fitness_function)
                                   for k in
                                   range(self.__num_individuals)])
        return fitness_scores

    def __np_to_int(self, chromosome):
        return int("".join(chromosome.astype('U')), 2)

    def get_results(self):
        return self.__results


class ESTemplateThresholdsGenerator:
    def __init__(self, streams, streams_labels, params, cls, chromosomes, bit_values, chosen_template=None,
                 use_encoding=False, iterations=500, num_individuals=32, cr_p=0.3, mt_p=0.1, elitism=3, rank=10,
                 fitness_function=7, maximize=True):
        self.__streams = streams
        self.__streams_labels = streams_labels
        self.__params = params
        self.__class = cls
        self.__bit_values = bit_values
        self.__use_encoding = use_encoding
        self.__chosen_template = chosen_template
        self.__iterations = iterations
        self.__num_individuals = num_individuals
        self.__crossover_probability = cr_p
        self.__mutation_probability = mt_p
        self.__elitism = elitism
        self.__rank = rank
        self.__fitness_function = fitness_function
        self.__maximize = maximize
        self.__templates_chromosomes = chromosomes
        self.__threshold_chromosomes = int(ceil(log2(params[0] * self.__templates_chromosomes))) + 2
        self.__scaling_factor = 2 ** (self.__threshold_chromosomes - 1)
        self.__m_wlcss_cuda = WLCSSCudaTemplatesTraining(self.__streams, self.__params, self.__templates_chromosomes,
                                                         self.__num_individuals, self.__use_encoding)
        self.__results = list()

    def optimize(self):
        self.__execute_ga()

    def __execute_ga(self):
        scores = list()
        max_scores = np.array([])
        best_templates = list()
        best_thresholds = list()
        templates_pop, thresholds_pop = self.__generate_population()
        bar = progressbar.ProgressBar(max_value=self.__iterations)
        fit_scores_distances = self.__compute_fitness_cuda(templates_pop, thresholds_pop)
        fit_scores = fit_scores_distances[:, 0]
        num_zero_grad = 0
        compute_grad = False
        i = 0
        while i < self.__iterations:
            pop_sort_idx = np.argsort(-fit_scores if self.__maximize else fit_scores)
            top_templates_individuals = templates_pop[pop_sort_idx]
            top_thresholds_individuals = thresholds_pop[pop_sort_idx]
            templates_selected_population, threshold_selected_population = self.__selection(top_templates_individuals,
                                                                                            top_thresholds_individuals,
                                                                                            self.__rank)
            templates_crossovered_population, threshold_crossovered_population = self.__crossover(
                templates_selected_population,
                threshold_selected_population,
                self.__crossover_probability)
            templates_pop, thresholds_pop = self.__mutation(templates_crossovered_population,
                                                            threshold_crossovered_population,
                                                            self.__mutation_probability)
            if self.__elitism > 0:
                templates_pop[0:self.__elitism] = top_templates_individuals[0:self.__elitism]
                thresholds_pop[0:self.__elitism] = top_thresholds_individuals[0:self.__elitism]
                templates_pop, thresholds_pop = self.__shuffle_pop(templates_pop, thresholds_pop)
            fit_scores_distances = self.__compute_fitness_cuda(templates_pop, thresholds_pop)
            fit_scores = fit_scores_distances[:, 0]
            good_distances = fit_scores_distances[:, 1]
            bad_distances = fit_scores_distances[:, 2]
            if self.__maximize:
                top_idx = np.argmax(fit_scores)
            else:
                top_idx = np.argmin(fit_scores)
            best_template = templates_pop[top_idx]
            best_threshold = self.__np_to_int(thresholds_pop[top_idx]) - self.__scaling_factor
            scores.append([np.mean(fit_scores), np.max(fit_scores), np.min(fit_scores), np.std(fit_scores),
                           good_distances[top_idx], bad_distances[top_idx]])
            best_templates.append(best_template)
            best_thresholds.append(best_threshold)
            max_scores = np.append(max_scores, np.max(fit_scores))
            if i > (self.__iterations / 100):
                compute_grad = True
            if compute_grad:
                if np.gradient(max_scores)[-1] < 0.05:
                    num_zero_grad += 1
            i += 1
            bar.update(i)
        bar.finish()
        if self.__maximize:
            top_idx = np.argmax(fit_scores)
        else:
            top_idx = np.argmin(fit_scores)
        top_score = fit_scores[top_idx]
        best_template = templates_pop[top_idx]
        best_threshold = self.__np_to_int(thresholds_pop[top_idx]) - self.__scaling_factor
        self.__m_wlcss_cuda.cuda_freemem()
        self.__results = [top_score, best_template, best_threshold, scores, best_templates, best_thresholds]

    def __generate_population(self):
        templates_pop = np.random.randint(0, self.__bit_values,
                                          size=(self.__num_individuals, self.__templates_chromosomes))
        if self.__chosen_template is not None:
            templates_pop[0] = self.__chosen_template
        thresholds_pop = (np.random.rand(self.__num_individuals, self.__threshold_chromosomes) < 0.5).astype(int)
        return templates_pop, thresholds_pop

    def __selection(self, top_templates_individuals, top_thresholds_individuals, rnk):
        top_templates_individuals = top_templates_individuals[0:rnk]
        reproduced_templates_individuals = np.array(
            [top_templates_individuals[i % len(top_templates_individuals)] for i in range(self.__num_individuals)])
        top_thresholds_individuals = top_thresholds_individuals[0:rnk]
        reproduced_thresholds_individuals = np.array(
            [top_thresholds_individuals[i % len(top_thresholds_individuals)] for i in range(self.__num_individuals)])
        return self.__shuffle_pop(reproduced_templates_individuals, reproduced_thresholds_individuals)

    def __crossover(self, templates_pop, thresholds_pop, cp):
        new_templates_pop = np.empty(templates_pop.shape, dtype=int)
        for i in range(0, len(templates_pop) - 1, 2):
            if np.random.random() < cp:
                crossover_position = random.randint(2, self.__templates_chromosomes - 2)
                new_templates_pop[i] = np.append(templates_pop[i][0:crossover_position],
                                                 templates_pop[i + 1][crossover_position:])
                new_templates_pop[i + 1] = np.append(templates_pop[i + 1][0:crossover_position],
                                                     templates_pop[i][crossover_position:])
            else:
                new_templates_pop[i] = templates_pop[i]
                new_templates_pop[i + 1] = templates_pop[i + 1]
        new_thresholds_pop = np.empty(thresholds_pop.shape, dtype=int)
        for i in range(0, len(thresholds_pop) - 1, 2):
            if np.random.random() < cp:
                crossover_position = random.randint(2, self.__threshold_chromosomes - 2)
                new_thresholds_pop[i] = np.append(thresholds_pop[i][0:crossover_position],
                                                  thresholds_pop[i + 1][crossover_position:])
                new_thresholds_pop[i + 1] = np.append(thresholds_pop[i + 1][0:crossover_position],
                                                      thresholds_pop[i][crossover_position:])
            else:
                new_thresholds_pop[i] = thresholds_pop[i]
                new_thresholds_pop[i + 1] = thresholds_pop[i + 1]
        return new_templates_pop, new_thresholds_pop

    def __mutation(self, templates_pop, thresholds_pop, mp):
        tmpl_sizes = templates_pop.shape
        mask = np.random.rand(templates_pop.shape[0], templates_pop.shape[1]) < mp
        new_templates_pop_mask = np.random.normal(0, 2, size=tmpl_sizes) * mask
        new_templates_pop = np.remainder(np.copy(templates_pop) + new_templates_pop_mask, self.__bit_values)

        mask = np.random.rand(thresholds_pop.shape[0], thresholds_pop.shape[1]) < mp
        new_thresholds_pop = np.mod(thresholds_pop + mask, 2)

        return new_templates_pop.astype(np.int), new_thresholds_pop

    def __compute_fitness_cuda(self, templates_pop, threshold_pop):
        matching_scores = self.__m_wlcss_cuda.compute_wlcss(templates_pop)
        fitness_scores = np.array([fit_fun.isolated_fitness_function_templates(matching_scores[0][:, k],
                                                                               self.__streams_labels,
                                                                               self.__np_to_int(threshold_pop[
                                                                                                    k]) - self.__scaling_factor,
                                                                               parameter_to_optimize=self.__fitness_function)
                                   for k in
                                   range(self.__num_individuals)])
        return fitness_scores

    def __shuffle_pop(self, templates_pop, thresholds_pop):
        random_idx = np.arange(0, self.__num_individuals)
        np.random.shuffle(random_idx)
        return templates_pop[random_idx], thresholds_pop[random_idx]

    def __np_to_int(self, chromosome):
        out = 0
        for bit in chromosome:
            out = (out << 1) | bit
        return out

    def get_results(self):
        return self.__results
