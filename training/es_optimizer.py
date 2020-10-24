import random

import numpy as np
import progressbar

from performance_evaluation import fitness_functions as fit_fun
from template_matching.wlcss_cuda_class import WLCSSCudaTraining


class ESOptimizer:
    def __init__(self, streams, streams_labels, cls, templates_chromosomes, templates_bit_values,
                 chosen_template=None,
                 use_encoding=False,
                 num_processes=1,
                 iterations=500,
                 num_individuals=32,
                 bits_parameters=5, bits_thresholds=10,
                 cr_p=0.3, mt_p=0.1, en_p=0.33, sh_p=0.33, l_weight=0.5,
                 elitism=3, rank=10, fitness_function=7, max_lr=1, min_lr=0.25, maximize=True):
        self.__streams = streams
        self.__streams_labels = streams_labels
        self.__class = cls
        self.__templates_bit_values = templates_bit_values
        self.__use_encoding = use_encoding
        self.__chosen_template = chosen_template
        self.__num_processes = num_processes
        self.__iterations = iterations
        self.__num_individuals = num_individuals
        self.__bits_parameters = bits_parameters
        self.__bits_thresholds = bits_thresholds
        self.__crossover_probability = cr_p
        self.__mutation_probability = mt_p
        self.__elitism = elitism
        self.__rank = rank
        self.__fitness_function = fitness_function
        self.__maximize = maximize
        self.__total_genes = bits_parameters * 3
        self.__templates_chromosomes = templates_chromosomes
        self.__max_templates_chromosomes = int(templates_chromosomes * max_lr)
        self.__min_templates_chromosomes = int(templates_chromosomes * min_lr)
        self.__enlarge_probability = en_p
        self.__shrink_probability = sh_p
        self.__length_weight = l_weight
        self.__fitness_weight = 1 - l_weight
        self.__m_wlcss_cuda = WLCSSCudaTraining(self.__streams, self.__num_individuals, self.__use_encoding)
        self.__results = list()

    def optimize(self):
        self.__execute_ga()

    def __execute_ga(self):
        scores = list()
        best_templates = list()
        best_params = list()
        params_pop, templates_pop = self.__generate_population()
        bar = progressbar.ProgressBar(max_value=self.__iterations)
        fit_scores_distances = self.__compute_fitness_cuda(params_pop, templates_pop)
        fit_scores = fit_scores_distances[:, 0]
        fit_scores = np.array(
            [(self.__length_weight * 1 / len(templates_pop[k])) + (self.__fitness_weight * fit_scores[k]) for k in
             range(self.__num_individuals)])
        for i in range(self.__iterations):
            pop_sort_idx = np.argsort(-fit_scores if self.__maximize else fit_scores)
            top_templates_individuals = [templates_pop[k] for k in pop_sort_idx]
            top_params_individuals = params_pop[pop_sort_idx]
            params_selected_population, templates_selected_population = self.__selection(top_params_individuals,
                                                                                         top_templates_individuals,
                                                                                         self.__rank)
            params_crossovered_population, templates_crossovered_population = self.__crossover(
                params_selected_population, templates_selected_population, self.__crossover_probability)
            params_pop, templates_mutated_population = self.__mutation(params_crossovered_population,
                                                                       templates_crossovered_population,
                                                                       self.__mutation_probability)
            templates_pop = self.__enlarge_shrink(templates_mutated_population, self.__enlarge_probability,
                                                  self.__shrink_probability)
            if self.__elitism > 0:
                templates_pop[0:self.__elitism] = top_templates_individuals[0:self.__elitism]
                params_pop[0:self.__elitism] = top_params_individuals[0:self.__elitism]
                params_pop, templates_pop = self.__shuffle_pop(params_pop, templates_pop)
            fit_scores_distances = self.__compute_fitness_cuda(params_pop, templates_pop)
            fit_scores = fit_scores_distances[:, 0]
            fit_scores = np.array(
                [(self.__length_weight * 1 / len(templates_pop[k])) + (self.__fitness_weight * fit_scores[k]) for k in
                 range(self.__num_individuals)])
            good_distances = fit_scores_distances[:, 1]
            bad_distances = fit_scores_distances[:, 2]
            if self.__maximize:
                top_idx = np.argmax(fit_scores)
            else:
                top_idx = np.argmin(fit_scores)
            best_template = templates_pop[top_idx]
            best_param = [self.__np_to_int(params_pop[top_idx][0:self.__bits_parameters]),
                          self.__np_to_int(params_pop[top_idx][self.__bits_parameters:self.__bits_parameters * 2]),
                          self.__np_to_int(params_pop[top_idx][self.__bits_parameters * 2:self.__bits_parameters * 3])]
            scores.append([np.mean(fit_scores), np.max(fit_scores), np.min(fit_scores), np.std(fit_scores),
                           good_distances[top_idx], bad_distances[top_idx]])
            best_templates.append(best_template)
            best_params.append(best_param)
            bar.update(i)
        bar.finish()
        if self.__maximize:
            top_idx = np.argmax(fit_scores)
        else:
            top_idx = np.argmin(fit_scores)
        top_score = fit_scores[top_idx]
        best_template = templates_pop[top_idx]
        best_param = [self.__np_to_int(params_pop[top_idx][0:self.__bits_parameters]),
                      self.__np_to_int(params_pop[top_idx][self.__bits_parameters:self.__bits_parameters * 2]),
                      self.__np_to_int(params_pop[top_idx][self.__bits_parameters * 2:self.__bits_parameters * 3])]
        self.__m_wlcss_cuda.cuda_freemem()
        self.__results = [top_score, best_param, best_template, scores, best_params, best_templates]

    def __generate_population(self):
        params_pop = (np.random.rand(self.__num_individuals, self.__total_genes) < 0.5).astype(int)
        templates_pop = np.random.randint(0, self.__templates_bit_values,
                                          size=(self.__num_individuals, self.__templates_chromosomes))
        if self.__chosen_template is not None:
            templates_pop[0] = self.__chosen_template
        return params_pop, templates_pop

    def __selection(self, top_params_individuals, top_templates_individuals, rnk):
        top_templates_individuals = top_templates_individuals[0:rnk]
        reproduced_templates_individuals = [top_templates_individuals[i % len(top_templates_individuals)] for i in
                                            range(self.__num_individuals)]
        top_params_individuals = top_params_individuals[0:rnk]
        reproduced_params_individuals = np.array(
            [top_params_individuals[i % len(top_params_individuals)] for i in range(self.__num_individuals)])
        return self.__shuffle_pop(reproduced_params_individuals, reproduced_templates_individuals)

    def __crossover(self, params_pop, templates_pop, cp):
        new_templates_pop = [[] for _ in range(len(templates_pop))]
        new_params_pop = np.empty(params_pop.shape, dtype=int)
        for i in range(0, len(templates_pop) - 1, 2):
            if np.random.random() < cp:
                crossover_position = random.randint(2, self.__total_genes - 2)
                new_params_pop[i] = np.append(params_pop[i][0:crossover_position],
                                              params_pop[i + 1][crossover_position:])
                new_params_pop[i + 1] = np.append(params_pop[i + 1][0:crossover_position],
                                                  params_pop[i][crossover_position:])

                random_perc_position = random.random()
                crossover_position_1 = int(len(templates_pop[i]) * random_perc_position)
                crossover_position_2 = int(len(templates_pop[i + 1]) * random_perc_position)
                new_templates_pop[i] = [templates_pop[i][j] for j in range(0, crossover_position_1)] + [
                    templates_pop[i + 1][j] for j in range(crossover_position_2, len(templates_pop[i + 1]))]
                new_templates_pop[i + 1] = [templates_pop[i][j] for j in
                                            range(crossover_position_1, len(templates_pop[i]))] + [
                                               templates_pop[i + 1][j] for j in range(0, crossover_position_2)]
            else:
                new_params_pop[i] = params_pop[i]
                new_params_pop[i + 1] = params_pop[i + 1]
                new_templates_pop[i] = templates_pop[i]
                new_templates_pop[i + 1] = templates_pop[i + 1]
        return new_params_pop, [np.array(t) for t in new_templates_pop]

    def __mutation(self, params_pop, templates_pop, mp):
        new_templates_pop = [[] for _ in templates_pop]
        for i, t in enumerate(templates_pop):
            mask = np.random.rand(t.shape[0]) < mp
            new_template_pop_mask = np.random.normal(0, 4, size=t.shape) * mask
            new_templates_pop[i] = np.remainder(t + new_template_pop_mask, self.__templates_bit_values).astype(int)
        mask = np.random.rand(params_pop.shape[0], params_pop.shape[1]) < mp
        new_params_pop = np.mod(params_pop + mask, 2)
        return new_params_pop, new_templates_pop

    def __enlarge_shrink(self, templates_pop, en_p, sh_p):
        new_templates_pop = [None for _ in templates_pop]
        for i, t in enumerate(templates_pop):
            choice = np.random.random()
            # Enlarge template
            if choice < en_p and len(templates_pop[i]) < self.__max_templates_chromosomes:
                en_pos = random.randint(1, len(templates_pop[i]) - 1)
                new_templates_pop[i] = np.insert(templates_pop[i], en_pos,
                                                 int(np.mean(templates_pop[i][en_pos - 1:en_pos + 1])))
            # Shrink template
            elif en_p <= choice < (en_p + sh_p) and len(templates_pop[i]) > self.__min_templates_chromosomes:
                sh_pos = random.randint(1, len(templates_pop[i]) - 1)
                new_templates_pop[i] = np.delete(templates_pop[i], sh_pos)
            # Keep the same
            else:
                new_templates_pop[i] = templates_pop[i]
        return new_templates_pop

    def __compute_fitness_cuda(self, params_pop, templates_pop):
        params = [
            [self.__np_to_int(p[0:self.__bits_parameters]),
             self.__np_to_int(p[self.__bits_parameters:self.__bits_parameters * 2]),
             self.__np_to_int(p[self.__bits_parameters * 2:self.__bits_parameters * 3])] for p in params_pop]
        matching_scores = self.__m_wlcss_cuda.compute_wlcss(params, templates_pop)
        fitness_scores = [fit_fun.isolated_fitness_function_templates(matching_scores[:, k],
                                                                      self.__streams_labels,
                                                                      None,
                                                                      parameter_to_optimize=self.__fitness_function)
                          for k in
                          range(self.__num_individuals)]
        return np.array(fitness_scores)

    def __shuffle_pop(self, params_pop, templates_pop):
        random_idx = np.arange(0, self.__num_individuals)
        np.random.shuffle(random_idx)
        return params_pop[random_idx], [templates_pop[k] for k in random_idx]

    def __np_to_int(self, chromosome):
        out = 0
        for bit in chromosome:
            out = (out << 1) | bit
        return out

    def get_results(self):
        return self.__results
