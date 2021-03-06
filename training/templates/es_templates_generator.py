import random
from math import ceil, log2

import numpy as np
import progressbar

from performance_evaluation import fitness_functions as fit_fun
from template_matching.wlcss_cuda_class import WLCSSCudaTemplatesTraining, WLCSSCudaVariableTemplatesTraining


class ESTemplateGenerator:
    def __init__(self, steams, stream_labels, params, threshold, cls, chromosomes, bit_values,
                 chosen_template=None,
                 use_encoding=False,
                 num_processes=1,
                 iterations=500,
                 num_individuals=32, save_internals=False, cr_p=0.3, mt_p=0.1,
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
        self.__save_internal = save_internals
        self.__intermediate_population = list()
        self.__intermediate_fitness_scores = list()
        self.__crossover_probability = cr_p
        self.__mutation_probability = mt_p
        self.__elitism = elitism
        self.__rank = rank
        self.__fitness_function = fitness_function
        self.__maximize = maximize
        self.__templates_chromosomes = chromosomes
        self.__max_scores = params[0] * self.__templates_chromosomes
        self.__min_scores = -self.__max_scores
        self.__max_fitness = self.__max_scores - self.__min_scores
        self.__min_fitness = self.__min_scores - self.__max_scores
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
        fit_scores = self.__normalize_fitness_scores(fit_scores)
        if self.__save_internal:
            self.__intermediate_population.append(templates_pop)
            self.__intermediate_fitness_scores.append(fit_scores)
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
            fit_scores = self.__normalize_fitness_scores(fit_scores)
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
            if self.__save_internal:
                self.__intermediate_population.append(templates_pop)
                self.__intermediate_fitness_scores.append(fit_scores)
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
        new_templates_pop_mask = np.random.normal(0, 4, size=tmpl_sizes) * mask
        new_templates_pop = np.remainder(np.copy(templates_pop) + new_templates_pop_mask, self.__bit_values)

        return new_templates_pop.astype(np.int)

    def __compute_fitness_cuda(self, templates_pop):
        matching_scores = self.__m_wlcss_cuda.compute_wlcss(templates_pop)
        fitness_scores = [fit_fun.isolated_fitness_function_templates(matching_scores[0][:, k],
                                                                      self.__streams_labels,
                                                                      self.__threshold,
                                                                      parameter_to_optimize=self.__fitness_function)
                          for k in
                          range(self.__num_individuals)]
        return np.array(fitness_scores)

    def __normalize_fitness_scores(self, fit_scores):
        min_fitness = np.min(fit_scores)
        if min_fitness < self.__min_fitness:
            self.__min_fitness = min_fitness
        # Min-max normalization fit_score
        norm_fit_scores = (fit_scores - self.__min_fitness) / (self.__max_fitness - self.__min_fitness)
        return norm_fit_scores

    def __np_to_int(self, chromosome):
        return int("".join(chromosome.astype('U')), 2)

    def get_results(self):
        return self.__results

    def get_internal_states(self):
        return np.array(self.__intermediate_fitness_scores), self.__intermediate_population


class ESVariableTemplateGenerator:
    def __init__(self, steams, stream_labels, params, threshold, cls, chromosomes, bit_values,
                 chosen_template=None,
                 use_encoding=False,
                 num_processes=1,
                 iterations=500,
                 num_individuals=32, save_internals=False,
                 cr_p=0.3, mt_p=0.1, en_p=0.33, sh_p=0.33, l_weight=0.5,
                 elitism=3, rank=10, fitness_function=7, max_lr=1, min_lr=0.25, maximize=True):
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
        self.__save_internal = save_internals
        self.__intermediate_population = list()
        self.__intermediate_fitness_scores = list()
        self.__crossover_probability = cr_p
        self.__mutation_probability = mt_p
        self.__elitism = elitism
        self.__rank = rank
        self.__fitness_function = fitness_function
        self.__maximize = maximize
        self.__templates_chromosomes = chromosomes
        self.__max_templates_chromosomes = int(chromosomes * max_lr)
        self.__min_templates_chromosomes = max(int(chromosomes * min_lr), 3)
        self.__max_scores = params[0] * self.__max_templates_chromosomes
        self.__b = 5 / self.__max_scores
        self.__min_scores = -self.__max_scores
        self.__max_fitness = self.__max_scores - self.__min_scores
        self.__min_fitness = self.__min_scores - self.__max_scores
        print("Max score: {}, Min score: {}, Max distance: {}, Min distance: {}".format(self.__max_scores,
                                                                                        self.__min_scores,
                                                                                        self.__max_fitness,
                                                                                        self.__min_fitness))
        self.__max_templates_length = self.__max_templates_chromosomes + 1
        self.__min_templates_length = self.__min_templates_chromosomes - 1
        self.__enlarge_probability = en_p
        self.__shrink_probability = sh_p
        self.__length_weight = l_weight
        self.__fitness_weight = 1 - l_weight
        self.__m_wlcss_cuda = WLCSSCudaVariableTemplatesTraining(self.__streams, self.__params,
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
        fit_scores = self.__combine_fitness(fit_scores, templates_pop, method='sigmoid_min_max')
        if self.__save_internal:
            self.__intermediate_population.append(templates_pop)
            self.__intermediate_fitness_scores.append(fit_scores)
        for i in range(self.__iterations):
            pop_sort_idx = np.argsort(-fit_scores if self.__maximize else fit_scores)
            top_templates_individuals = [templates_pop[k] for k in pop_sort_idx]
            templates_selected_population = self.__selection(top_templates_individuals, self.__rank)
            templates_crossovered_population = self.__crossover(templates_selected_population,
                                                                self.__crossover_probability)
            mutated_templates_pop = self.__mutation(templates_crossovered_population, self.__mutation_probability)
            templates_pop = self.__enlarge_shrink(mutated_templates_pop, self.__enlarge_probability,
                                                  self.__shrink_probability)
            if self.__elitism > 0:
                templates_pop[0:self.__elitism] = top_templates_individuals[0:self.__elitism]
            fit_scores_distances = self.__compute_fitness_cuda(templates_pop)
            fit_scores = fit_scores_distances[:, 0]
            fit_scores = self.__combine_fitness(fit_scores, templates_pop, method='sigmoid_min_max')
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
            if self.__save_internal:
                self.__intermediate_population.append(templates_pop)
                self.__intermediate_fitness_scores.append(fit_scores)
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
        reproduced_templates_individuals = [top_templates_individuals[i % len(top_templates_individuals)] for i in
                                            range(self.__num_individuals)]
        random.shuffle(reproduced_templates_individuals)
        return reproduced_templates_individuals

    def __crossover(self, templates_pop, cp):
        new_templates_pop = [[] for _ in range(len(templates_pop))]
        for i in range(0, len(templates_pop) - 1, 2):
            if np.random.random() < cp:
                random_perc_position = random.random()
                crossover_position_1 = int(len(templates_pop[i]) * random_perc_position)
                crossover_position_2 = int(len(templates_pop[i + 1]) * random_perc_position)
                new_templates_pop[i] = [templates_pop[i][j] for j in range(0, crossover_position_1)] + [
                    templates_pop[i + 1][j] for j in range(crossover_position_2, len(templates_pop[i + 1]))]
                new_templates_pop[i + 1] = [templates_pop[i][j] for j in
                                            range(crossover_position_1, len(templates_pop[i]))] + [
                                               templates_pop[i + 1][j] for j in range(0, crossover_position_2)]
            else:
                new_templates_pop[i] = templates_pop[i]
                new_templates_pop[i + 1] = templates_pop[i + 1]
        return [np.array(t) for t in new_templates_pop]

    def __mutation(self, templates_pop, mp):
        new_templates_pop = [[] for _ in templates_pop]
        for i, t in enumerate(templates_pop):
            mask = np.random.rand(t.shape[0]) < mp
            new_template_pop_mask = np.random.normal(0, 4, size=t.shape) * mask
            new_templates_pop[i] = np.remainder(t + new_template_pop_mask, self.__bit_values).astype(int)
        return new_templates_pop

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

    def __compute_fitness_cuda(self, templates_pop):
        matching_scores = self.__m_wlcss_cuda.compute_wlcss(templates_pop)
        fitness_scores = [fit_fun.isolated_fitness_function_templates(matching_scores[:, k],
                                                                      self.__streams_labels,
                                                                      self.__threshold,
                                                                      parameter_to_optimize=self.__fitness_function)
                          for k in
                          range(self.__num_individuals)]
        return np.array(fitness_scores)

    def __combine_fitness(self, fit_scores, templates_pop, method='min_max'):
        templates_lengths = np.array([len(t) for t in templates_pop])
        if method == 'min_max':
            # print("Mean: {}, Std: {}, Min: {}, Max: {}".format(np.mean(fit_scores), np.std(fit_scores),
            #                                                    np.min(fit_scores), np.max(fit_scores)), end="\n",
            #       flush=True)
            # Min-max normalization fit_score
            norm_fit_scores = (fit_scores - self.__min_fitness) / (self.__max_fitness - self.__min_fitness)
            # Min-max normalization template length
            norm_templates_lengths = (templates_lengths - self.__max_templates_length) / (
                    self.__min_templates_length - self.__max_templates_length)
        elif method == 'z_score_local':
            # Min-max normalization fit_score
            print("Mean: {}, Std: {}, Min: {}, Max: {}".format(np.mean(fit_scores), np.std(fit_scores),
                                                               np.min(fit_scores), np.max(fit_scores)), end="\n",
                  flush=True)
            norm_fit_scores = (fit_scores - np.mean(fit_scores)) / np.std(fit_scores)
            norm_templates_lengths = (templates_lengths - np.mean(templates_lengths)) / np.std(templates_lengths)
        elif method == 'moving_min_max':
            min_fitness = np.min(fit_scores)
            if min_fitness < self.__min_fitness:
                self.__min_fitness = min_fitness
            # Min-max normalization fit_score
            norm_fit_scores = (fit_scores - self.__min_fitness) / (self.__max_fitness - self.__min_fitness)
            # Min-max normalization template length
            norm_templates_lengths = (templates_lengths - self.__max_templates_length) / (
                    self.__min_templates_length - self.__max_templates_length)
        elif method == 'sigmoid_min_max':
            norm_fit_scores = 1 / (1 + np.e ** (-self.__b * fit_scores))
            norm_templates_lengths = (templates_lengths - self.__max_templates_length) / (
                    self.__min_templates_length - self.__max_templates_length)
        return self.__length_weight * norm_templates_lengths + self.__fitness_weight * norm_fit_scores

    def __np_to_int(self, chromosome):
        return int("".join(chromosome.astype('U')), 2)

    def get_results(self):
        return self.__results

    def get_internal_states(self):
        return np.array(self.__intermediate_fitness_scores), self.__intermediate_population


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
