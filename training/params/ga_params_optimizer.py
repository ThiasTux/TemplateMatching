import multiprocessing as mp
import random

import numpy as np
import progressbar

from performance_evaluation import fitness_functions as fit_fun
from template_matching.wlcss_cuda import WLCSSCuda


class GAParamsOptimizer:
    def __init__(self, templates, instances, templates_labels, use_encoding=False, num_processes=1, iterations=500,
                 num_individuals=32,
                 bits_parameter=5, bits_threshold=10, cr_p=0.3, mt_p=0.1, elitism=3, rank=10, maximize=True):
        self.templates = templates
        self.instances = instances
        self.templates_labels = templates_labels
        self.use_encoding = use_encoding
        self.num_processes = num_processes
        self.iterations = iterations
        self.num_individuals = num_individuals
        self.bits_parameter = bits_parameter
        self.bits_threshold = bits_threshold
        self.crossover_probability = cr_p
        self.mutation_probability = mt_p
        self.elitism = elitism
        self.rank = rank
        self.maximize = maximize
        self.chromosomes = 3
        self.total_genes = self.bits_parameter * self.chromosomes
        self.chromosomes += len(templates)
        self.total_genes += bits_threshold * len(templates)
        self.m_wlcss_cuda = WLCSSCuda(self.templates, self.instances, self.num_individuals, self.use_encoding)
        self.results = list()

    def optimize(self):
        results_queue = mp.Queue()
        processes = [mp.Process(target=self.execute_ga, args=([results_queue])) for t in range(self.num_processes)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        self.results = [results_queue.get() for p in processes]

    def execute_ga(self, results_queue):
        scores = list()
        pop = self.generate_population()
        bar = progressbar.ProgressBar(max_value=self.iterations)
        fit_scores = self.compute_fitness_cuda(pop)
        for i in range(self.iterations):
            pop_sort_idx = np.argsort(-fit_scores if self.maximize else fit_scores)
            top_individuals = pop[pop_sort_idx]
            selected_population = self.selection(top_individuals, self.rank)
            crossovered_population = self.crossover(selected_population, self.crossover_probability)
            pop = self.mutation(crossovered_population, self.mutation_probability)
            if self.elitism > 0:
                pop[0:self.elitism] = top_individuals[0:self.elitism]
            fit_scores = self.compute_fitness_cuda(pop)
            top_idx = np.argmax(fit_scores)
            penalty = self.np_to_int(pop[top_idx][0:self.bits_parameter])
            reward = self.np_to_int(pop[top_idx][self.bits_parameter:self.bits_parameter * 2])
            accepted_distance = self.np_to_int(pop[top_idx][self.bits_parameter * 2:self.bits_parameter * 3])
            thresholds = [self.np_to_int(
                pop[top_idx][self.bits_parameter * 3 + (j * self.bits_threshold):self.bits_parameter * 3 + (
                        j + 1) * self.bits_threshold])
                for j
                in range(len(self.templates))]
            scores.append([np.mean(fit_scores), np.max(fit_scores), np.min(fit_scores), np.std(fit_scores),
                           [penalty, reward, accepted_distance, thresholds]])
            bar.update(i)
        bar.finish()
        if self.maximize:
            top_idx = np.argmax(fit_scores)
        else:
            top_idx = np.argmin(fit_scores)
        top_score = fit_scores[top_idx]
        penalty = self.np_to_int(pop[top_idx][0:self.bits_parameter])
        reward = self.np_to_int(pop[top_idx][self.bits_parameter:self.bits_parameter * 2])
        accepted_distance = self.np_to_int(pop[top_idx][self.bits_parameter * 2:self.bits_parameter * 3])
        thresholds = [self.np_to_int(pop[top_idx][
                                     self.bits_parameter * 3 + (j * self.bits_threshold):self.bits_parameter * 3 + (
                                             j + 1) * self.bits_threshold]) for j in range(len(self.templates))]
        results_queue.put([penalty, reward, accepted_distance, thresholds, top_score, scores])

    def generate_population(self):
        return (np.random.rand(self.num_individuals, self.total_genes) < 0.5).astype(int)

    def selection(self, top_individuals, rnk):
        top_individuals = top_individuals[0:rnk]
        reproduced_individuals = np.array(
            [top_individuals[i % len(top_individuals)] for i in range(self.num_individuals)])
        np.random.shuffle(reproduced_individuals)
        return reproduced_individuals

    def crossover(self, pop, cp):
        new_pop = np.empty([len(pop), len(pop[0])], dtype=int)
        for i in range(0, len(pop) - 1, 2):
            if np.random.random() < cp:
                chromosomes_len = self.total_genes
                crossover_position = random.randint(0, chromosomes_len - 2)
                new_pop[i] = np.append(pop[i][0:crossover_position], pop[i + 1][crossover_position:])
                new_pop[i + 1] = np.append(pop[i + 1][0:crossover_position], pop[i][crossover_position:])
            else:
                new_pop[i] = pop[i]
                new_pop[i + 1] = pop[i]
        return new_pop

    def mutation(self, pop, mp):
        mask = np.random.rand(pop.shape[0], pop.shape[1]) < mp
        new_pop = np.mod(pop + mask, 2)
        return new_pop

    def compute_fitness_cuda(self, pop):
        params = [
            [self.np_to_int(p[self.bits_parameter:self.bits_parameter * 2]), self.np_to_int(p[0:self.bits_parameter]),
             self.np_to_int(p[self.bits_parameter * 2:self.bits_parameter * 3])] for p in pop]
        thresholds = [[self.np_to_int(p[self.bits_parameter * 3 + (j * self.bits_threshold):self.bits_parameter * 3 + (
                j + 1) * self.bits_threshold]) for j in range(len(self.templates))] for p in pop]
        matching_scores = self.m_wlcss_cuda.compute_cuda(params)
        fitness_scores = np.array([fit_fun.isolated_fitness_function_params(matching_scores[k], thresholds[k],
                                                                            self.templates_labels,
                                                                            parameter_to_optimize=5) for k in
                                   range(len(pop))])
        return fitness_scores

    def np_to_int(self, chromosome):
        return int("".join(chromosome.astype('U')), 2)
