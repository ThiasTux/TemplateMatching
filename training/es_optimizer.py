import random

import numpy as np
import progressbar

from performance_evaluation import fitness_functions as fit_fun
from template_matching.wlcss_cuda_class import WLCSSCudaTraining


class ESOptimizer:
    def __init__(self, instances, instances_labels, cls, template_chromosomes, templates_bit_values, file=None,
                 chosen_template=None,
                 use_encoding=False,
                 num_processes=1,
                 iterations=500,
                 num_individuals=32,
                 bits_parameters=5, bits_thresholds=10,
                 cr_p=0.3, mt_p=0.1,
                 elitism=3, rank=10, fitness_function=7, maximize=True):
        self.__instances = instances
        self.__instances_labels = np.array(instances_labels).reshape((len(instances), 1))
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
        self.__total_genes = bits_parameters * 3 + bits_thresholds
        self.__templates_chromosomes = template_chromosomes
        self.__scaling_factor = 2 ** (self.__bits_thresholds - 1)
        self.__m_wlcss_cuda = WLCSSCudaTraining(self.__instances, self.__templates_chromosomes,
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
        best_params = list()
        params_pop, templates_pop = self.__generate_population()
        bar = progressbar.ProgressBar(max_value=self.__iterations)
        fit_scores = self.__compute_fitness_cuda(params_pop, templates_pop)
        i = 0
        while i < self.__iterations and np.max(fit_scores) < 0:
            pop_sort_idx = np.argsort(-fit_scores if self.__maximize else fit_scores)
            top_templates_individuals = templates_pop[pop_sort_idx]
            top_params_individuals = params_pop[pop_sort_idx]
            params_selected_population, templates_selected_population = self.__selection(top_params_individuals,
                                                                                         top_templates_individuals,
                                                                                         self.__rank)
            params_crossovered_population, templates_crossovered_population = self.__crossover(
                params_selected_population, templates_selected_population, self.__crossover_probability)
            params_pop, templates_pop = self.__mutation(params_crossovered_population,
                                                        templates_crossovered_population,
                                                        self.__mutation_probability)
            if self.__elitism > 0:
                templates_pop[0:self.__elitism] = top_templates_individuals[0:self.__elitism]
                params_pop[0:self.__elitism] = top_params_individuals[0:self.__elitism]
                params_pop, templates_pop = self.__shuffle_pop(params_pop, templates_pop)
            fit_scores = self.__compute_fitness_cuda(params_pop, templates_pop)
            if self.__maximize:
                top_idx = np.argmax(fit_scores)
            else:
                top_idx = np.argmin(fit_scores)
            best_template = templates_pop[top_idx]
            best_param = [self.__np_to_int(params_pop[top_idx][0:self.__bits_parameters]),
                          self.__np_to_int(params_pop[top_idx][self.__bits_parameters:self.__bits_parameters * 2]),
                          self.__np_to_int(params_pop[top_idx][self.__bits_parameters * 2:self.__bits_parameters * 3]),
                          self.__np_to_int(params_pop[top_idx][self.__bits_parameters * 3:]) - self.__scaling_factor]
            scores.append([np.mean(fit_scores), np.max(fit_scores), np.min(fit_scores), np.std(fit_scores)])
            best_templates.append(best_template)
            best_params.append(best_param)
            i += 1
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
                      self.__np_to_int(params_pop[top_idx][self.__bits_parameters * 2:self.__bits_parameters * 3]),
                      self.__np_to_int(params_pop[top_idx][self.__bits_parameters * 3:]) - self.__scaling_factor]
        if self.__write_to_file:
            output_scores_path = "{}_{:02d}_{}_scores.txt".format(self.__output_file, num_test, self.__class)
            with open(output_scores_path, 'w') as f:
                for item in scores:
                    f.write("%s\n" % str(item).replace("[", "").replace("]", ""))
            output_templates_path = "{}_{:02d}_{}_all.txt".format(self.__output_file, num_test, self.__class)
            with open(output_templates_path, 'w') as f:
                for i, item in enumerate(best_templates):
                    f.write("{} {}\n".format(" ".join([str(x) for x in item.tolist()]),
                                             best_params[i]))
        self.__m_wlcss_cuda.cuda_freemem()
        return [top_score, best_template, best_param]

    def __generate_population(self):
        params_pop = (np.random.rand(self.__num_individuals, self.__total_genes) < 0.5).astype(int)
        templates_pop = np.random.randint(0, self.__templates_bit_values,
                                          size=(self.__num_individuals, self.__templates_chromosomes))
        if self.__chosen_template is not None:
            templates_pop[0] = self.__chosen_template[:, 1]
        return params_pop, templates_pop

    def __selection(self, top_params_individuals, top_templates_individuals, rnk):
        top_templates_individuals = top_templates_individuals[0:rnk]
        reproduced_templates_individuals = np.array(
            [top_templates_individuals[i % len(top_templates_individuals)] for i in range(self.__num_individuals)])
        top_params_individuals = top_params_individuals[0:rnk]
        reproduced_params_individuals = np.array(
            [top_params_individuals[i % len(top_params_individuals)] for i in range(self.__num_individuals)])
        return self.__shuffle_pop(reproduced_params_individuals, reproduced_templates_individuals)

    def __crossover(self, params_pop, templates_pop, cp):
        new_templates_pop = np.empty(templates_pop.shape, dtype=int)
        for i in range(0, self.__num_individuals - 1, 2):
            if np.random.random() < cp:
                crossover_position = random.randint(2, self.__templates_chromosomes - 2)
                new_templates_pop[i] = np.append(templates_pop[i][0:crossover_position],
                                                 templates_pop[i + 1][crossover_position:])
                new_templates_pop[i + 1] = np.append(templates_pop[i + 1][0:crossover_position],
                                                     templates_pop[i][crossover_position:])
            else:
                new_templates_pop[i] = templates_pop[i]
                new_templates_pop[i + 1] = templates_pop[i + 1]
        new_params_pop = np.empty(params_pop.shape, dtype=int)
        for i in range(0, self.__num_individuals - 1, 2):
            if np.random.random() < cp:
                crossover_position = random.randint(2, self.__total_genes - 2)
                new_params_pop[i] = np.append(params_pop[i][0:crossover_position],
                                              params_pop[i + 1][crossover_position:])
                new_params_pop[i + 1] = np.append(params_pop[i + 1][0:crossover_position],
                                                  params_pop[i][crossover_position:])
            else:
                new_params_pop[i] = params_pop[i]
                new_params_pop[i + 1] = params_pop[i + 1]
        return new_params_pop, new_templates_pop

    def __mutation(self, params_pop, templates_pop, mp):
        tmpl_sizes = templates_pop.shape
        mask = np.random.rand(templates_pop.shape[0], templates_pop.shape[1]) < mp
        new_templates_pop_mask = np.random.normal(0, 8, size=tmpl_sizes) * mask
        new_templates_pop = np.remainder(np.copy(templates_pop) + new_templates_pop_mask, self.__templates_bit_values)

        mask = np.random.rand(params_pop.shape[0], params_pop.shape[1]) < mp
        new_params_pop = np.mod(params_pop + mask, 2)

        return new_params_pop, new_templates_pop.astype(np.int)

    def __compute_fitness_cuda(self, params_pop, templates_pop):
        params = [
            [self.__np_to_int(p[0:self.__bits_parameters]),
             self.__np_to_int(p[self.__bits_parameters:self.__bits_parameters * 2]),
             self.__np_to_int(p[self.__bits_parameters * 2:self.__bits_parameters * 3])] for p in params_pop]
        thresholds = [self.__np_to_int(p[self.__bits_parameters * 3:]) for p in params_pop]
        matching_scores = self.__m_wlcss_cuda.compute_wlcss(params, templates_pop)
        matching_scores = [np.concatenate((ms, self.__instances_labels), axis=1) for ms in matching_scores]
        fitness_scores = np.array([fit_fun.isolated_fitness_function_all(matching_scores[0][:, k],
                                                                         matching_scores[0][:, -1],
                                                                         thresholds[k] - self.__scaling_factor,
                                                                         parameter_to_optimize=self.__fitness_function)
                                   for k in
                                   range(self.__num_individuals)])
        return fitness_scores

    def __shuffle_pop(self, params_pop, templates_pop):
        random_idx = np.arange(0, self.__num_individuals)
        np.random.shuffle(random_idx)
        return params_pop[random_idx], templates_pop[random_idx]

    def __np_to_int(self, chromosome):
        out = 0
        for bit in chromosome:
            out = (out << 1) | bit
        return out

    def get_results(self):
        return self.__results
