import numpy as np
import math


class ElephantHerdingOptimization(object):

    def __init__(self, pop_size, clan_size, dim_size, max_generation, alpha, beta):
        self.pop_size = pop_size
        self.clan_size = clan_size
        self.dim_size = dim_size
        self.alpha = alpha
        self.beta = beta
        self.max_generation = max_generation

    def initialize_population(self):

        init_population = np.random.uniform(-10, 10, (self.pop_size, self.dim_size))

        return init_population

    # fitness of a clan
    def get_fitness(self, population):

        fitness_value = np.zeros(self.pop_size)
        for i in range(self.pop_size):

            fitness = 0
            # ackley function #
            # first_sum = 0
            # second_sum = 0
            # for j in range(self.dim_size):
            #     first_sum += population[i, j] ** 2
            #     second_sum += math.cos(2 * math.pi * population[i, j])
            #
            # fitness = -20.0 * math.exp(-0.2 * math.sqrt(first_sum/self.dim_size)) - math.exp(second_sum/self.dim_size) \
            #           + 20 + math.e

            # alpine function
            # for j in range(self.dim_size):
            #     a = population[i, j]
            #     fitness += math.fabs(a*math.sin(a) + 0.1*a)

            # Dixon & Price function
            # a = population[i]
            # fitness += (a[0] - 1)**2
            # for j in range(1, self.dim_size):
            #     b = a[j]
            #     fitness += j*((2*(b**2) - a[j-1])**2)

            # Grieward function
            part1 = 0
            part2 = 1
            for j in range(self.dim_size):
                a = population[i, j]
                part1 += a**2
                part2 *= math.cos(float(a)/math.sqrt(j+1))

            fitness += 1 + (1/4000)*part1 - part2
            fitness_value[i] += fitness

        return fitness_value

    def get_clan_fitness(self, clan):

        fitness_value = np.zeros(self.clan_size)
        for i in range(self.clan_size):

            fitness = 0

            # ackley function
            # first_sum = 0
            # second_sum = 0
            # for j in range(self.dim_size):
            #     first_sum += clan[i, j]**2
            #     second_sum += math.cos(2*math.pi*clan[i, j])
            #
            # fitness = -20.0*math.exp(-0.2*math.sqrt(first_sum/self.dim_size)) - math.exp(second_sum/self.dim_size) \
            #          + 20 + math.e

            # alpine function
            # for j in range(self.dim_size):
            #     a = clan[i, j]
            #     fitness += math.fabs(a * math.sin(a) + 0.1 * a)

            # Grieward function
            part1 = 0
            part2 = 1
            for j in range(self.dim_size):
                a = clan[i, j]
                part1 += a ** 2
                part2 *= math.cos(a / math.sqrt(j+1))

            fitness += 1 + (1 / 4000) * part1 - part2

            fitness_value[i] += fitness

        return fitness_value

    def sort_population_by_fitness(self, population):

        fitness = self.get_fitness(population)
        sort_indices = fitness.argsort()

        return population[sort_indices[::-1]]

    def sort_clan_by_fitness(self, clan):

        fitness = self.get_clan_fitness(clan)
        sort_indices = fitness.argsort()

        return clan[sort_indices[::-1]]

    # divide population into clans => list clan collection
    def get_clans(self, population):
        clan_collection = []
        num_clans = math.ceil(self.pop_size / self.clan_size)

        for i in range(num_clans):
            a = int(i * self.clan_size)
            b = int(i * self.clan_size + self.clan_size)

            clan_i = population[a:b, :]
            clan_collection.append(clan_i)

        return clan_collection

    def update_position(self, clan_collection):

        updated_clan_collection = []

        for clan in clan_collection:
            r1 = np.random.uniform(0, 1, self.dim_size)
            # print(self.get_fitness(sorted_clan))

            matriarch = clan[-1]

            ###### Updating Operation #######
            new_clan = clan + self.alpha*(matriarch - clan) * r1

            # update matriarch
            clan_center = np.mean(clan, axis=0)
            new_child_1 = self.beta*clan_center
            new_clan[-1] = new_child_1

            new_clan = self.sort_clan_by_fitness(new_clan)

            ##### Separating Operation #######
            max_elephant = 10 * np.ones(self.dim_size)
            min_elephant = -10 * np.ones(self.dim_size)
            rand = np.random.uniform(0, 1, self.dim_size)
            new_child = min_elephant + (max_elephant - min_elephant + 1) * rand
            # new_child += (new_clan[-1] - new_clan[0])*self.alpha
            # new_child = new_clan[-1] * rand
            new_clan[0] = new_child

            updated_clan_collection.append(new_clan)

        return updated_clan_collection

    def combine_clans(self, clan_collection):

        return np.vstack(clan_collection)

    def evaluate_new_population(self, population):

        for i in range(self.pop_size):
            for j in range(self.dim_size):
                if population[i, j] > 10:
                    population[i, j] = 10
                if population[i, j] < -10:
                    population[i, j] = -10

        return self.sort_population_by_fitness(population)

    def get_result(self, clan_collection):

        best_fitness = []
        best_elephants = []

        for clan in clan_collection:
            fitness = self.get_fitness(clan)
            best_fitness.append(fitness[-1])
            best_elephants.append(clan[-1])

        min_index = best_fitness.index(min(best_fitness))

        best_fitness_value = best_fitness[min_index]
        best_elephant = best_elephants[min_index]

        return best_fitness_value, best_elephant

    def run(self):

        population = self.initialize_population()
        result = []

        for x in range(self.max_generation):
            elites_kept = population[-2:]
            clan_collection = self.get_clans(population)
            updated_clan_collection = self.update_position(clan_collection)
            new_population = self.combine_clans(updated_clan_collection)
            population = self.evaluate_new_population(new_population)
            population = self.sort_population_by_fitness(population)
            population[0:2] = elites_kept

            a = self.sort_population_by_fitness(population)
            b = self.get_fitness(a)
            result.append(b[-1])

        return result


class GenericAlgorithm(object):

    def __init__(self, pop_size, gen_size, num_selected_parents, crossover_rate, mutation_rate, epochs):

        self.pop_size = pop_size        # number of population
        self.gen_size = gen_size        # number of genes in each chromosome
        self.num_selected_parents = num_selected_parents    # the number of selected parents in each epoch
        self.mutation_rate = mutation_rate      # mutation rate
        self.crossover_rate = crossover_rate    # crossover_rate
        self.epochs = epochs            # number of epochs

    def initialize_population(self):

        init_population = np.random.uniform(-10, 10, (self.pop_size, self.gen_size))

        return init_population

    def choose_population(self, population, num_citizens):
        num_delete = self.pop_size - num_citizens
        random = np.random.permutation(self.pop_size)
        delete_index = random[0:num_delete]
        population = np.delete(population, delete_index, 0)

        return population

    # def get_fitness(self, pop):
    #     fitness = np.zeros((self.pop_size, 1))
    #     for i in range(0, self.pop_size):
    #         sum = 0
    #         # print(pop[i])
    #         for j in range(0, self.gen_size):
    #             if (j % 2 == 0):
    #                 sum += pop[i, j] ** 2
    #             else:
    #                 sum += pop[i, j] ** 3
    #         fitness[i] = sum
    #     return fitness

    def get_fitness(self, population):
        fitness_value = np.zeros((self.pop_size, 1))
        for i in range(self.pop_size):
            fitness = 0
            # apline function
            # for j in range(self.gen_size):
            #     a = population[i, j]
            #     fitness += math.fabs(a * math.sin(a) + 0.1 * a)

            # ackley function
            # first_sum = 0
            # second_sum = 0
            # for j in range(self.gen_size):
            #     first_sum += population[i, j] ** 2
            #     second_sum += math.cos(2 * math.pi * population[i, j])
            # fitness = -20.0 * math.exp(-0.2 * math.sqrt(first_sum / self.gen_size)) - math.exp(
            #     second_sum / self.gen_size) \
            #           + 20 + math.e

            # Grieward function
            part1 = 0
            part2 = 1
            for j in range(self.gen_size):
                a = population[i, j]
                part1 += a ** 2
                part2 *= math.cos(a / math.sqrt(j+1))

            fitness += 1 + (1 / 4000) * part1 - part2
            fitness_value[i] += fitness
        return fitness_value

    def get_best_fitness(self, population):

        fitness_value = self.get_fitness(population)
        indices = np.where(fitness_value == np.min(fitness_value))[0]
        index = indices[0]
        # print(indices)
        best_solution = population[index, :]
        best_fitness = fitness_value[index]
        # print(best_fitness)
        return best_fitness.reshape((1,))

    def get_index_chromosome_by_fitness(self, value, fitness_value, population):
        index = np.where(fitness_value == value)[0]
        chromosome = population[index]
        return chromosome[0]

    def select_mating_pool(self, num_parents, population):

        selected_parents = np.zeros((num_parents, self.gen_size))

        fitness_value = self.get_fitness(population)
        # print(fitness_value)
        sorted_fitness = np.sort(fitness_value, axis=0)
        # print(sorted_fitness)
        for i in range(num_parents):
            parent = self.get_index_chromosome_by_fitness(sorted_fitness[i], fitness_value, population)
            selected_parents[i] += parent

        return selected_parents

    def choose_parent_pair(self, parents):
        size = parents.shape[0]
        parent_pair_list = []
        parents_indices = np.random.permutation(size)
        for i in range(0, size, 2):
            parent1_index = parents_indices[i]
            parent2_index = parents_indices[i+1]
            parent1 = parents[parent1_index].reshape((1, self.gen_size))
            parent2 = parents[parent2_index].reshape((1, self.gen_size))
            pair = [parent1, parent2]
            parent_pair_list.append(pair)

        return parent_pair_list

    def crossover(self, parent_pair):

        parent1 = parent_pair[0]
        parent2 = parent_pair[1]

        child1 = np.zeros((1, self.gen_size))
        child2 = np.zeros((1, self.gen_size))

        num_gens_parent1 = int(self.gen_size*self.crossover_rate)
        num_gens_parent2 = self.gen_size - num_gens_parent1

        permutation = np.random.permutation(self.gen_size)

        for i in range(num_gens_parent1):
            index = permutation[i]
            child1[:, index] += parent1[:, index]
            child2[:, index] += parent2[:, index]

        permutation = permutation[num_gens_parent1:self.gen_size]

        for i in permutation:
            index = i
            child1[:, index] += parent2[:, index]
            child2[:, index] += parent1[:, index]

        return child1, child2

    def mutation(self, child):

        a = np.random.randint(0, self.gen_size-1, 2)
        a1 = a[0]
        # print(a1)
        range = int(self.mutation_rate*self.pop_size)
        if a1+range > self.pop_size:
            a2 = int(a1 + range)
        else:
            a2 = int(a1 - range)

        if a1 < a2:
            selected_part = child[:, a1:a2]
            reversed_part = np.flip(selected_part)
            child[:, a1:a2] = reversed_part

        if a1 > a2:
            selected_part = child[:, a2:a1]
            reversed_part = np.flip(selected_part)
            child[:, a2:a1] = reversed_part

        return child

    def run(self):
        population = self.initialize_population()
        result = []
        result_file_path = "result_GA.csv"
        for i in range(self.epochs):
            parents = self.select_mating_pool(int(self.pop_size/2), population)
            population = self.choose_population(population, int(self.pop_size / 2))
            pair_list = self.choose_parent_pair(parents)

            for pair in pair_list:
                child1, child2 = self.crossover(pair)
                population = np.concatenate((population, self.mutation(child1), self.mutation(child2)), axis=0)

            epoch = i+1
            best_fitness = np.round(self.get_best_fitness(population), 3)
            result.append(best_fitness)

        return result, population



