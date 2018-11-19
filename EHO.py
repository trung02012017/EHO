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

    # devide population into clans => list clan collection
    def get_clans(self, population):

        clan_collection = []
        num_clans = math.ceil(self.pop_size/self.clan_size)

        for i in range(num_clans):

            a = int(i*self.clan_size)
            b = int(i*self.clan_size + self.clan_size)

            clan_i = population[a:b, :]
            clan_collection.append(clan_i)

        return clan_collection

    # fitness of a clan
    def get_fitness(self, clan):

        fitness_value = np.zeros((self.clan_size, 1))
        for i in range(self.clan_size):
            fitness = 0
            for j in range(self.dim_size):
                if j % 2 == 0:
                    fitness += clan[i, j] ** 3
                else:
                    fitness += clan[i, j] ** 2

            fitness_value[i] += fitness

        return fitness_value

    # get matriarch - the fittest one
    def get_matriarch(self, clan):

        fitness_value = self.get_fitness(clan)
        indices = np.where(fitness_value == np.amin(fitness_value))[0]
        index = indices[0]
        matriarch = clan[index, :]

        return matriarch, index

    # get male elephant which will leave his clan - the worst one
    def get_mature_male_elephant(self, clan):

        fitness_value = self.get_fitness(clan)
        # print(fitness_value)
        indices = np.where(fitness_value == np.amax(fitness_value))[0]
        index = indices[0]
        male_elephant = clan[index, :]

        return male_elephant, index

    # Updating operation ( in a clan, each one will be updated follow the leadership of the matriarch
    # matriarch will be updated follow the center of clan
    def updating_operation(self, clan):

        matriarch, matriarch_index = self.get_matriarch(clan)
        r = np.random.uniform(0, 1)

        # print(r)
        # print("clan", clan)
        new_clan = clan + self.alpha*(matriarch - clan) * r
        # print(new_clan)

        # update matriarch
        clan_center = np.mean(clan, axis=0)
        new_child = matriarch + self.beta*clan_center
        new_clan[matriarch_index] = new_child

        # print("new_clan", new_clan)

        return new_clan

    # separating operation - when a male leave his clan and a baby will be born in new generation
    # to replace the male leaving
    def separating_operation(self, clan):

        male_elephant, male_elephant_index = self.get_mature_male_elephant(clan)
        # print(male_elephant)
        new_child_index = male_elephant_index

        max_elephant = 10 * np.ones(self.dim_size)
        min_elephant = -10 * np.ones(self.dim_size)
        rand = np.random.uniform(-1, 1, self.dim_size)

        new_child = min_elephant + (max_elephant - min_elephant + 1) * rand
        # new_child = matriarch * rand
        # print(new_child)

        return new_child, new_child_index

    # updating position and create new generation
    def update_position(self, clan):

        new_clan = self.updating_operation(clan)
        new_child, new_child_index = self.separating_operation(clan)
        new_clan[new_child_index] = new_child

        for i in range(new_clan.shape[0]):
            for j in range(new_clan.shape[1]):
                if new_clan[i, j] < -10:
                    new_clan[i, j] = -10
                if new_clan[i, j] > 10:
                    new_clan[i, j] = 10
        return new_clan

    # get best result of generation
    def get_result(self, clan_collection):

        num_clans = len(clan_collection)
        result = np.zeros(num_clans)

        for i in range(num_clans):

            fitness = self.get_fitness(clan_collection[i])
            best_fitness = np.amin(fitness)
            result[i] += best_fitness

        # print(result)

        best_result = np.amin(result)
        # index =  np.where(best_result == np.amin(best_result))[0]
        # index = index[0]
        # matriarch, matriarch_index = self.get_matriarch(clan_collection[index])

        return best_result

    def run(self):

        result = []
        population = self.initialize_population()
        clan_collection = self.get_clans(population)

        for i in range(self.max_generation):
            new_generation = []
            for clan in clan_collection:

                new_clan_i = self.update_position(clan)
                new_generation.append(new_clan_i)

            # print(self.get_result(new_generation), i+1)
            clan_collection = new_generation

            result.append(self.get_result(clan_collection))

        return result





