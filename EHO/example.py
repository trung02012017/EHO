import numpy as np
from EHO import ElephantHerdingOptimization

def main():
    # pop_sizes = [40, 60, 80, 100, 120, 140, 160, 180, 200]
    # clan_sizes = [5, 10, 20]
    # alphas = [0.4, 0.5, 0.6, 0.7]
    # betas = [0.1, 0.2, 0.3]
    # dim_size = 50
    # max_generation = 1000

    pop_sizes = [50]
    clan_sizes = [10]
    alphas = [0.5]
    betas = [0.1]
    dim_size = 50
    max_generation = 1000

    combinations = []

    for pop_size in pop_sizes:
        for clan_size in clan_sizes:
            for alpha in alphas:
                for beta in betas:
                    combination = [pop_size, clan_size, alpha, beta, dim_size, max_generation]
                    combinations.append(combination)

    for combination in combinations:
        pop_size = combination[0]
        clan_size = combination[1]
        alpha = combination[2]
        beta = combination[3]
        dim_size = combination[4]
        max_generation = combination[5]

        EHO = ElephantHerdingOptimization(pop_size, clan_size, dim_size, max_generation, alpha, beta)
        population = EHO.initialize_population()
        # print(population)
        # print(EHO.get_fitness(population))
        # print(EHO.get_fitness(EHO.sort_population_by_fitness(population)))

        result_EHO = EHO.run()


if __name__ == "__main__":
    main()