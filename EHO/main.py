from EHO import ElephantHerdingOptimization
from EHO import GenericAlgorithm
import matplotlib.pyplot as plt
import time


def save_fig(EHO_result, GA_result, EHO_time, GA_time):
    plt.figure(2)
    plt.plot(EHO_result, '-r', label='EHO')
    plt.plot(GA_result, '-b', label="GA")
    plt.legend()
    name = ''
    name += 'EHO_time = '
    name += str(EHO_time)
    name += ' and GA_time = '
    name += str(GA_time)
    name += ' (Grieward function)'
    name += '.png'
    plt.savefig(name)
    plt.show()


def main():

###### EHO #######
    pop_sizes = [50]
    clan_sizes = [10]
    alphas = [0.6]
    betas = [0.1]
    dim_size = 50
    max_generation = 3000

    combinations = []

    for pop_size in pop_sizes:
        for clan_size in clan_sizes:
            for alpha in alphas:
                for beta in betas:
                    combination = [pop_size, clan_size, alpha, beta, dim_size, max_generation]
                    combinations.append(combination)

    for combination in combinations:
        start_time = time.time()
        pop_size = combination[0]
        clan_size = combination[1]
        alpha = combination[2]
        beta = combination[3]
        dim_size = combination[4]
        max_generation = combination[5]

        EHO = ElephantHerdingOptimization(pop_size, clan_size, dim_size, max_generation, alpha, beta)

        EHO_result = EHO.run()

        end_time = time.time()
        execute_EHO_time = end_time - start_time
        EHO_time = round(execute_EHO_time, 3)

##### GA ########

    pop_size = 100
    gen_size = 50
    num_selected_parents = int(pop_size / 2)
    crossover_rate = 0.4
    mutation_rate = 0.05
    epochs = 3000

    start_time = time.time()
    GA = GenericAlgorithm(pop_size, gen_size, num_selected_parents, crossover_rate, mutation_rate, epochs)
    GA_result, population = GA.run()
    end_time = time.time()
    execute_EHO_time = end_time - start_time
    GA_time = round(execute_EHO_time, 3)

    save_fig(EHO_result=EHO_result, GA_result=GA_result, EHO_time=EHO_time, GA_time=GA_time)


if __name__ == "__main__":
    main()