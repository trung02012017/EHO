from EHO import ElephantHerdingOptimization


def main():
    pop_size = 100
    clan_size = 5
    dim_size = 50
    max_generation = 1000
    alpha = 0.5
    beta = 0.1

    EHO = ElephantHerdingOptimization(pop_size, clan_size, dim_size, max_generation, alpha, beta)

    # init_population = EHO.initialize_population()
    # clan_collection = EHO.get_clans(init_population)
    # clan = clan_collection[0]
    # print(EHO.get_fitness(clan))
    # print(clan)
    #
    # new_child, index = EHO.separating_operation(clan)
    # clan[index] = new_child
    #
    # for i in range(clan.shape[0]):
    #     for j in range(clan.shape[1]):
    #         if clan[i, j] < -10:
    #             clan[i, j] = -10
    #         if clan[i, j] > 10:
    #             clan[i, j] = 10
    #
    # print(EHO.get_fitness(clan))
    # print(clan)
    EHO.run()


if __name__ == "__main__":
    main()