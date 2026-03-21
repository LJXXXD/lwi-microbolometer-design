import numpy as np

# Define the chromosome length (number of parameters)
CHROMOSOME_LENGTH = 5

# Define the population size
POPULATION_SIZE = 50

# Define the number of generations
GENERATIONS = 100

# Define the mutation rate
MUTATION_RATE = 0.01


def create_population(size, chromosome_length):
    return np.random.rand(size, chromosome_length)


def fitness_function(chromosome):
    # Replace this with your own fitness function
    return np.sum(chromosome)


def select_parents(population, fitness):
    parents = population[np.argsort(fitness)[-2:]]
    return parents


def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, CHROMOSOME_LENGTH)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2


def mutate(chromosome):
    for i in range(len(chromosome)):
        if np.random.rand() < MUTATION_RATE:
            chromosome[i] = np.random.rand()
    return chromosome


def genetic_algorithm():
    population = create_population(POPULATION_SIZE, CHROMOSOME_LENGTH)

    for generation in range(GENERATIONS):
        fitness = np.array([fitness_function(chromosome) for chromosome in population])
        new_population = []

        for _ in range(POPULATION_SIZE // 2):
            parents = select_parents(population, fitness)
            child1, child2 = crossover(parents[0], parents[1])
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = np.array(new_population)

        # Output the best fitness in each generation
        best_fitness = np.max(fitness)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    # Output the best solution
    best_index = np.argmax(fitness)
    best_solution = population[best_index]
    print("Best Solution:", best_solution)
    print("Best Fitness:", fitness[best_index])


if __name__ == "__main__":
    genetic_algorithm()
