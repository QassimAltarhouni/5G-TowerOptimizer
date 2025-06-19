import numpy as np
from fitness_function import calculate_fitness
from knowledge_utils import apply_knowledge_rules


def initialize_population(pop_size, num_towers):
    return np.random.randint(0, 2, size=(pop_size, num_towers))

def evaluate_population(population, df_towers, df_users, normalization_bounds=None):
    fitness_scores = []
    for individual in population:
        active_towers = df_towers.copy()
        active_towers["active"] = individual
        df_active = active_towers[active_towers["active"] == 1].copy()
        score = calculate_fitness(
            df_active,
            df_users,
            normalization_bounds=normalization_bounds,
            verbose=False,
        )["fitness"]
        fitness_scores.append(score)
    return np.array(fitness_scores)

def select_parents(population, fitness_scores, num_parents):
    sorted_indices = np.argsort(fitness_scores)
    return population[sorted_indices[:num_parents]]

def uniform_crossover(parent1, parent2):
    mask = np.random.randint(0, 2, size=parent1.shape).astype(bool)
    return np.where(mask, parent1, parent2)

def mutate(individual, mutation_rate=0.05):
    mutation_mask = np.random.rand(len(individual)) < mutation_rate
    individual[mutation_mask] = 1 - individual[mutation_mask]
    return individual

def run_ga(df_towers, df_users, pop_size=30, num_generations=100, mutation_rate=0.05, num_parents=10, normalization_bounds=None):
    num_towers = len(df_towers)
    population = initialize_population(pop_size, num_towers)

    best_solution = None
    best_fitness = float("inf")

    for generation in range(num_generations):
        print(f"Generation {generation + 1}/{num_generations}")

        fitness_scores = evaluate_population(
            population,
            df_towers,
            df_users,
            normalization_bounds=normalization_bounds
        )

        best_idx = np.argmin(fitness_scores)
        if fitness_scores[best_idx] < best_fitness:
            best_fitness = fitness_scores[best_idx]
            best_solution = population[best_idx].copy()

        print(f"   🔹 Best Fitness: {best_fitness:.4f}")

        parents = select_parents(population, fitness_scores, num_parents)
        children = []

        while len(children) < pop_size:
            p1 = parents[np.random.randint(0, num_parents)]
            p2 = parents[np.random.randint(0, num_parents)]
            child = uniform_crossover(p1, p2)
            child = mutate(child, mutation_rate)
            children.append(child)

        population = np.array(children)

    return best_solution, best_fitness

def run_kbga(df_towers, df_users, pop_size=30, num_generations=100, mutation_rate=0.05, num_parents=10, normalization_bounds=None):
    num_towers = len(df_towers)
    population = initialize_population(pop_size, num_towers)

    best_solution = None
    best_fitness = float("inf")

    for generation in range(num_generations):
        print(f"[KBGA] Generation {generation + 1}/{num_generations}")

        fitness_scores = evaluate_population(
            population,
            df_towers,
            df_users,
            normalization_bounds=normalization_bounds
        )

        best_idx = np.argmin(fitness_scores)
        if fitness_scores[best_idx] < best_fitness:
            best_fitness = fitness_scores[best_idx]
            best_solution = population[best_idx].copy()

        print(f"   🔹 Best Fitness: {best_fitness:.4f}")

        parents = select_parents(population, fitness_scores, num_parents)
        children = []

        while len(children) < pop_size:
            p1 = parents[np.random.randint(0, num_parents)]
            p2 = parents[np.random.randint(0, num_parents)]
            child = uniform_crossover(p1, p2)
            child = mutate(child, mutation_rate)

            # 👇 تطبيق قواعد المعرفة بعد الطفرة
            child = apply_knowledge_rules(child, df_towers, df_users)

            children.append(child)

        population = np.array(children)

    return best_solution, best_fitness
