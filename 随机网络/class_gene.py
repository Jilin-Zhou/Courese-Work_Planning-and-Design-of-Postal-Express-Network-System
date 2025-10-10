import time
import numpy as np
import random
import matplotlib.pyplot as plt

# 读取数据
distance_matrix = np.loadtxt("result/distance_matrix.csv", delimiter=",", skiprows=1)
network_data = np.loadtxt("result/network.csv", delimiter=",", skiprows=1)

# 遗传算法参数
POPULATION_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.01
ELITE_SIZE = 10


# 初始化种群
def create_individual():
    path = [0] + random.sample(range(1, len(distance_matrix)), len(distance_matrix) - 1)
    return path


def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]


# 计算路径长度
def calculate_path_length(path):
    return sum(distance_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1)) + distance_matrix[path[-1]][path[0]]


# 适应度函数
def fitness(individual):
    return 1 / calculate_path_length(individual)


# 选择
def selection(population):
    return random.choices(
        population,
        weights=[fitness(ind) for ind in population],
        k=2
    )


# 交叉
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(1, len(parent1)), 2))
    child = [-1] * len(parent1)
    child[0] = 0
    child[start:end] = parent1[start:end]
    remaining = [item for item in parent2 if item not in child and item != 0]
    for i in range(1, len(child)):
        if child[i] == -1:
            child[i] = remaining.pop(0)
    return child


# 变异
def mutate(individual):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(1, len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual


# 主遗传算法循环
def genetic_algorithm():
    population = create_population()
    best_fitness = 0
    best_individual = None

    for generation in range(GENERATIONS):
        population = sorted(population, key=fitness, reverse=True)

        if fitness(population[0]) > best_fitness:
            best_fitness = fitness(population[0])
            best_individual = population[0]

        new_population = population[:ELITE_SIZE]

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = selection(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

        if generation % 100 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness}")

    return best_individual


# 运行算法
start_time=time.time()
best_path = genetic_algorithm()
best_distance = calculate_path_length(best_path)
end_time=time.time()
print(f"Best path: {best_path}")
print(f"Best distance: {best_distance}")


# 绘制最优路径
def plot_best_path(best_path):
    x = network_data[:, 0]
    y = network_data[:, 1]
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(12, 8))
    plt.scatter(x[0],y[0],c='red',s=50,label="配送中心",zorder=5)
    plt.scatter(x[1:], y[1:], c='blue', s=50,label="配送点",zorder=5)
    for i, txt in enumerate(range(len(x))):
        plt.annotate(txt, (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
    for i in range(len(best_path)):
        start = best_path[i]
        end = best_path[(i + 1) % len(best_path)]
        plt.plot([x[start], x[end]], [y[start], y[end]], 'b-')
    plt.title("TSP问题的解")
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.grid(True)
    plt.show()


plot_best_path(best_path)
print(end_time-start_time)