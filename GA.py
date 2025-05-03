import numpy as np
import matplotlib.pyplot as plt
import random

def read_distances(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    n_cities = len(lines) - 1
    distances = np.zeros((n_cities, n_cities))
    intra_city_distances = np.zeros(n_cities)
    city_names = lines[0].strip().split()[1:]  # 从第一行读取城市名称，跳过第一个元素
    start_index = city_names.index("start")  # 找到起点的索引
    for i in range(1, len(lines)):
        parts = lines[i].strip().split()
        intra_city_distances[i - 1] = float(parts[i])  # 读取城市内距离（对角线上的数据）
        for j in range(1, len(parts)):
            distances[j - 1][i - 1] = float(parts[j])  # 转置矩阵以匹配输入和输出
    return city_names, distances, intra_city_distances, start_index

def create_individual(n_cities, n_drones, start_index):
    # 随机分配城市给无人机，确保每个城市只被分配一次
    individual = [[] for _ in range(n_drones)]
    cities = list(range(n_cities))
    cities.remove(start_index)  # 移除起点
    random.shuffle(cities)
    for i, city in enumerate(cities):
        drone_index = i % n_drones
        individual[drone_index].append(city)
    return individual

def create_population(pop_size, n_cities, n_drones, start_index):
    # 创建初始种群
    return [create_individual(n_cities, n_drones, start_index) for _ in range(pop_size)]

def calculate_total_distance(individual, distances, intra_city_distances, start_index):
    total_distance = 0
    drone_distances = []
    for drone_route in individual:
        if not drone_route:
            drone_distances.append(0)
            continue
        # 从起点到第一个城市
        drone_distance = distances[start_index][drone_route[0]]
        for i in range(len(drone_route)):
            city = drone_route[i]
            # 城市内距离
            drone_distance += intra_city_distances[city]
            if i < len(drone_route) - 1:
                next_city = drone_route[i + 1]
                # 城市间距离
                drone_distance += distances[city][next_city]
        # 从最后一个城市回到起点
        drone_distance += distances[drone_route[-1]][start_index]
        total_distance += drone_distance
        drone_distances.append(drone_distance)
    return total_distance, drone_distances

def fitness(individual, distances, intra_city_distances, start_index):
    # 适应度函数，总距离越短，适应度越高
    total_distance, _ = calculate_total_distance(individual, distances, intra_city_distances, start_index)
    return 1 / total_distance

def tournament_selection(population, distances, intra_city_distances, start_index, tournament_size):
    tournament = random.sample(population, tournament_size)
    best = max(tournament, key=lambda x: fitness(x, distances, intra_city_distances, start_index))
    return best

# 优化后的交叉操作
def order_crossover(parent1, parent2):
    child = [[] for _ in range(len(parent1))]
    for i in range(len(parent1)):
        start, end = sorted(random.sample(range(len(parent1[i])), 2))
        child[i] = parent1[i][start:end]
        remaining = [city for city in parent2[i] if city not in child[i]]
        child[i] = remaining[:start] + child[i] + remaining[start:]
    return child

# 优化后的变异操作
def swap_mutation(individual, mutation_rate):
    for drone_route in individual:
        if random.random() < mutation_rate and len(drone_route) > 1:
            idx1, idx2 = random.sample(range(len(drone_route)), 2)
            drone_route[idx1], drone_route[idx2] = drone_route[idx2], drone_route[idx1]
    return individual

# 验证个体是否有效
def validate_individual(individual, start_index):
    all_cities = []
    for drone_route in individual:
        all_cities.extend(drone_route)
    return len(all_cities) == len(set(all_cities)) and start_index not in all_cities

# 优化后的遗传算法
def genetic_algorithm(pop_size, generations, n_cities, n_drones, distances, intra_city_distances, start_index, tournament_size, mutation_rate):
    population = create_population(pop_size, n_cities, n_drones, start_index)
    best_fitness_history = []
    best_solution = None
    best_fitness = 0

    for gen in range(generations):
        new_population = []
        for _ in range(pop_size):
            parent1 = tournament_selection(population, distances, intra_city_distances, start_index, tournament_size)
            parent2 = tournament_selection(population, distances, intra_city_distances, start_index, tournament_size)
            child = order_crossover(parent1, parent2)
            child = swap_mutation(child, mutation_rate)
            # 如果生成的个体无效，则重新生成
            while not validate_individual(child, start_index):
                parent1 = tournament_selection(population, distances, intra_city_distances, start_index, tournament_size)
                parent2 = tournament_selection(population, distances, intra_city_distances, start_index, tournament_size)
                child = order_crossover(parent1, parent2)
                child = swap_mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population
        current_best = max(population, key=lambda x: fitness(x, distances, intra_city_distances, start_index))
        current_fitness = fitness(current_best, distances, intra_city_distances, start_index)

        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = current_best

        best_fitness_history.append(1 / best_fitness)

        # 早停机制：如果连续50代没有提升，则停止
        if gen > 50 and len(set(best_fitness_history[-50:])) == 1:
            print("早停机制触发，提前终止迭代。")
            break

    return best_solution, best_fitness_history

# 主程序
file_path = 'path_lengths_A.txt'  # 替换为你的文件路径
city_names, distances, intra_city_distances, start_index = read_distances(file_path)
n_cities = len(city_names)
n_drones = 5
pop_size = 500
generations = 150
tournament_size = 5
mutation_rate = 0.1

best_solution, best_fitness_history = genetic_algorithm(pop_size, generations, n_cities, n_drones, distances, intra_city_distances, start_index, tournament_size, mutation_rate)
total_distance, drone_distances = calculate_total_distance(best_solution, distances, intra_city_distances, start_index)

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 打印结果
all_scanned_cities = set()
for i, drone_route in enumerate(best_solution):
    route_names = [city_names[start_index]] + [city_names[city] for city in drone_route] + [city_names[start_index]]
    all_scanned_cities.update(route_names[1:-1])  # 排除重复的起点
    print(f"UAV {i + 1} Flight path: {' -> '.join(route_names)}")
    print(f"UAV {i + 1} Flight distance: {drone_distances[i]}")

# 检查是否所有城市都被扫描
all_cities_set = set(city_names)
all_cities_set.remove(city_names[start_index])  # 移除起点
missing_cities = all_cities_set - all_scanned_cities
if missing_cities:
    print(f"以下城市未被扫描: {missing_cities}")
else:
    print("All cities have been scanned。")

print(f"Total flight distance: {total_distance}")

# 可视化
plt.plot(range(len(best_fitness_history)), best_fitness_history)
plt.xlabel('Number of iterations')
plt.ylabel('Total flight distance')
plt.title('Images of the total distance and number of iterations flown by five UAVs')
plt.show()