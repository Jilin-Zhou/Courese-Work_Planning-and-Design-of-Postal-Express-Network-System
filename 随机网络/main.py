import time

from class_ALNS import ALNSOptimizer
import pandas as pd
from class_Rnetwork import network
import matplotlib.pyplot as plt
import numpy as np
from class_saving import Vrp

par = network(500)
vrp = Vrp()
iterations = 1000
def read_distance_matrix(file_path):
    df = pd.read_csv(file_path)
    distance_matrix = df.values[:, :].astype(float)
    return distance_matrix

partitions = par.getpartitions()
input_dir = par.getPosition()
distribution_center = par.getDistributionCenter()

# 对每个分区调用算法，这里可以通过修改opt来实现几种启发式和ALNS的切换
sum=0
sum_time=0
for i, partition in enumerate(partitions):
    start_time = time.time()
    # 读取距离矩阵
    dist_matrix = read_distance_matrix(f"{input_dir}/partition_{i + 1}_distance_matrix.csv")

    # 初始化
    optimizer = ALNSOptimizer(dist_matrix)

    # 运行算法以求解
    best_route, best_cost = optimizer.run(iterations)
    sum+=best_cost
    end_time=time.time()
    sum_time+=end_time-start_time
# 以下是在调用启发式
# for i, partition in enumerate(partitions):
#
#
#     # 运行算法以求解(返回值还没写，之后写上)
#     best_route, best_cost = vrp.start()

    # 输出路径和最优解
    with open(f"{input_dir}/partition_{i + 1}_solution.csv", "w") as f:
        f.write("Route,Cost\n")
        # f.write(f"{best_route},{best_cost}\n")
        f.write(f"\"{best_route}\",{best_cost}\n")


def plot_partition(partition, route, center, ax, show_labels=True):
    partition = np.array(partition)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    x_coords = partition[:, 0]
    y_coords = partition[:, 1]

    ax.scatter(x_coords, y_coords, c='blue', label='节点' if show_labels else "")
    ax.scatter(center[0], center[1], c='red', label='配送中心' if show_labels else "",zorder=5)

    for i in range(len(route)):
        start_node = partition[route[i - 1]]
        end_node = partition[route[i]]
        ax.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], 'k-')

    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Y 坐标')
    if show_labels:
        ax.legend()

# 单独绘制每个分区的路径
for i, partition in enumerate(partitions):
    solution_file = f"{input_dir}/partition_{i + 1}_solution.csv"
    solution_df = pd.read_csv(solution_file)

    route_str = solution_df['Route'][0]  # 读取路径字符串
    route = eval(route_str)  # 解析路径字符串

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_partition(partition, route, distribution_center, ax)
    plt.title(f'分区 {i + 1} 的配送路线')
    plt.savefig(f"{input_dir}/partition_{i + 1}_route.png")
    plt.show()

# 合在一起绘制所有分区的路径,合的时候标签也合了，之后改改(加了个标签，OK了)
fig = plt.figure()
ax = fig.add_subplot(111)
for i, partition in enumerate(partitions):
    solution_file = f"{input_dir}/partition_{i + 1}_solution.csv"
    solution_df = pd.read_csv(solution_file)

    route_str = solution_df['Route'][0]
    route = eval(route_str)
    plot_partition(partition, route, distribution_center, ax, show_labels=(i == 0))



print(sum)
print(sum_time)
plt.title('随机网络的配送路径优化')
plt.savefig(f"{input_dir}/all_partitions_routes.png")
plt.show()
