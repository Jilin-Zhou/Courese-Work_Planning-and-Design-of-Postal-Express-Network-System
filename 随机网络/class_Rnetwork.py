import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import os
class network:
    # 第一步：生成一个numnode大小的随机网络，并生成物流配送需求,这里直接使用random库生成二维随机数,可以在构造函数里传入希望生成的网络中的节点数，后面优化
    def __init__(self,numnode):
        num_nodes = numnode
        max_load = 500
        np.random.seed(80)
        nodes = np.random.rand(num_nodes, 2) * 100
        demands = np.random.randint(20, 51, num_nodes)  # 随机生成节点需求（更好的构思其实是随机生成人口，然后用重力模型估计需求）
        
        # 将节点和需求结合在一起做成一个表
        self.data = np.column_stack((nodes, demands))
        
        
        # 第二步：利用重心法确定配送中心的位置
        # 这里采用加权平均坐标的方法确定配送中心位置，后面可以改成求解一个更好的无约束优化问题，然后直接调Gurobi求解
        weights = self.data[:, 2]
        center_x = np.average(self.data[:, 0], weights=weights)
        center_y = np.average(self.data[:, 1], weights=weights)
        self.distribution_center = np.array([center_x, center_y, 0])  # 配送中心节点，需求量为0

        self.dist_matrix = distance_matrix(self.data[:, :2], self.data[:, :2])

        def save_data(self, output_dir):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            df = pd.DataFrame(self.data, columns=['X', 'Y', 'Demand'])
            df.to_csv(f"{output_dir}/network.csv", index=False)
            dist_df = pd.DataFrame(self.dist_matrix)
            dist_df.to_csv(f"{output_dir}/distance_matrix.csv", index=False)

        save_data(self,"result")

        # 第三步：使用旋转射线法对节点进行分区，大体思路是角度计算加上个累加器，累加到最大承载量就产生新分区，这样的话就能把复杂网络拆分成很多个子网络
        def partition_nodes(nodes, center, max_load):
            self.partitions = []
            # 这里千万小心点，每个分区都得有配送中心，这里可以定义为空，但是下面重新初始化的时候不能定义为空
            current_partition = [self.distribution_center]
            current_load = 0
            angles = np.arctan2(nodes[:, 1] - center[1], nodes[:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            sorted_nodes = nodes[sorted_indices]
        
            for node in sorted_nodes:
                demand = node[2]
                if current_load + demand > max_load:
                    self.partitions.append(current_partition)
                    # 满了就重新初始化
                    current_partition = [self.distribution_center]
                    current_load = 0
                current_partition.append(node)
                current_load += demand
        
            if current_partition:
                self.partitions.append(current_partition)
        
            return self.partitions
        
        


        self.partitions = partition_nodes(self.data, self.distribution_center, max_load)

        # 输出每个分区的X-Y坐标为不同的CSV文件
        self.output_dir = "result/partitions"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        for i, partition in enumerate(self.partitions):
            df = pd.DataFrame(partition, columns=['X', 'Y', 'Demand'])
            df.to_csv(f"{self.output_dir}/partition_{i + 1}.csv", index=False)
    
    
        # 生成每个分区的距离矩阵并输出为CSV文件
        for i, partition in enumerate(self.partitions):
            partition_array = np.array(partition)  # Convert partition to a NumPy array
            df = pd.DataFrame(partition_array, columns=['X', 'Y', 'Demand'])
            dist_matrix = distance_matrix(partition_array[:, :2], partition_array[:, :2])
            dist_df = pd.DataFrame(dist_matrix)
            dist_df.to_csv(f"{self.output_dir}/partition_{i + 1}_distance_matrix.csv", index=False)


    
    # 绘制节点和配送中心，方便后续的可视化展示,这里也要优化下，应该随网络直接生成，再调函数的话会不会很鸡肋？
    def printPic(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.scatter(self.data[:, 0], self.data[:, 1], c='blue', label='节点')
        plt.scatter(self.distribution_center[0], self.distribution_center[1], c='red', label='配送中心')
        plt.legend()
        plt.xlabel('X 坐标')
        plt.ylabel('Y 坐标')
        plt.title('物流配送网络')
        plt.show()



    def getpartitions(self):
        return self.partitions
    
    def getPosition(self):
        return self.output_dir

    def getDistributionCenter(self):
        return self.distribution_center



