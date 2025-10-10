import csv
import time
from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Vrp():
    def __init__(self):
        self.distance = []  # 各个客户及配送中心距离
        self.q = []  # 客户需求量
        self.savings = []  # 节约度
        self.Routes = []  # 路线
        self.Cost = 0  # 总路程
        self.tons = 500  # 车辆载重(校园里的话，设成18就可以了)
        self.distanceLimit=1000000 #这里先不考虑距离约束，假设只有容量影响

    def datainput(self, distance_matrix_file="distance_matrix.csv", network_file="network.csv"):
        # 读取距离矩阵
        df = pd.read_csv(distance_matrix_file, header=None)
        self.distance=df.values[1:, :].astype(float) #把第一行切掉了，不然把id也给算进去了

        # 读取网络数据并提取需求量
        network_data = pd.read_csv(network_file)
        self.q = network_data['Demand'].tolist()

    def savingsAlgorithms(self):
        # 初始化路线
        for i in range(1, len(self.q)):
            self.Routes.append([i])

        # 计算节约里程
        for i in range(1, len(self.q)):
            for j in range(1, len(self.q)):
                if i != j:
                    saving = (self.distance[i][0] + self.distance[0][j]) - self.distance[i][j]
                    if saving > 0:  # 只保留正的节约里程
                        self.savings.append([i, j, saving])

        # 按节约里程排序
        self.savings = sorted(self.savings, key=itemgetter(2), reverse=True)

        # 合并路线
        for i in range(len(self.savings)):
            startRoute = []
            endRoute = []
            routeDemand = 0
            for j in range(len(self.Routes)):
                if self.savings[i][0] == self.Routes[j][-1]:
                    endRoute = self.Routes[j]
                elif self.savings[i][1] == self.Routes[j][0]:
                    startRoute = self.Routes[j]

            if len(startRoute) != 0 and len(endRoute) != 0:
                for k in range(len(startRoute)):
                    routeDemand += self.q[startRoute[k]]
                for k in range(len(endRoute)):
                    routeDemand += self.q[endRoute[k]]
                routeDistance = 0
                routestore = [0] + endRoute + startRoute + [0]
                for l in range(len(routestore) - 1):
                    routeDistance += self.distance[routestore[l]][routestore[l + 1]]

                if routeDemand <= self.tons and routeDistance <= self.distanceLimit:
                    self.Routes.remove(startRoute)
                    self.Routes.remove(endRoute)
                    self.Routes.append(endRoute + startRoute)

        # 添加起始点和结束点
        for i in range(len(self.Routes)):
            self.Routes[i].insert(0, 0)
            self.Routes[i].append(0)

    def printRoutes(self):
        for i in self.Routes:
            costs = 0
            for j in range(len(i) - 1):
                costs += self.distance[i[j]][i[j + 1]]
            print("路线: ", i, " 路程: ", costs)

    def calcCosts(self):
        self.Cost = 0
        for i in range(len(self.Routes)):
            for j in range(len(self.Routes[i]) - 1):
                self.Cost += self.distance[self.Routes[i][j]][self.Routes[i][j + 1]]

        print("\n总距离: ", round(self.Cost, 3))

    def plotRoutes(self, network_file="network.csv"):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        network_data = pd.read_csv(network_file)
        coords = network_data[['X', 'Y']].values

        # 先绘制配送中心（起点），确保其颜色为红色
        plt.scatter(coords[0][0], coords[0][1], c='red', label='配送中心', zorder=5)

        # 绘制路线
        for route in self.Routes:
            x_coords = [coords[i][0] for i in route]
            y_coords = [coords[i][1] for i in route]
            plt.plot(x_coords, y_coords, marker='o', zorder=1)

        # 标注每个节点的编号
        for i, (x, y) in enumerate(coords):
            plt.text(x, y, str(i), fontsize=12, ha='right')

        plt.title('优化路径的结果展示')
        plt.xlabel('X 坐标')
        plt.ylabel('Y 坐标')
        plt.legend()
        plt.show()

    def start(self):
        print("== 导入数据 ==")
        self.datainput()
        print("== 距离矩阵 ==")
        for row in self.distance:
            print(row)
        print("== 需求量 ==")
        print(self.q)
        print("== 节约度 ==")
        start_time=time.time() #这里插个start计时器
        self.savingsAlgorithms()
        end_time=time.time() #插个end计时器
        print("== 结果 ==")
        self.printRoutes()
        self.calcCosts()
        self.plotRoutes()
        print(end_time-start_time)


if __name__ == '__main__':
    vrp = Vrp()
    vrp.start()
