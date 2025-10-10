# 以下没什么大问题
# 需要注意的是轮盘赌那里，要加新算子的话只需要加函数和改一下选择函数就行了
import numpy as np
import random as rd
import copy
import math


# 从文件读取距离矩阵
def read_distance_matrix(file_path):
    with open(file_path, 'r') as f:
        return np.loadtxt(f, delimiter=',')


# 计算路径距离
def disCal(path, distmat):
    distance = 0
    for i in range(len(path) - 1):
        distance += distmat[path[i]][path[i + 1]]
    distance += distmat[path[-1]][path[0]]  # 返回起点
    return distance


# 随机破坏算子
def randomDestroy(sol):
    numToRemove = rd.randint(1, len(sol) // 2)
    removed = rd.sample(sol[1:], numToRemove)  # 不移除起点
    for city in removed:
        sol.remove(city)
    return removed

# 最坏破坏算子
def max3Destroy(sol,distmat):
    solNew = copy.deepcopy(sol)
    removed = []
    dis = []
    for i in range(len(sol) - 1):
        dis.append(distmat[sol[i]][sol[i + 1]])
    dis.append(distmat[sol[-1]][sol[0]])
    disSort = copy.deepcopy(dis)
    disSort.sort()
    for i in range(3):
        if dis.index(disSort[len(disSort) - i - 1]) == len(dis) - 1:
            removed.append(solNew[0])
            sol.remove(solNew[0])
        else:
            removed.append(solNew[dis.index(disSort[len(disSort) - i - 1]) + 1])
            sol.remove(solNew[dis.index(disSort[len(disSort) - i - 1]) + 1])
    return removed



# 随机修复算子
def randomRepair(destroyedSolution, removeList):
    for city in removeList:
        insertPos = rd.randint(1, len(destroyedSolution))  # 不插入到起点之前
        destroyedSolution.insert(insertPos, city)
    return destroyedSolution


# 贪心修复算子
def greedyRepair(destroyedSolution, removeList, distmat):
    for city in removeList:
        bestPos = -1
        bestIncrease = float('inf')
        for i in range(1, len(destroyedSolution) + 1):  # 不插入到起点之前
            newSolution = destroyedSolution[:i] + [city] + destroyedSolution[i:]
            increase = disCal(newSolution, distmat)
            if increase < bestIncrease:
                bestIncrease = increase
                bestPos = i
        destroyedSolution.insert(bestPos, city)
    return destroyedSolution


# 选择并使用破坏算子
def selectAndUseDestroyOperator(destroyWeight,currentSolution,distmat):
    destroyOperator = -1
    sol = copy.deepcopy(currentSolution)
    removedCities = []
    destroyRoulette = np.array(destroyWeight).cumsum()
    r = rd.uniform(0, max(destroyRoulette))
    for i in range(len(destroyRoulette)):
        if destroyRoulette[i] >= r:
            if i == 0:
                destroyOperator = i
                removedCities = randomDestroy(sol)
                break
            elif i == 1:
                destroyOperator = i
                removedCities = max3Destroy(sol,distmat)
                break
    return sol, removedCities, destroyOperator


# 选择并使用修复算子
def selectAndUseRepairOperator(repairWeight, destroyedSolution, removeList, distmat):
    repairOperator = -1
    repairRoulette = np.array(repairWeight).cumsum()
    r = rd.uniform(0, max(repairRoulette))
    for i in range(len(repairRoulette)):
        if repairRoulette[i] >= r:
            if i == 0:
                repairOperator = i
                repairedSolution = randomRepair(destroyedSolution, removeList)
                break
            elif i == 1:
                repairOperator = i
                repairedSolution = greedyRepair(destroyedSolution, removeList, distmat)
                break
    return repairedSolution, repairOperator


# 更新权重
def updateWeights(weights, scores, operatorIndex, scoreGain, rho=0.1):
    scores[operatorIndex] += scoreGain
    weights[operatorIndex] = (1 - rho) * weights[operatorIndex] + rho * scores[operatorIndex]


# 模拟退火接受准则
def acceptNewSolution(current_distance, new_distance, temperature):
    if new_distance < current_distance:
        return True
    else:
        delta = new_distance - current_distance
        acceptance_probability = math.exp(-delta / temperature)
        return rd.random() < acceptance_probability


# 主程序
def main(dist_matrix,iterations):
    distmat = dist_matrix
    city_num = distmat.shape[0]

    # 初始化
    initial_solution = list(range(1, city_num))  # 不包括起点
    rd.shuffle(initial_solution)
    current_solution = [0] + initial_solution  # 从起点出发
    current_distance = disCal(current_solution, distmat)

    # 迭代过程
    destroyWeight = [0.5, 0.5]  # 假设两个破坏算子的权重
    repairWeight = [0.5, 0.5]  # 假设两个修复算子的权重
    destroyScores = [0, 0]  # 破坏算子的初始分数
    repairScores = [0, 0]  # 修复算子的初始分数
    temperature = 100  # 初始温度
    cooling_rate = 0.99  # 冷却速率

    for _ in range(iterations):
        new_solution, removedCities, destroyIndex = selectAndUseDestroyOperator(destroyWeight, current_solution,distmat)
        new_solution, repairIndex = selectAndUseRepairOperator(repairWeight, new_solution, removedCities, distmat)
        new_distance = disCal(new_solution, distmat)

        if acceptNewSolution(current_distance, new_distance, temperature):
            current_solution = new_solution
            current_distance = new_distance
            updateWeights(destroyWeight, destroyScores, destroyIndex, 1.5)  # 假设破坏算子得分为1.5
            updateWeights(repairWeight, repairScores, repairIndex, 1.5)  # 假设修复算子得分为1.5
        else:
            updateWeights(destroyWeight, destroyScores, destroyIndex, -0.5)  # 假设破坏算子得分为-0.5
            updateWeights(repairWeight, repairScores, repairIndex, -0.5)  # 假设修复算子得分为-0.5

        # 降低温度
        temperature *= cooling_rate

    return current_solution,current_distance

    # print("最佳解:", current_solution)
    # print("最佳距离:", current_distance)


class ALNSOptimizer:
    def __init__(self,dist_matrix):
        self.dis=dist_matrix


    def run(self,iterations):
        best_route, best_cost =main(self.dis,iterations)
        return best_route, best_cost







