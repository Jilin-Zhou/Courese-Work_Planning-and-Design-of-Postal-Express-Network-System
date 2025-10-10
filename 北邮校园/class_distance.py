import numpy as np
import pandas as pd

# 从 CSV 文件读取数据
df = pd.read_csv('network.csv')

# # 计算距离矩阵
# def calculate_distance_matrix(df):
#     coords = df[['X', 'Y']].values
#     distance_matrix = np.sqrt(((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=2))
#     return distance_matrix
#



# 计算距离矩阵
def calculate_distance_matrix(df):
    coords = df[['X', 'Y']].values
    # 曼哈顿距离计算
    distance_matrix = np.abs(coords[:, np.newaxis, :] - coords[np.newaxis, :, :]).sum(axis=2)
    return distance_matrix

distance_matrix = calculate_distance_matrix(df)

# 将距离矩阵转换为 DataFrame
distance_matrix_df = pd.DataFrame(distance_matrix)

# 将距离矩阵保存到 CSV 文件
distance_matrix_df.to_csv('distance_matrix.csv', index=False)

print("OK")