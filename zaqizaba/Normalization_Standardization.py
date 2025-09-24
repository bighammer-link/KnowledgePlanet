# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
#
# # 设置随机种子以确保可重复性
# np.random.seed(42)
#
# # 生成二维数据：200个样本，两个特征
# n_samples = 200
# data = np.random.randn(n_samples, 2) * [2, 1] + [3, 5]  # 模拟正态分布，均值[3, 5]，标准差[2, 1]
#
# # 1. 减去均值
# mean = np.mean(data, axis=0)
# data_centered = data - mean
#
# # 2. 标准化
# scaler = StandardScaler()
# data_standardized = scaler.fit_transform(data)
#
# # 计算所有数据的x和y轴范围
# all_data = np.concatenate([data, data_centered, data_standardized], axis=0)
# x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
# y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
#
# # 添加一些边界缓冲区以美观
# x_margin = (x_max - x_min) * 0.1
# y_margin = (y_max - y_min) * 0.1
# x_min, x_max = x_min - x_margin, x_max + x_margin
# y_min, y_max = y_min - y_margin, y_max + y_margin
#
# # 绘制散点图
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
#
# # 原始数据分布
# ax1.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.5)
# ax1.set_title('Original Data')
# ax1.set_xlabel('Feature 1')
# ax1.set_ylabel('Feature 2')
# ax1.set_xlim(x_min, x_max)
# ax1.set_ylim(y_min, y_max)
# ax1.grid(True)
#
# # 减去均值后的分布
# ax2.scatter(data_centered[:, 0], data_centered[:, 1], c='green', alpha=0.5)
# ax2.set_title('Centered Data (Mean Subtracted)')
# ax2.set_xlabel('Feature 1')
# ax2.set_ylabel('Feature 2')
# ax2.set_xlim(x_min, x_max)
# ax2.set_ylim(y_min, y_max)
# ax2.grid(True)
#
# # 标准化后的分布
# ax3.scatter(data_standardized[:, 0], data_standardized[:, 1], c='red', alpha=0.5)
# ax3.set_title('Standardized Data')
# ax3.set_xlabel('Feature 1')
# ax3.set_ylabel('Feature 2')
# ax3.set_xlim(x_min, x_max)
# ax3.set_ylim(y_min, y_max)
# ax3.grid(True)
#
# # 调整布局并显示
# plt.tight_layout()
# plt.show()
#
# # 打印统计信息
# print("原始数据均值:", np.mean(data, axis=0))
# print("原始数据标准差:", np.std(data, axis=0))
# print("减去均值后均值:", np.mean(data_centered, axis=0))
# print("减去均值后标准差:", np.std(data_centered, axis=0))
# print("标准化后均值:", np.mean(data_standardized, axis=0))
# print("标准化后标准差:", np.std(data_standardized, axis=0))


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
#
# # 设置随机种子以确保可重复性
# np.random.seed(42)
#
# # 生成二维数据：200个样本，两个特征
# n_samples = 200
# data = np.random.randn(n_samples, 2) * [2, 1] + [3, 5]  # 模拟正态分布，均值[3, 5]，标准差[2, 1]
#
# # 归一化（Min-Max）
# scaler = MinMaxScaler()
# data_normalized = scaler.fit_transform(data)
#
# # 绘制散点图
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#
# # 原始数据分布
# ax1.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.5)
# ax1.set_title('Original Data')
# ax1.set_xlabel('Feature 1')
# ax1.set_ylabel('Feature 2')
# ax1.grid(True)
#
# # 归一化后的分布
# ax2.scatter(data_normalized[:, 0], data_normalized[:, 1], c='red', alpha=0.5)
# ax2.set_title('Normalized Data (Min-Max)')
# ax2.set_xlabel('Feature 1')
# ax2.set_ylabel('Feature 2')
# ax2.grid(True)
#
# # 调整布局并显示
# plt.tight_layout()
# plt.show()
#
# # 打印统计信息
# print("原始数据范围:", data.min(axis=0), data.max(axis=0))
# print("归一化后范围:", data_normalized.min(axis=0), data_normalized.max(axis=0))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成二维数据：200个样本，两个特征
n_samples = 200
data = np.random.randn(n_samples, 2) * [2, 1] + [3, 5]  # 模拟正态分布，均值[3, 5]，标准差[2, 1]

# 标准化
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# 计算统一的x、y轴范围
all_data = np.concatenate([data, data_standardized], axis=0)
x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
x_margin = (x_max - x_min) * 0.1
y_margin = (y_max - y_min) * 0.1
x_min, x_max = x_min - x_margin, x_max + x_margin
y_min, y_max = y_min - y_margin, y_max + y_margin

# 绘制散点图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 原始数据分布
ax1.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.5)
ax1.set_title('Original Data')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.grid(True)

# 标准化后的分布
ax2.scatter(data_standardized[:, 0], data_standardized[:, 1], c='red', alpha=0.5)
ax2.set_title('Standardized Data')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.grid(True)

plt.tight_layout()
plt.show()

# 绘制直方图以验证分布形状（仅对Feature 1）
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 5))

# 原始数据的直方图
ax3.hist(data[:, 0], bins=30, color='blue', alpha=0.5, density=True)
ax3.set_title('Original Data (Feature 1)')
ax3.set_xlabel('Value')
ax3.set_ylabel('Density')
ax3.grid(True)

# 标准化后的直方图
ax4.hist(data_standardized[:, 0], bins=30, color='red', alpha=0.5, density=True)
ax4.set_title('Standardized Data (Feature 1)')
ax4.set_xlabel('Value')
ax4.set_ylabel('Density')
ax4.grid(True)

plt.tight_layout()
plt.show()

# 打印统计信息
print("原始数据均值:", np.mean(data, axis=0))
print("原始数据标准差:", np.std(data, axis=0))
print("标准化后均值:", np.mean(data_standardized, axis=0))
print("标准化后标准差:", np.std(data_standardized, axis=0))