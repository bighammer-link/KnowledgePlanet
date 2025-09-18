import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# 加载加州房价数据集
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseVal'] = california.target


# 分离特征变量和目标变量
X = data.drop('MedHouseVal', axis=1)
y = data['MedHouseVal']

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)
# 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# 真实值 vs 预测值
plt.figure(figsize=(10, 5))
plt.scatter(y_train, y_train_pred, color='blue', label='Train data')
plt.scatter(y_test, y_test_pred, color='red', label='Test data')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()
plt.title('True Values vs Predictions')
plt.show()

# 残差分布
plt.figure(figsize=(10, 5))
sns.histplot((y_train - y_train_pred), bins=50, kde=True, label='Train data', color='blue')
sns.histplot((y_test - y_test_pred), bins=50, kde=True, label='Test data', color='red')
plt.legend()
plt.title('Residuals Distribution')
plt.show()