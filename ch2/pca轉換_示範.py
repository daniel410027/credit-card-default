import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import pandas as pd

# 創建一個二維數據集
np.random.seed(0)
n = 100
data = np.random.randn(n, 28)  # 100個樣本，每個樣本有兩個特徵
# 初始化PCA模型，指定要降到的維度
pca = PCA(n_components=28)

# 對數據進行PCA轉換
transformed_data = pca.fit_transform(data)
# 咒術反轉!歐維度一樣資料就不會遺失，逆轉換要增維就用這條
# data_original = pca.inverse_transform(transformed_data)

# .T是轉置矩陣，正交還啥的，實作起來怪怪的
data_reduced = np.dot(transformed_data - pca.mean_, pca.components_.T)
data_original = np.dot(transformed_data, pca.components_) + pca.mean_

print(pca.components_)
print(pca.mean_)

# 可視化原始數據和轉換後的數據
plt.figure(figsize=(10, 5))

# 原始數據
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1])
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 轉換後的數據
plt.subplot(1, 2, 2)
plt.scatter(transformed_data, np.zeros_like(transformed_data))
plt.title('Transformed Data')
plt.xlabel('Principal Component 1')
plt.gca().axes.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()

count_ones = np.sum(data[:, 1] == 1)
count_zeros = np.sum(data[:, 1] == 0)
