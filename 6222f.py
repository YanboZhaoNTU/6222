import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
import seaborn as sns


from umap import UMAP

# 数据集
mnist = fetch_openml('Fashion-MNIST')
X = mnist.data / 255.0
y = mnist.target.astype(int)

# 降维处理
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

tsne = TSNE(n_components=2,n_jobs=-1)
X_tsne = tsne.fit_transform(X)

umap = UMAP(n_components=2,n_jobs=-1)
X_umap = umap.fit_transform(X)
target_names = [str(i) for i in range(10)]


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 使用最近邻分类器
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN分类器的准确率: {accuracy:.4f}")
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('PCA Confusion Matrix')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X_tsne, y, test_size=0.2, random_state=42)

# 使用最近邻分类器
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN分类器的准确率: {accuracy:.4f}")
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Tsne Confusion Matrix')
plt.show()

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_umap, y, test_size=0.2, random_state=42)

# 使用最近邻分类器
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN分类器的准确率: {accuracy:.4f}")

# 画降维效果图
plt.figure(figsize=(18,6))

plt.subplot(1,3,1)
for i in range(10):

    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], s=1, label=str(i))
plt.title('PCA Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Digit')
plt.grid(True)

plt.subplot(1,3,2)
for i in range(10):

    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], s=1, label=str(i))
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Digit')
plt.grid(True)

plt.subplot(1,3,3)
for i in range(10):

    plt.scatter(X_umap[y == i, 0], X_umap[y == i, 1], s=1, label=str(i))
plt.title('UMAP Visualization')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.legend(title='Digit')
plt.grid(True)

plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('umap Confusion Matrix')
plt.show()