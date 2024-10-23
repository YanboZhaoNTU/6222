import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from  sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from umap import UMAP
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import fetch_openml
import time
##############马氏距离计算函数
########################################### 数据集
#mnist = fetch_openml('Fashion-MNIST')
#mnist = fetch_openml('mnist_784')

#X = mnist.data / 255.0
#y = mnist.target.astype(int)
iris = load_iris()
X = iris.data
y = iris.target
n = 2
def deal_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    start_time = time.time()
    line(X_train, y_train, X_test, y_test, "without")
    end_time = time.time()  # 记录结束时间
    runtime = end_time - start_time  # 计算运行时间
    print("w l "+f"运行时间: {runtime:.6f} 秒")

    start_time = time.time()
    KNN(X_train, y_train, X_test, y_test, "without")
    end_time = time.time()  # 记录结束时间
    runtime = end_time - start_time  # 计算运行时间
    print("w k "+f"运行时间: {runtime:.6f} 秒")

    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    start_time = time.time()
    line(X_train, y_train, X_test, y_test,"PCA")
    end_time = time.time()  # 记录结束时间
    runtime = end_time - start_time  # 计算运行时间
    print("pca l "+f"运行时间: {runtime:.6f} 秒")

    start_time = time.time()
    KNN(X_train, y_train, X_test, y_test,"PCA")
    end_time = time.time()  # 记录结束时间
    runtime = end_time - start_time  # 计算运行时间
    print("pca k "+f"运行时间: {runtime:.6f} 秒")
    print("PCA finsih")



    print("tsne begin")
    tsne = TSNE(n_components=n, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_tsne, y, test_size=0.2, random_state=42)

    start_time = time.time()
    line(X_train, y_train, X_test, y_test,"TSNE")
    end_time = time.time()  # 记录结束时间
    runtime = end_time - start_time  # 计算运行时间
    print("tsne l "+f"运行时间: {runtime:.6f} 秒")

    start_time = time.time()
    KNN(X_train, y_train, X_test, y_test,"TSNE")
    end_time = time.time()  # 记录结束时间
    runtime = end_time - start_time  # 计算运行时间
    print("tsne k "+f"运行时间: {runtime:.6f} 秒")
    print("tsne finish")

    print("umap begin")
    umap = UMAP(n_components=n, n_jobs=-1)
    X_umap = umap.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_umap, y, test_size=0.2, random_state=42)

    start_time = time.time()
    line(X_train, y_train, X_test, y_test,"UMAP")
    end_time = time.time()  # 记录结束时间
    runtime = end_time - start_time  # 计算运行时间
    print("umap l "+f"运行时间: {runtime:.6f} 秒")

    start_time = time.time()
    KNN(X_train, y_train, X_test, y_test,"UMAP")
    end_time = time.time()  # 记录结束时间
    runtime = end_time - start_time  # 计算运行时间
    print("umap k "+f"运行时间: {runtime:.6f} 秒")
    print("umap finish")

    print("LDA begin")
    lda = LatentDirichletAllocation(n_components=n)
    X_lda = lda.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)

    start_time = time.time()
    line(X_train, y_train, X_test, y_test,"LDA")
    end_time = time.time()  # 记录结束时间
    runtime = end_time - start_time  # 计算运行时间
    print("lda l "+f"运行时间: {runtime:.6f} 秒")

    start_time = time.time()
    KNN(X_train, y_train, X_test, y_test,"LDA")
    end_time = time.time()  # 记录结束时间
    runtime = end_time - start_time  # 计算运行时间
    print("lda k "+f"运行时间: {runtime:.6f} 秒")
    print("LDA finsih")



#    pca = PCA(n_components=2)
#    X_pca = pca.fit_transform(X)
#    lda = LatentDirichletAllocation(n_components=2)
#    X_lda = lda.fit_transform(X)
#    tsne = TSNE(n_components=2, n_jobs=-1)
 #   X_tsne = tsne.fit_transform(X)
#    umap = UMAP(n_components=2, n_jobs=-1)
#    X_umap = umap.fit_transform(X)
#    draw(X,y,X_pca,X_lda,X_tsne,X_umap)

def KNN(X_train, y_train, X_test, y_test,name):
    target_names = [str(i) for i in range(10)]
    print(5)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    # 预测
    print(6)
    y_pred = knn.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(name+f"KNN 分类器的准确率: {accuracy:.4f}")

    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(name+'KNN  Confusion Matrix')
    plt.show()

    # map
    y_score = knn.predict_proba(X_test)
    # 将真实标签进行二值化，适用于多类别计算
    n_classes = 10
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2,3,4,5,6,7,8,9])

    average_precisions = []
    for i in range(n_classes):
        ap = average_precision_score(y_test_binarized[:, i], y_score[:, i])
        average_precisions.append(ap)

    mean_ap = np.mean(average_precisions)
    print(name+f"KNN Mean Average Precision (mAP): {mean_ap:.4f}")

    plt.figure(figsize=(10, 6))
    plt.bar(range(n_classes), average_precisions, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Average Precision')
    plt.xticks(range(n_classes),
               ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8',
                'Class 9'])
    plt.title('KNN Average Precision for Each Class')
    plt.ylim(0, 1)
    plt.show()

    # 计算 top-1 和 top-5 准确率
    top1_correct = 0
    top5_correct = 0
    y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
    for i in range(len(y_test)):
        top5_preds = np.argsort(y_score[i])[-5:][::-1]  # 获取前5个最高概率的类别
        if y_test[i] == top5_preds[0]:
            top1_correct += 1
        if y_test[i] in top5_preds:
            top5_correct += 1

    top1_accuracy = top1_correct / len(y_test)
    top5_accuracy = top5_correct / len(y_test)
    print(name+f"KNN Top-1 Accuracy: {top1_accuracy:.4f}")
    print(name+f"KNN Top-5 Accuracy: {top5_accuracy:.4f}")

    # 计算宏平均指标
    macro_precision = precision_score(y_test, y_pred, average='macro')
    macro_recall = recall_score(y_test, y_pred, average='macro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(name+f"KNN Macro Precision: {macro_precision:.4f}")
    print(name+f"KNN Macro Recall: {macro_recall:.4f}")
    print(name+f"KNN Macro F1 Score: {macro_f1:.4f}")

    # 使用 plt 展示宏平均指标
    macro_metrics = [macro_precision, macro_recall, macro_f1]
    metrics = ['Precision', 'Recall', 'F1 Score']
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, macro_metrics, color='lightblue')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('KNN Macro-Averaged Metrics')
    plt.ylim(0, 1)
    plt.show()

    micro_precision = precision_score(y_test, y_pred, average='micro')
    micro_recall = recall_score(y_test, y_pred, average='micro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    print(f"KNN Micro Precision: {micro_precision:.4f}")
    print(f"KNN Micro Recall: {micro_recall:.4f}")
    print(f"KNN Micro F1 Score: {micro_f1:.4f}")

    # 使用 plt 展示微平均指标
    metrics = ['Precision', 'Recall', 'F1 Score']
    micro_metrics = [micro_precision, micro_recall, micro_f1]
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, micro_metrics, color='lightgreen')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('KNN Micro-Averaged Metrics')
    plt.ylim(0, 1)
    plt.show()


    conf_matrix = confusion_matrix(y_test, y_pred)
def line(X_train, y_train, X_test, y_test,name):
    print(1)
    print(2)
    target_names = [str(i) for i in range(10)]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # 预测
    print(3)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(name+f"Linear Classifier的准确率: {accuracy:.4f}")
    print(4)
    ########################## 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(name+' PCA Confusion Matrix')
    plt.show()

    ###########################map
    y_score = model.predict_proba(X_test)
    # 将真实标签进行二值化，适用于多类别计算
    n_classes = 10
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    average_precisions = []
    for i in range(n_classes):
        ap = average_precision_score(y_test_binarized[:, i], y_score[:, i])
        average_precisions.append(ap)

    mean_ap = np.mean(average_precisions)
    print(name+f"line Mean Average Precision (mAP): {mean_ap:.4f}")

    plt.figure(figsize=(10, 6))
    plt.bar(range(n_classes), average_precisions, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Average Precision')
    plt.xticks(range(n_classes), ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9'])
    plt.title('line Average Precision for Each Class')
    plt.ylim(0, 1)
    plt.show()

    # 计算 top-1 和 top-5 准确率
    top1_correct = 0
    top5_correct = 0
    y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
    for i in range(len(y_test)):

        top5_preds = np.argsort(y_score[i])[-5:][::-1]  # 获取前5个最高概率的类别
        if y_test[i] == top5_preds[0]:
            top1_correct += 1
        if y_test[i] in top5_preds:
            top5_correct += 1

    top1_accuracy = top1_correct / len(y_test)
    top5_accuracy = top5_correct / len(y_test)
    print(name+f"line Top-1 Accuracy: {top1_accuracy:.4f}")
    print(name+f"line Top-5 Accuracy: {top5_accuracy:.4f}")

    # 计算宏平均指标
    macro_precision = precision_score(y_test, y_pred, average='macro')
    macro_recall = recall_score(y_test, y_pred, average='macro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(name+f"Macro Precision: {macro_precision:.4f}")
    print(name+f"Macro Recall: {macro_recall:.4f}")
    print(name+f"Macro F1 Score: {macro_f1:.4f}")

    # 使用 plt 展示宏平均指标
    macro_metrics = [macro_precision, macro_recall, macro_f1]
    metrics = ['Precision', 'Recall', 'F1 Score']
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, macro_metrics, color='lightblue')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('line Macro-Averaged Metrics')
    plt.ylim(0, 1)
    plt.show()

    # 计算微平均指标
    micro_precision = precision_score(y_test, y_pred, average='micro')
    micro_recall = recall_score(y_test, y_pred, average='micro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    print(name+f"line Micro Precision: {micro_precision:.4f}")
    print(name+f"line Micro Recall: {micro_recall:.4f}")
    print(name+f"line Micro F1 Score: {micro_f1:.4f}")

    # 使用 plt 展示微平均指标
    metrics = ['Precision', 'Recall', 'F1 Score']
    micro_metrics = [micro_precision, micro_recall, micro_f1]
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, micro_metrics, color='lightgreen')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('line Micro-Averaged Metrics')
    plt.ylim(0, 1)
    plt.show()

def draw(X,y,X_pca, X_lda, X_tsne, X_umap):
    X = np.array(X) if isinstance(X, (pd.DataFrame, pd.Series)) else X
    y = np.array(y) if isinstance(y, (pd.Series, pd.DataFrame)) else y

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    for i in range(10):
        plt.scatter(X[y == i, 0], X[y == i, 1], s=1, label=str(i))

    plt.title('without dimensionality reduction')
    plt.xlabel('WDR Component 1')
    plt.ylabel('WDR Component 2')
    plt.legend(title='Digit')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    for i in range(10):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], s=1, label=str(i))
    plt.title('PCA Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Digit')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    for i in range(10):
        plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], s=1, label=str(i))
    plt.title('LDA Visualization')
    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2')
    plt.legend(title='Digit')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 6))

    # plt.subplot(2,2,1)
    for i in range(10):
        plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], s=1, label=str(i))
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Digit')
    plt.grid(True)
    plt.show()
    # plt.subplot(2,2,2)
    for i in range(10):
        plt.scatter(X_umap[y == i, 0], X_umap[y == i, 1], s=1, label=str(i))
    plt.title('UMAP Visualization')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend(title='Digit')
    plt.grid(True)

    plt.show()

deal_data(X, y)



###################################### 画降维效果图
'''


'''
