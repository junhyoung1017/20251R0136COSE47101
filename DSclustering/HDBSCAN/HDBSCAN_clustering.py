# from google.colab import drive
# drive.mount('/content/drive')




# !pip install hdbscan
# !pip install umap-learn




import pandas as pd
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import glob




file_path = '/content/drive/MyDrive/Colab Notebooks/COSE471/test/results2/github_profiles_total_v2.csv'

# 2. CSV 로드 + 컬럼 이름 공백 제거
df = pd.read_csv(file_path, index_col=False).fillna(0)
df.columns = df.columns.str.strip()  # ← 여기에서 컬럼 이름 공백 제거

print(f"총 {len(df)}개의 사용자 데이터 로드 완료")
print(f"컬럼 이름: {list(df.columns)}")




# 2. repo_count 기준 필터링 (3번째 컬럼 기준으로 'repo_count'라고 명시)
filtered_df = df[df.iloc[:, 2] >= 10].copy()

# 3. 언어 비율 컬럼만 추출 (0: user_ID, 1: username, 2: repo_count → 3부터 끝까지)
language_columns = filtered_df.columns[3:]
X = filtered_df[language_columns].values

print(f"필터링된 사용자 수: {len(filtered_df)}")
print(f"사용된 feature 개수: {X.shape[1]}")




# 데이터셋 검증용(Hopkins 점수 계산)
from sklearn.neighbors import NearestNeighbors
import random

def hopkins(X, sampling_size=0.05):
    """
    Calculate the Hopkins statistic for the dataset X.

    Parameters:
        X: array-like, shape (n_samples, n_features)
        sampling_size: float or int, number or fraction of samples to use

    Returns:
        Hopkins statistic (float)
    """
    if isinstance(sampling_size, float):
        n_samples = int(sampling_size * X.shape[0])
    else:
        n_samples = min(sampling_size, X.shape[0])

    d = X.shape[1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)

    rand_X = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (n_samples, d))
    rand_indices = random.sample(range(0, X.shape[0]), n_samples)
    ujd = []
    wjd = []

    for j in range(n_samples):
        u_dist, _ = nbrs.kneighbors([rand_X[j]], 2, return_distance=True)
        ujd.append(u_dist[0][0])

        w_dist, _ = nbrs.kneighbors([X[rand_indices[j]]], 2, return_distance=True)
        wjd.append(w_dist[0][1])  # [1] because [0] is the point itself

    H = sum(ujd) / (sum(ujd) + sum(wjd))
    return H

hopkins_score = hopkins(X)
print(f"Hopkins Score: {hopkins_score:.3f}")





# 3. 차원 축소 (UMAP)
print("UMAP으로 차원 축소 중")
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_reduced = umap_reducer.fit_transform(X)





# 4. Dunn Index 계산 함수
def dunn_index(data, labels):
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    if len(unique_labels) < 2:
        return 0
    inter_cluster_distances = []
    for i in unique_labels:
        for j in unique_labels:
            if i < j:
                cluster_i = data[labels == i]
                cluster_j = data[labels == j]
                inter_cluster_distances.append(np.min(pdist(np.vstack([cluster_i, cluster_j]))))
    intra_cluster_distances = []
    for i in unique_labels:
        cluster_i = data[labels == i]
        if len(cluster_i) > 1:
            intra_cluster_distances.append(np.max(pdist(cluster_i)))
        else:
            intra_cluster_distances.append(0)
    if np.max(intra_cluster_distances) == 0:
        return 0
    return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)

def visualize_clusters(X_reduced, labels):
    """
    2차원으로 차원 축소된 X_reduced와 클러스터 라벨을 이용해 시각화합니다.
    noise 포인트(-1)는 회색으로 표시됩니다.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])

    plt.figure(figsize=(10, 8))

    for label in unique_labels:
        mask = labels == label
        if label == -1:
            # noise
            plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                        c='gray', s=30, label='Noise', alpha=0.5)
        else:
            plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                        s=30, label=f'Cluster {label}', alpha=0.8)

    plt.title(f'HDBSCAN Clustering 결과 (클러스터 수: {n_clusters})')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





best_result = {
    'min_cluster_size': None,
    'min_samples': None,
    'labels': None,
    'silhouette': -1,
    'dunn': 0,
    'n_clusters': 0
}

print("HDBSCAN 자동 튜닝 중 (목표는 5~10개 클러스터)")
for min_cluster_size in range(20, 101, 10):  # 20, 30, ..., 100
    for min_samples in [5, 10, 15]:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(X_reduced)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters < 5 or n_clusters > 10:
            continue

        sil = silhouette_score(X_reduced, labels) if n_clusters >= 2 else -1
        dunn = dunn_index(X_reduced, labels)

        print(f"  min_cluster_size={min_cluster_size}, min_samples={min_samples} → clusters={n_clusters}, silhouette={sil:.4f}, dunn={dunn:.4f}")

        if sil > best_result['silhouette']:
            best_result.update({
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'labels': labels,
                'silhouette': sil,
                'dunn': dunn,
                'n_clusters': n_clusters
            })




# 결과 출력
if best_result['labels'] is not None:
    print("\n 최적 결과:")
    print(f"  min_cluster_size = {best_result['min_cluster_size']}")
    print(f"  min_samples = {best_result['min_samples']}")
    print(f"  클러스터 수 = {best_result['n_clusters']}")
    print(f"  Silhouette Score = {best_result['silhouette']:.4f}")
    print(f"  Dunn Index = {best_result['dunn']:.4f}")

    # 저장
    filtered_df['Cluster_Label'] = best_result['labels']
    filtered_df.to_csv('/content/drive/MyDrive/Colab Notebooks/COSE471/test/result_hdbscan_tuned.csv', index=False)

    # 시각화
    visualize_clusters(X_reduced, best_result['labels'])
else:
    print("\n 클러스터 수가 5~10인 결과를 찾지 못했습니다.")





import seaborn as sns
import matplotlib.pyplot as plt

# 1. 언어 컬럼만 선택
language_columns = filtered_df.columns[3:-1]  # Cluster_Label 제외

# 2. 클러스터별 평균 계산
cluster_avg = filtered_df.groupby('Cluster_Label')[language_columns].mean()

# 3. 시각화
plt.figure(figsize=(14, 6))
sns.heatmap(cluster_avg, cmap='YlGnBu', annot=True, fmt=".2f", linewidths=0.5)
plt.title("클러스터별 언어 평균 사용 비율 (Top 언어 분석용)")
plt.xlabel("Language")
plt.ylabel("Cluster")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()





