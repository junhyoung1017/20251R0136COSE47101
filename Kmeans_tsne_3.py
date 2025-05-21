import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cdist,pdist,squareform

##그래프 깨짐 방지지
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows의 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 마이너스(-) 깨짐 방지
######
data=pd.read_csv('C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/github/data/github_profiles_total_v2.csv')
X=data
df=data.drop(columns=["user_ID","username","repo_count"])
'''# Explained Variance Ratio 누적합으로 Truncated SVD의 최적 차원 수 결정
svd = TruncatedSVD(n_components=17, random_state=42)
svd.fit(df)
plt.plot(np.cumsum(svd.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Truncated SVD - Explained Variance")
plt.grid(True)
plt.show()'''
# 1. SVD로 차원 축소
svd = TruncatedSVD(n_components=8, random_state=42)
df_svd = svd.fit_transform(df)

# 2. 클러스터링 (SVD 결과로)
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(df_svd)

# 1. 클러스터 중심 (표준화 공간) → 역정규화
cluster_centers_original = svd.inverse_transform(kmeans.cluster_centers_)

# 2. 각 클러스터 중심에서 가장 큰 언어 (대표 언어)
df_columns = df.columns
top_languages = []
for center in cluster_centers_original:
    top_idx = np.argmax(center)
    top_lang = df_columns[top_idx]
    top_languages.append(top_lang)

# 2D t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(df_svd)

tsne_centers = TSNE(n_components=2, random_state=42, perplexity=3)
centers_tsne = tsne_centers.fit_transform(kmeans.cluster_centers_)
user_idx = 4  # 원하는 유저의 인덱스
user_cluster = clusters[user_idx]
print(f"인덱스 {user_idx}번 유저는 {user_cluster}번 클러스터에 속합니다.")
# 각 클러스터별로 중심에 가장 가까운 데이터 인덱스 찾기
closest_indices = []
for i, center in enumerate(kmeans.cluster_centers_):
    # 해당 클러스터에 속한 데이터의 인덱스
    cluster_indices = np.where(clusters == i)[0]
    # 해당 클러스터 데이터의 scaled 값
    cluster_points = df_svd[cluster_indices]
    # 중심과의 거리 계산
    distances = cdist([center], cluster_points)
    # 가장 가까운 데이터 인덱스
    min_idx = cluster_indices[np.argmin(distances)]
    closest_indices.append(min_idx)


plt.figure(figsize=(10, 7))
sc=plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)
# t-SNE 좌표에서 해당 위치에 텍스트 표시
for i, idx in enumerate(closest_indices):
    x, y = X_tsne[idx]
    plt.text(x, y, top_languages[i], fontsize=12, color='red', weight='bold', ha='center', va='center')
# 대표 언어 텍스트 표시
plt.title("2D t-SNE Projection of GitHub Users (Colored by Cluster)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(sc,label="Cluster")
plt.show()

# 3. 원본 데이터에서 언어 비율 출력
for i in range(kmeans.n_clusters):
    cluster_indices = np.where(clusters == i)[0]
    cluster_data = df.iloc[cluster_indices]  # 또는 rare_df.iloc[cluster_indices]
    mean_ratios = cluster_data.mean(axis=0)
    mean_ratios_percent = mean_ratios / mean_ratios.sum() * 100

    # TOP3 언어만 시각화
    top3_idx = np.argsort(mean_ratios_percent)[-3:][::-1]
    top3_langs = [df.columns[idx] for idx in top3_idx]
    top3_vals = [mean_ratios_percent.iloc[idx] for idx in top3_idx]

    plt.figure(figsize=(6, 4))
    plt.bar(top3_langs, top3_vals, color='skyblue')
    plt.title(f"Cluster {i} TOP3 언어 비율(%)")
    plt.ylabel("비율(%)")
    plt.xlabel("언어")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()


#evaluation
sum=0
for i in range(kmeans.n_clusters):
    cluster_points = X_tsne[clusters == i]
    center = cluster_points.mean(axis=0)
    spread = np.mean(np.linalg.norm(cluster_points - center, axis=1))
    sum=sum+spread
    print(f"Cluster {i}: t-SNE 내 평균 거리(응집도) = {spread:.2f}")

# 클러스터별 t-SNE 중심 좌표 계산
cluster_centers_tsne = []
for i in range(kmeans.n_clusters):
    cluster_points = X_tsne[clusters == i]
    center = cluster_points.mean(axis=0)
    cluster_centers_tsne.append(center)
cluster_centers_tsne = np.array(cluster_centers_tsne)

# 클러스터 중심 간 거리 행렬 계산
dist_matrix = squareform(pdist(cluster_centers_tsne))
# 자기 자신과의 거리는 0이므로, 0이 아닌 값만 추출
nonzero_dists = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

print(f"t-SNE 클러스터 내 평균 응집도: {sum/kmeans.n_clusters:.2f}")
print(f"클러스터 중심 간 평균 거리(분리도): {nonzero_dists.mean():.2f}")
print(f"클러스터 중심 간 최소 거리(최소 분리도): {nonzero_dists.min():.2f}")