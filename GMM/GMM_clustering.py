# from google.colab import drive
# drive.mount('/content/drive')

import numpy as np
import pandas as pd
import glob
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 데이터 로드
# 경로에 있는 모든 github_profiles*.csv 파일을 찾음
file_path = '/content/drive/MyDrive/Colab Notebooks/COSE471/test/results2/github_profiles_total_v2.csv'

# 2. CSV 로드 + 컬럼 이름 공백 제거
df = pd.read_csv(file_path, index_col=False).fillna(0)
df.columns = df.columns.str.strip()  # ← 여기에서 컬럼 이름 공백 제거

print(f"총 {len(df)}개의 사용자 데이터 로드 완료")
print(f"컬럼 이름: {list(df.columns)}")





# repo_count 기준으로 df 자체를 필터링
df = df[df.iloc[:, 2] >= 6].reset_index(drop=True)

# 프로그래밍 언어 데이터 추출
language_columns = df.columns[2:]
X = df[language_columns].values




'''Solution: Use UMAP by reducing to 5 dimensions'''
import umap
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

# UMAP으로 5차원 축소
reducer = umap.UMAP(n_components=5, random_state=42)
X_umap = reducer.fit_transform(X)




# 2. GMM 학습
gmm = GaussianMixture(n_components=4, random_state=42)
labels = gmm.fit_predict(X_umap)

# 4. Silhouette Score
sil_score = silhouette_score(X_umap, labels)
print("Silhouette score with UMAP:", sil_score)
# Dunn Index 함수 정의
def dunn_index(X, labels):
    n_clusters = len(np.unique(labels))
    distances = pairwise_distances(X)

    inter_cluster_distances = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i = X[labels == i]
            cluster_j = X[labels == j]
            inter_dist = np.min(pairwise_distances(cluster_i, cluster_j))
            inter_cluster_distances.append(inter_dist)

    intra_cluster_distances = []
    for i in range(n_clusters):
        cluster_i = X[labels == i]
        intra_dist = np.max(pairwise_distances(cluster_i))
        intra_cluster_distances.append(intra_dist)

    min_inter = np.min(inter_cluster_distances)
    max_intra = np.max(intra_cluster_distances)

    return min_inter / max_intra

# Dunn Index 계산
dunn = dunn_index(X_umap, labels)
print("Dunn Index with UMAP:", dunn)




probs = gmm.predict_proba(X_umap)
print(probs[0])




df['cluster'] = labels
# CSV로 저장
df.to_csv('/content/drive/MyDrive/Colab Notebooks/COSE471/test/final_profiles_with_clusters.csv', index=False)





language_columns = df.columns[3:-1]  # '유저', 'ID', ..., 'cluster' 제외

# 클러스터별 평균 언어 비율 계산
cluster_profiles = df.groupby('cluster')[language_columns].mean()

# 시각화: 하나의 그림에 2x2 서브플롯으로 표시
fig, axes = plt.subplots(2, 2, figsize=(15, 12)) # 그림 크기 조절
axes = axes.flatten() # 2x2 배열을 1차원 배열로 만듬 (인덱싱을 쉽게 하기 위함)

for i, cluster_id in enumerate(cluster_profiles.index):
    ax = axes[i]
    cluster_profiles.loc[cluster_id].plot(kind='bar', ax=ax)
    ax.set_title(f"Cluster {cluster_id} rate by languages")
    ax.set_ylabel("rates")
    ax.set_xlabel("PL")
    ax.tick_params(axis='x', rotation=45) # x축 레이블 회전
    ax.grid(axis='y')

plt.tight_layout() # 서브플롯 간의 간격 자동 조절
plt.show()





# 클러스터별로 가장 높은 확률을 가진 인덱스 찾기
probs = gmm.predict_proba(X_umap)  # 또는 GMM에 사용한 입력 (X_scaled 등)
# 몇 명 뽑을지 설정
top_n = 5

# 클러스터별 상위 top_n 사용자 ID 출력
for cluster_id in range(gmm.n_components):
    # 해당 클러스터에 대한 소속 확률 벡터
    cluster_probs = probs[:, cluster_id]

    # 확률 내림차순으로 인덱스 정렬
    top_indices = np.argsort(cluster_probs)[::-1][:top_n]

    print(f"\n🔹 Cluster {cluster_id} 상위 {top_n} 대표자:")
    for rank, idx in enumerate(top_indices, 1):
        username = df.iloc[idx]['username']
        prob = cluster_probs[idx]
        print(f"  {rank}. ID: {username} (확률: {prob:.4f})")
