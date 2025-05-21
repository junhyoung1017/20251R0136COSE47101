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
##
data=pd.read_csv('C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/github/data/github_profiles_total_v2.csv')
X=data
df=data.drop(columns=["user_ID","username","repo_count"])

rare_langs = ['Go','Rust', 'Kotlin', 'Swift', 'Dart', 'Scala', 'MATLAB', 'Assembly']

# 사용된 rare 언어들을 문자열로 병합
df['rare_lang'] = df[rare_langs].apply(
    lambda row: ','.join(lang for lang in rare_langs if row[lang] > 0),
    axis=1
)
# 개별 rare 언어 column은 제거
df.drop(columns=rare_langs, inplace=True)
#모든 언어 값이 0인 행 제거 (rare_lang 제외)
df_numeric = df.drop(columns=['rare_lang'])  # 수치형 열만
df = df[df_numeric.sum(axis=1) > 0]          # 총합이 0인 행 제거

df['rare_lang_list'] = df['rare_lang'].apply(lambda x: x.split(',') if x else [])
# One-hot 인코딩
mlb = MultiLabelBinarizer()
rare_matrix = mlb.fit_transform(df['rare_lang_list'])
rare_df = pd.DataFrame(rare_matrix, columns=mlb.classes_, index=df.index)
# 수치형 데이터와 병합
df_final = pd.concat([df.drop(columns=['rare_lang', 'rare_lang_list']), rare_df], axis=1)
# df_final.to_csv('C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/github/data/github_profiles_total_v2_test.csv', index=False)
# 1. SVD로 차원 축소
svd = TruncatedSVD(n_components=8, random_state=42)

df_svd = svd.fit_transform(df_final)
# #2D 시각화
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(df_svd)

# 1. 클러스터 중심 (표준화 공간) → 역정규화

cluster_centers_original = svd.inverse_transform(kmeans.cluster_centers_)

# 2. 각 클러스터 중심에서 가장 큰 언어 (대표 언어)
df_columns = df_final.columns
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
n_clusters = kmeans.n_clusters
n_cols = 5
n_rows = int(np.ceil(n_clusters / n_cols))

plt.figure(figsize=(n_cols * 4, n_rows * 4))
for i in range(kmeans.n_clusters):
    cluster_indices = np.where(clusters == i)[0]
    # rare 언어까지 포함한 데이터 사용
    cluster_data = df_final.iloc[cluster_indices]
    # # 수치형 컬럼만 선택
    # cluster_data = df.drop(columns=['rare_lang', 'rare_lang_list']).iloc[cluster_indices]
    mean_ratios = cluster_data.mean(axis=0)
    mean_ratios_percent = mean_ratios / mean_ratios.sum() * 100

    # TOP3 언어만 시각화
    top3_idx = np.argsort(mean_ratios_percent)[-3:][::-1]
    top3_langs = [df_final.columns[idx] for idx in top3_idx]
    top3_vals = [mean_ratios_percent.iloc[idx] for idx in top3_idx]

    plt.subplot(n_rows, n_cols, i + 1)
    plt.bar(top3_langs, top3_vals, color='skyblue')
    plt.title(f"Cluster {i} TOP3 언어 비율(%)")
    plt.ylabel("비율(%)")
    plt.xlabel("언어")
    plt.ylim(0, 100)
    plt.tight_layout()
#여러 그래프 한 화면에 출력
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