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
#3D
data=pd.read_csv('C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/github/data/github_profiles_total_v2.csv')
X=data
df=data.drop(columns=["user_ID","username","repo_count"])
# df_scaled = StandardScaler().fit_transform(df) #각 행은 하나의 사용자, 각 열은 하나의 언어
# ['JavaScript', 'Python', 'Java', 'C', 'TypeScript', 'C++', 'C#'] 남기기
# 그 외 스택 예측에 도움될만한 언어
# rare한 언어들 선택
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
#28개 삭제
# CSV 파일로 저장
# df.to_csv('C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/github/data/github_profiles_total_v2_rare.csv', index=False)
# 문자열 rare_lang → 리스트화
df['rare_lang_list'] = df['rare_lang'].apply(lambda x: x.split(',') if x else [])
# One-hot 인코딩
mlb = MultiLabelBinarizer()
rare_matrix = mlb.fit_transform(df['rare_lang_list'])
print(rare_matrix)
# # 차원 축소
# svd = TruncatedSVD(n_components=3)
# rare_reduced = svd.fit_transform(rare_matrix)
# print(rare_reduced)
rare_df = pd.DataFrame(rare_matrix, columns=mlb.classes_, index=df.index)
# 수치형 데이터와 병합
df_final = pd.concat([df.drop(columns=['rare_lang', 'rare_lang_list']), rare_df], axis=1)
# df_final = pd.concat([
#     df.drop(columns=['rare_lang', 'rare_lang_list']),
#     pd.DataFrame(rare_reduced, columns=['rare_1', 'rare_2', 'rare_3'], index=df.index)
# ], axis=1)
# CSV 파일로 저장
# df_final.to_csv('C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/github/data/github_profiles_total_v2_test.csv', index=False)
# 1. 데이터 표준화
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_final)
# #2D 시각화
X_pca = PCA(n_components=2).fit_transform(df_scaled)
kmeans = KMeans(n_clusters=8, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

# 2D Visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=clusters, cmap='viridis', s=50, alpha=0.8)
ax.set_title("2D PCA Projection of GitHub Users (Colored by Cluster)", fontsize=14)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
plt.colorbar(scatter, label="Cluster")



# 1. 클러스터 중심 (표준화 공간) → 역정규화
# scaler = StandardScaler().fit(df_final)
cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)

# 2. 각 클러스터 중심에서 가장 큰 언어 (대표 언어)
df_columns = df_final.columns
top_languages = []
for center in cluster_centers_original:
    top_idx = np.argmax(center)
    top_lang = df_columns[top_idx]
    top_languages.append(top_lang)

# 3. 클러스터 중심을 PCA 공간으로 변환
cluster_centers_pca = PCA(n_components=2).fit(df_scaled).transform(kmeans.cluster_centers_)
# 4. 대표 언어 텍스트를 PCA 시각화에 표시
for i, (x, y) in enumerate(cluster_centers_pca):
    plt.text(x, y,top_languages[i], fontsize=12, color='red', weight='bold')
plt.tight_layout()
plt.show()
print(silhouette_score(df_scaled, clusters))



# 5. 클러스터링 평가
silhouette_scores = []
for k in range(2, 17):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, clusters)
    silhouette_scores.append(score)

plt.plot(range(2, 17), silhouette_scores, marker='o')
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette score")
plt.title("Silhouette vs K")
plt.grid(True)
plt.show()
inertia = []
K = range(1, 16)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('Elbow Method For Optimal k')
plt.xticks(K)
plt.grid(True)
plt.show()
# # 가장 증가율이 큰 구간 5->6 0.67 증가, 10->11 0.74 증가