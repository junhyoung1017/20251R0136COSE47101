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
from IPython.display import display

data=pd.read_csv('C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/github/data/github_profiles_total_v2_re.csv')
X=data
df=data.drop(columns=["user_ID","username","repo_count"])

svd = TruncatedSVD(n_components=min(df.shape), random_state=42)
svd.fit(df)

explained = np.cumsum(svd.explained_variance_ratio_)

# 시각화
plt.plot(explained, marker='o')
plt.xlabel("Number of SVD Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("TruncatedSVD - Cumulative Variance Explained")
plt.grid(True)
plt.show()

# 출력 예시
for i, v in enumerate(explained[:12]):
    print(f"Component {i+1}: {v:.4f}")

# 컴포넌트 불러오기
loadings = pd.DataFrame(
    svd.components_.T,  # shape: (n_features, n_components)
    index=df.columns,   # 원래 피처명
    columns=[f"Component_{i+1}" for i in range(svd.n_components)]
)

# 절댓값 기준으로 상위 기여 피처 출력
for i in range(12):  # 예: 앞 5개 성분만 분석
    print(f"\n📌 Component {i+1}의 주요 기여 피처:")
    display(loadings.iloc[:, i].abs().sort_values(ascending=False).head(5))