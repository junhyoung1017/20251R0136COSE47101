#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DBSCAN 클러스터링 자동 eps 결정 및 평가 스크립트
- repo_count ≥ 10 데이터 필터링
- 숫자형 피처 표준화
- UMAP(5차원) 임베딩
- k-distance plot을 통한 eps 자동 결정 (k=min_samples)
- DBSCAN 클러스터링
- Silhouette Coefficient, Dunn Index 계산
- 결과 CSV 저장 및 시각화
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from umap import UMAP
import matplotlib.pyplot as plt
import warnings

# 1) 엘보(elbow) 검출 함수
def find_elbow(distances):
    """내림차순 정렬된 거리 배열에서 엘보 지점에 해당하는 거리 값을 반환"""
    n = len(distances)
    idx = np.arange(n)
    # 선분 양 끝점
    x1, y1 = 0, distances[0]
    x2, y2 = n - 1, distances[-1]
    # 각 점에서 선분까지 거리
    num = np.abs((y2 - y1) * idx - (x2 - x1) * distances + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    dists = num / den
    elbow_idx = np.argmax(dists)
    return distances[elbow_idx]

# 2) Dunn Index 계산 함수
def dunn_index(X, labels):
    clusters = [c for c in np.unique(labels) if c != -1]
    if len(clusters) < 2:
        return np.nan
    # intra-cluster 최대 거리
    intra_dists = []
    for c in clusters:
        pts = X[labels == c]
        if pts.shape[0] > 1:
            intra_dists.append(np.max(cdist(pts, pts)))
    max_intra = np.max(intra_dists) if intra_dists else 0
    # inter-cluster 최소 거리
    inter_dists = []
    for i, ci in enumerate(clusters):
        for cj in clusters[i+1:]:
            pts_i = X[labels == ci]
            pts_j = X[labels == cj]
            inter_dists.append(np.min(cdist(pts_i, pts_j)))
    min_inter = np.min(inter_dists) if inter_dists else 0
    return min_inter / max_intra if max_intra > 0 else np.nan


def main():
    warnings.filterwarnings('ignore')
    # 데이터 로드 및 repo_count ≥ 10 필터링
    df = pd.read_csv('./github_profiles_total_v2.csv')
    df.columns = df.columns.str.strip()
    df = df[df['repo_count'] >= 10].reset_index(drop=True)

    # 숫자형 피처만 선택 후 표준화
    numeric = df.select_dtypes(include=[np.number]).drop(columns=['user_ID'], errors='ignore')
    X = StandardScaler().fit_transform(numeric.values)

    # UMAP 5차원 임베딩
    umap_model = UMAP(n_components=5, random_state=42)
    X_emb = umap_model.fit_transform(X)

    # k-distance 계산 (k = min_samples)
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_emb)
    distances, _ = nbrs.kneighbors(X_emb)
    k_dist = np.sort(distances[:, k-1])[::-1]

    # eps 자동 결정
    eps = find_elbow(k_dist)
    print(f"Estimated eps (k={k}): {eps:.4f}")

    # k-distance plot 시각화
    plt.figure(figsize=(6,4))
    plt.plot(k_dist, linewidth=2)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-NN distance')
    plt.title(f'k-distance plot (k={k})')
    plt.axhline(eps, color='red', linestyle='--', label=f'eps={eps:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # DBSCAN 클러스터링
    db = DBSCAN(eps=eps, min_samples=k)
    labels = db.fit_predict(X_emb)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    # Silhouette Coefficient
    if n_clusters >= 2:
        sil = silhouette_score(X_emb, labels)
    else:
        sil = np.nan
    # Dunn Index
    dunn = dunn_index(X_emb, labels)

    # 결과 저장
    df['cluster'] = labels
    os.makedirs('./result', exist_ok=True)
    out_path = './result/github_profiles_clustered.csv'
    df.to_csv(out_path, index=False)

    # 클러스터링 결과 시각화 (2차원으로 표시)
    plt.figure(figsize=(6,5))
    cmap = plt.get_cmap('Spectral')
    unique_labels = sorted(set(labels))
    colors = cmap(np.linspace(0,1,len(unique_labels)))
    for lbl, col in zip(unique_labels, colors):
        mask = (labels == lbl)
        pts = X_emb[mask]
        if lbl == -1:
            plt.scatter(pts[:,0], pts[:,1], marker='x', c=[col], label='Noise', s=50)
        else:
            plt.scatter(pts[:,0], pts[:,1], c=[col], label=f'Cluster {lbl}', s=50)
    plt.title(f'DBSCAN (eps={eps:.4f}, min_samples={k}) on 5D UMAP')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 지표 출력
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise}")
    print(f"Silhouette Coefficient: {sil:.4f}")
    print(f"Dunn Index: {dunn:.4f}")
    print(f"Results saved to: {out_path}")

if __name__ == '__main__':
    main()
