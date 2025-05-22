#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def dunn_index(dist_matrix, labels):
    unique_labels = np.unique(labels)
    intra = []
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        if len(idx) > 1:
            intra.append(dist_matrix[np.ix_(idx, idx)].max())
    max_intra = np.max(intra) if intra else 0
    inter = []
    for i, lbl1 in enumerate(unique_labels):
        for lbl2 in unique_labels[i + 1:]:
            idx1 = np.where(labels == lbl1)[0]
            idx2 = np.where(labels == lbl2)[0]
            inter.append(dist_matrix[np.ix_(idx1, idx2)].min())
    min_inter = np.min(inter) if inter else 0
    return min_inter / max_intra if max_intra > 0 else np.nan


def main():
    # ─── 설정 ─────────────────────────────
    CSV_PATH = 'github_profiles_total_v2.csv'
    FILTER_REPOS = 10
    METRIC = 'euclidean'
    LINKAGE_METHOD = 'average'  # 'single' / 'complete' / 'average'
    K_LINE = 10  # 덴드로그램 및 Top3 계산 기준 클러스터 수
    # ───────────────────────────────────────

    # 1) 데이터 로드 및 전처리
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    df = df[df['repo_count'] >= FILTER_REPOS].reset_index(drop=True)
    X = df.drop(columns=['user_ID', 'username', 'repo_count']).values

    # 2) 거리 행렬 계산
    D_condensed = pdist(X, metric=METRIC)
    D_square = squareform(D_condensed)

    # 3) 덴드로그램 그리기
    Z = linkage(D_condensed, method=LINKAGE_METHOD)
    threshold = Z[-(K_LINE - 1), 2]
    plt.figure(figsize=(16, 8))
    dendrogram(
        Z,
        labels=[str(i) for i in range(len(X))],
        leaf_rotation=90,
        leaf_font_size=6,
        color_threshold=threshold
    )
    plt.axhline(y=threshold, color='k', linestyle='--', linewidth=1)
    plt.title(f'Dendrogram ({METRIC} distance, {LINKAGE_METHOD} linkage, k={K_LINE})')
    plt.xlabel('샘플 인덱스')
    plt.ylabel(f'{METRIC.capitalize()} distance')
    plt.tight_layout()
    plt.show()

    # 4) k=4~10 실루엣 & Dunn 평가
    results = []
    for k in range(4, 11):
        agg = AgglomerativeClustering(
            n_clusters=k,
            metric=METRIC,
            linkage=LINKAGE_METHOD
        )
        labels = agg.fit_predict(X)
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(X, labels, metric=METRIC)
            dunn = dunn_index(D_square, labels)
        else:
            sil, dunn = np.nan, np.nan
        results.append({'k': k, 'silhouette': sil, 'dunn': dunn})
    res_df = pd.DataFrame(results)
    print('\n클러스터 평가 결과 (k=4~10):')
    print(res_df.to_string(index=False, float_format='%.4f'))

    # 5) K_LINE 클러스터에 대한 Top3 언어 분석
    agg_line = AgglomerativeClustering(
        n_clusters=K_LINE,
        metric=METRIC,
        linkage=LINKAGE_METHOD
    )
    labels_line = agg_line.fit_predict(X)
    df['cluster'] = labels_line

    lang_cols = df.columns.difference(['user_ID', 'username', 'repo_count', 'cluster'])
    cluster_means = df.groupby('cluster')[lang_cols].mean()

    top3_list = []
    for lbl, row in cluster_means.iterrows():
        top3 = row.sort_values(ascending=False).head(3)
        top3_list.append({
            'cluster': lbl,
            'top1_lang': top3.index[0], 'top1_ratio': top3.iloc[0],
            'top2_lang': top3.index[1], 'top2_ratio': top3.iloc[1],
            'top3_lang': top3.index[2], 'top3_ratio': top3.iloc[2],
        })
    top3_df = pd.DataFrame(top3_list).sort_values('cluster')
    print(f'\nCluster Top 3 Languages (k={K_LINE}):')
    print(top3_df.to_string(index=False, float_format='%.4f'))

    # 6) Top3 언어 사용 비율 시각화
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    axes = axes.flatten()
    for i in range(K_LINE):
        ax = axes[i]
        row = cluster_means.loc[i]
        top3 = row.sort_values(ascending=False).head(3)
        x_pos = np.arange(len(top3))
        ax.bar(x_pos, top3.values, width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(top3.index, rotation=45, ha='right')
        ax.set_ylim(0, top3.values.max() * 1.1)
        ax.set_title(f'Cluster {i}')
        ax.set_ylabel('Usage Ratio')
    # 빈 플롯 숨기기
    for j in range(K_LINE, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
