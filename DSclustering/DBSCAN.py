import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors


# Dunn Index 계산 함수
def dunn_index(data, labels):
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]  # Noise 제외
    if len(unique_labels) < 2:
        return 0  # 클러스터가 2개 미만이면 Dunn Index 계산 불가

    # 클러스터 간 거리 계산
    inter_cluster_distances = []
    for i in unique_labels:
        for j in unique_labels:
            if i != j:
                cluster_i = data[labels == i]
                cluster_j = data[labels == j]
                if len(cluster_i) > 0 and len(cluster_j) > 0:
                    # vstack을 사용한 방식이 메모리 문제를 일으킬 수 있으므로 대체 방식 사용
                    min_dist = float('inf')
                    for point_i in cluster_i:
                        for point_j in cluster_j:
                            dist = np.linalg.norm(point_i - point_j)
                            min_dist = min(min_dist, dist)
                    inter_cluster_distances.append(min_dist)

    if not inter_cluster_distances:
        return 0

    # 클러스터 내 거리 계산
    intra_cluster_distances = []
    for i in unique_labels:
        cluster_i = data[labels == i]
        if len(cluster_i) > 1:  # 클러스터에 점이 2개 이상인 경우만 계산
            max_dist = 0
            for idx1, point1 in enumerate(cluster_i):
                for idx2 in range(idx1 + 1, len(cluster_i)):
                    dist = np.linalg.norm(point1 - cluster_i[idx2])
                    max_dist = max(max_dist, dist)
            intra_cluster_distances.append(max_dist)

    if not intra_cluster_distances:
        return 0

    return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)


# 최적의 epsilon 값을 찾는 함수
def find_optimal_eps(data, min_samples=5):
    # k-거리 그래프를 위한 최근접 이웃 계산
    neigh = NearestNeighbors(n_neighbors=min_samples)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # min_samples번째 최근접 이웃까지의 거리를 오름차순 정렬
    distances = np.sort(distances[:, -1])
    
    # 'elbow point' 찾기 시도
    try:
        kneedle = KneeLocator(np.arange(len(distances)), distances, 
                              curve='convex', direction='increasing')
        eps = distances[kneedle.elbow] if kneedle.elbow else 0.5
    except:
        # 기본값으로 fallback
        eps = 0.3
    
    # 결과 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(distances)
    if kneedle.elbow:
        plt.axvline(x=kneedle.elbow, color='r', linestyle='--', label=f'Elbow at eps={eps:.3f}')
    plt.title('K-distance Graph')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'Distance to {min_samples}th nearest neighbor')
    plt.legend()
    plt.savefig('k_distance_graph.png')
    plt.close()
    
    return eps


# DBSCAN 클러스터링 함수
def perform_dbscan_clustering(file_path, output_path, eps=None, min_samples=10, 
                             normalize=True, metric='euclidean'):
    # CSV 파일 로드
    df = pd.read_csv(file_path)
    
    # 클러스터링에 사용할 데이터 추출 (프로그래밍 언어 비율만 사용)
    data = df.iloc[:, 3:].values  # 3번째 열부터 끝까지 사용
    
    # 결측치 처리
    data = np.nan_to_num(data)
    
    # 데이터 정규화
    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    
    # eps 자동 계산 (사용자가 지정하지 않은 경우)
    if eps is None:
        eps = find_optimal_eps(data, min_samples)
        print(f"자동 계산된 eps 값: {eps}")
    
    # DBSCAN 실행
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = dbscan.fit_predict(data)
    
    # 클러스터링 성능 평가
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_points = list(labels).count(-1)
    
    print(f"클러스터 수: {n_clusters}")
    print(f"노이즈 포인트 수: {noise_points}")
    
    # 클러스터링 평가 지표 계산
    silhouette = silhouette_score(data, labels) if len(set(labels)) > 1 and -1 not in labels else -1
    dunn = dunn_index(data, labels)
    
    # 클러스터링 결과를 데이터프레임에 추가
    df['Cluster_Label'] = labels
    
    # 결과 저장
    df.to_csv(output_path, index=False)
    
    # 결과 시각화
    visualize_clusters(data, labels)
    
    return labels, silhouette, dunn


# 시각화 함수 수정
def visualize_clusters(data, labels):
    # PCA로 차원 축소
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    
    # 설명된 분산 비율
    explained_variance = pca.explained_variance_ratio_
    
    # 시각화
    plt.figure(figsize=(12, 10))
    
    # 클러스터 수 확인
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    
    # 적절한 컬러맵 선택
    cmap = plt.colormaps['tab20']    
    
    # 클러스터별 산점도 - 노이즈와 클러스터 분리해서 처리
    for i, label in enumerate(unique_labels):
        mask = labels == label
        
        if label == -1:
            # 노이즈 포인트 (x 마커)는 edgecolors 없이 표시
            plt.scatter(
                reduced_data[mask, 0], 
                reduced_data[mask, 1],
                s=50, 
                color='black',  # 노이즈는 검정색
                marker='x',
                alpha=0.7,
                linewidth=0.5,
                label=f"Noise ({np.sum(mask)})"
            )
        else:
            # 클러스터 포인트는 원형 마커와 흰색 테두리 표시
            plt.scatter(
                reduced_data[mask, 0], 
                reduced_data[mask, 1],
                s=50, 
                color=cmap(i % cmap.N),
                marker='o',
                alpha=0.7,
                edgecolors='w',
                linewidth=0.5,
                label=f"Cluster {label} ({np.sum(mask)})"
            )
    
    plt.title(f"DBSCAN Clustering Result: {n_clusters} clusters identified", fontsize=14)
    plt.xlabel(f"PC1 ({explained_variance[0]:.2f}% variance)", fontsize=12)
    plt.ylabel(f"PC2 ({explained_variance[1]:.2f}% variance)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.legend(loc='best', fontsize=10)
    plt.savefig('dbscan_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()


# 다양한 파라미터 시도 함수
def try_different_parameters(file_path, output_prefix):
    # CSV 파일 로드
    df = pd.read_csv(file_path)
    
    # 클러스터링에 사용할 데이터 추출
    data = df.iloc[:, 3:].values
    
    # 결측치 처리 및 정규화
    data = np.nan_to_num(data)
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    
    # 자동 eps 계산
    optimal_eps = find_optimal_eps(normalized_data, min_samples=10)
    print(f"계산된 최적 eps: {optimal_eps}")
    
    # 다양한 파라미터 조합 시도
    eps_values = [optimal_eps * 0.5, optimal_eps, optimal_eps * 1.5]
    min_samples_values = [5, 10, 15]
    
    results = []
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            print(f"\nDBSCAN 실행 (eps={eps:.3f}, min_samples={min_samples})")
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(normalized_data)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_points = list(labels).count(-1)
            
            # 평가 지표 계산
            if n_clusters > 1:
                # 노이즈 포인트 제외하고 실루엣 계산
                non_noise_idx = labels != -1
                if sum(non_noise_idx) > 1 and len(set(labels[non_noise_idx])) > 1:
                    silhouette = silhouette_score(normalized_data[non_noise_idx], labels[non_noise_idx])
                else:
                    silhouette = -1
            else:
                silhouette = -1
                
            dunn = dunn_index(normalized_data, labels)
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'noise_points': noise_points,
                'silhouette': silhouette,
                'dunn_index': dunn
            })
            
            # 출력 파일명
            output_path = f"{output_prefix}_eps{eps:.3f}_ms{min_samples}.csv"
            
            # 결과 저장
            df_result = df.copy()
            df_result['Cluster_Label'] = labels
            df_result.to_csv(output_path, index=False)
            
            # 시각화
            plt.figure(figsize=(10, 7))
            plt.title(f"DBSCAN Clustering (eps={eps:.3f}, min_samples={min_samples})\n"
                      f"Clusters: {n_clusters}, Noise: {noise_points}, "
                      f"Silhouette: {silhouette:.3f}, Dunn: {dunn:.3f}")
            
            # 결과 시각화
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(normalized_data)
            
            for label in set(labels):
                if label == -1:
                    color = 'black'
                    marker = 'x'
                    label_name = 'Noise'
                else:
                    color = plt.cm.tab10(label % 10)
                    marker = 'o'
                    label_name = f'Cluster {label}'
                
                plt.scatter(
                    reduced_data[labels == label, 0],
                    reduced_data[labels == label, 1],
                    marker=marker,
                    color=color,
                    alpha=0.7,
                    label=f"{label_name} ({sum(labels == label)})"
                )
            
            plt.legend()
            plt.savefig(f"dbscan_eps{eps:.3f}_ms{min_samples}.png")
            plt.close()
    
    # 결과 정리
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('silhouette', ascending=False)
    results_df.to_csv("dbscan_parameter_comparison.csv", index=False)
    print("\n최적 파라미터 추천:")
    print(results_df.head())
    
    return results_df


# 사용 예시
if __name__ == "__main__":
    # CSV 파일 경로
    file_path = "./github_profiles_v2_1000-2044.csv"
    output_path = "clustered_output.csv"
    
    # 다양한 파라미터 시도
    print("다양한 DBSCAN 파라미터 시도 중...")
    results = try_different_parameters(file_path, "clustered_output")
    
    # 최적 파라미터로 최종 클러스터링 실행
    best_params = results.iloc[0]
    print(f"\n최적 파라미터로 최종 클러스터링 수행:")
    print(f"eps={best_params['eps']}, min_samples={best_params['min_samples']}")
    
    labels, silhouette, dunn = perform_dbscan_clustering(
        file_path, 
        "final_clustered_output.csv",
        eps=best_params['eps'],
        min_samples=int(best_params['min_samples'])
    )
    
    print("\n최종 클러스터링 결과:")
    print(f"클러스터 수: {len(set(labels)) - (1 if -1 in labels else 0)}")
    print(f"노이즈 포인트 수: {list(labels).count(-1)}")
    print(f"Silhouette Coefficient: {silhouette:.3f}")
    print(f"Dunn Index: {dunn:.3f}")