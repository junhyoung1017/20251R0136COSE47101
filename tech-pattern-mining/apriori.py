import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules

# ① 데이터 불러오기
file_path = 'C:\\Users\\gse07\\Downloads\\github_profiles_total_v5.csv'
df = pd.read_csv(file_path, encoding='utf-8')
df.columns = df.columns.str.strip()

language_columns = ['Assembly', 'C', 'C++', 'C#', 'Dart', 'Go', 'Java', 'JavaScript',
                    'Kotlin', 'MATLAB', 'PHP', 'Python', 'Ruby', 'Rust', 'Scala', 'Swift', 'TypeScript']

# ② threshold 기반 이진화
threshold = 0.2
df_bin = df[language_columns].applymap(lambda x: 1 if x >= threshold else 0)

# ③ Apriori & Association Rules (조금 완화된 파라미터)
frequent_itemsets = apriori(df_bin, min_support=0.003, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage']]

# ④ 네트워크 그래프 구축
G = nx.DiGraph()

# 색상 맵 (confidence)
confidences = rules['confidence'].tolist()
norm_conf = mcolors.Normalize(vmin=min(confidences), vmax=max(confidences))
cmap_conf = plt.get_cmap('RdYlGn_r')  # Red (low) → Green (high)

# 두께 맵 (lift)
lifts = rules['lift'].tolist()
norm_lift = mcolors.Normalize(vmin=min(lifts), vmax=max(lifts))

for _, row in rules.iterrows():
    for antecedent in row['antecedents']:
        for consequent in row['consequents']:
            confidence = row['confidence']
            lift = row['lift']
            color = cmap_conf(norm_conf(confidence))
            width = 1 + 4 * norm_lift(lift)  # 두께는 1~5 정도로 스케일링
            G.add_edge(antecedent, consequent, confidence=confidence, color=color, width=width)

# 노드 위치 설정
fig, ax = plt.subplots(figsize=(15, 12))
pos = nx.spring_layout(G, seed=42, k=0.7)

# 엣지 색상 및 두께 추출
edges = G.edges()
edge_colors = [G[u][v]['color'] for u, v in edges]
edge_widths = [G[u][v]['width'] for u, v in edges]

# 노드 및 엣지 그리기
nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='skyblue', edgecolors='black')
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrows=True, arrowstyle='-|>', arrowsize=20, connectionstyle='arc3,rad=0.15')
nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')

# 엣지 confidence 값 표시
edge_labels = {(u, v): f"{G[u][v]['confidence']:.2f}" for u, v in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

# 컬러바 추가 (confidence용)
sm_conf = cm.ScalarMappable(cmap=cmap_conf, norm=norm_conf)
sm_conf.set_array([])
cbar_conf = plt.colorbar(sm_conf, ax=ax, shrink=0.7)
cbar_conf.set_label('Confidence')

plt.title("Developer Stack Association Network", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()
