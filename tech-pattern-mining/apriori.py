import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import networkx as nx
import re
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# ① 데이터 불러오기
file_path = 'C:\\Users\\gse07\\Downloads\\github_profiles_total_v5.csv'
df = pd.read_csv(file_path, encoding='utf-8')
df.columns = df.columns.str.strip()

# ② 언어 이진화
language_columns = ['Assembly', 'C', 'C++', 'C#', 'Dart', 'Go', 'Java', 'JavaScript',
                    'Kotlin', 'MATLAB', 'PHP', 'Python', 'Ruby', 'Rust', 'Scala', 'Swift', 'TypeScript']

threshold = 0.2
df_bin = df[language_columns].applymap(lambda x: 1 if x >= threshold else 0).astype(bool)

# ③ 키워드 추출
keyword_list = [
    # 기존 키워드
    'Next.js', 'TensorFlow', 'PyTorch', 'React', 'Vue', 'Selenium',
    'GAN', 'OAuth', 'Kubernetes', 'Redis', 'Docker', 'GraphQL',
    
    # AI/ML 추가
    'scikit-learn', 'XGBoost', 'LightGBM', 'CatBoost', 'Transformers',
    'OpenCV', 'Stable Diffusion', 'YOLO', 'LLaMA', 'BERT', 'LangChain',
    
    # 데이터 엔지니어링
    'Pandas', 'NumPy', 'Spark', 'Hadoop', 'Airflow', 'Kafka', 'DBT',
    
    # 클라우드/DevOps
    'AWS', 'GCP', 'Azure', 'Terraform', 'Ansible', 'Prometheus', 'Grafana', 'Helm',
    
    # 프론트엔드 추가
    'TypeScript', 'Svelte', 'Astro', 'TailwindCSS', 'Webpack', 'Vite', 'Babel',
    
    # 백엔드 프레임워크
    'Spring Boot', 'Django', 'FastAPI', 'Flask', 'Express.js', 'NestJS', 'Rails',
    
    # 데이터베이스
    'PostgreSQL', 'MySQL', 'MongoDB', 'Elasticsearch', 'Neo4j',
    
    # 보안
    'JWT', 'SSO', 'OAuth2'
]


def extract_keywords(text):
    keywords = []
    if pd.isna(text):
        return keywords
    for kw in keyword_list:
        if re.search(re.escape(kw), text, re.IGNORECASE):
            keywords.append(kw)
    return keywords

df['keywords'] = df['text'].apply(extract_keywords)

# ④ 키워드 이진화
mlb = MultiLabelBinarizer()
keyword_df = pd.DataFrame(mlb.fit_transform(df['keywords']), columns=mlb.classes_)

# ⑤ 언어 + 키워드 통합
df_combined = pd.concat([df_bin, keyword_df], axis=1)

# ⑥ Apriori & Association Rules
frequent_itemsets = apriori(df_combined, min_support=0.003, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage']]

### Association Rule Table Formatting
rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))

# lift 기준 정렬
rules_sorted = rules.sort_values(by='lift', ascending=False)
print("\n[ Association Rule Table ]\n")
print(rules_sorted[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].head(20))

# 시각화용 rule 필터링 (여기서만 필터 적용)
viz_rules = rules_sorted[(rules_sorted['confidence'] >= 0.5) & (rules_sorted['lift'] >= 5)]
# ⑦ 네트워크 그래프 구축
G = nx.DiGraph()

# 색상맵 (confidence)
confidences = viz_rules['confidence'].tolist()
norm_conf = mcolors.Normalize(vmin=min(confidences), vmax=max(confidences))
cmap_conf = plt.get_cmap('RdYlGn_r')

# 두께맵 (lift)
lifts = viz_rules['lift'].tolist()
norm_lift = mcolors.Normalize(vmin=min(lifts), vmax=max(lifts))

for _, row in viz_rules.iterrows():
    for antecedent in row['antecedents']:
        for consequent in row['consequents']:
            confidence = row['confidence']
            lift = row['lift']
            color = cmap_conf(norm_conf(confidence))
            width = 1 + 4 * norm_lift(lift)
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

# 컬러바 추가
sm_conf = cm.ScalarMappable(cmap=cmap_conf, norm=norm_conf)
sm_conf.set_array([])
cbar_conf = plt.colorbar(sm_conf, ax=ax, shrink=0.7)
cbar_conf.set_label('Confidence')

plt.title("Developer Stack Association Network (Strong Rules Only)", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()


#################################
# A. 특정 유저의 스택 추천
username = ''

# 유저 스택 추출
def get_user_stack(username, df, df_combined):
    user_row = df[df['username'] == username].iloc[0]
    user_bin = df_combined.loc[user_row.name]
    user_stack = set(user_bin[user_bin == 1].index)
    return user_stack

# 스택 추천
def recommend_stack(user_stack, rules_sorted, top_n=5):
    recommendations = []

    for _, row in rules_sorted.iterrows():
        antecedents = set(row['antecedents'])
        consequents = set(row['consequents'])
        if antecedents.issubset(user_stack) and not consequents.issubset(user_stack):
            recommendations.append({
                'antecedents': antecedents,
                'consequents': consequents,
                'confidence': row['confidence'],
                'lift': row['lift']
            })

    recommendations = sorted(recommendations, key=lambda x: (-x['lift'], -x['confidence']))
    return recommendations[:top_n]

# 실제 실행
user_stack = get_user_stack(username, df, df_combined)
recommendations = recommend_stack(user_stack, rules_sorted)

for rec in recommendations:
    print(f"기존 보유: {rec['antecedents']} → 추천 스택: {rec['consequents']}")
    print(f"  confidence: {rec['confidence']}, lift: {rec['lift']}\n")




##############################
# B. 유사한 유저 추천
def get_user_index(username, df):
    return df[df['username'] == username].index[0]

def recommend_similar_users(username, df, df_combined, top_n=5):
    user_idx = get_user_index(username, df)
    
    # 전체 cosine similarity 계산
    sim_matrix = cosine_similarity(df_combined.values)
    
    # 해당 유저의 similarity vector
    user_sim = sim_matrix[user_idx]
    
    # 자기 자신 제외 (자기 자신 similarity=1)
    sim_scores = list(enumerate(user_sim))
    sim_scores = [(i, score) for i, score in sim_scores if i != user_idx]
    
    # 높은 순으로 정렬
    sim_scores = sorted(sim_scores, key=lambda x: -x[1])
    
    # top-N 추출
    top_indices = [i for i, _ in sim_scores[:top_n]]
    
    # 추천 유저 출력
    recommended_users = df.iloc[top_indices]['username'].tolist()
    return recommended_users

similar_users = recommend_similar_users(username, df, df_combined)
print(f"{username}와 유사한 유저 추천: {similar_users}")