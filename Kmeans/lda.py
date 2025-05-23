import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 1. user_tokens.json 파일 읽기
with open('nlp_results/user_tokens.json', encoding='utf-8') as f:
    user_tokens = json.load(f)

# 2. 사용자명과 토큰을 문서로 변환
usernames = []
documents = []
for user in user_tokens:
    usernames.append(user['username'])
    # 토큰을 공백으로 합쳐 하나의 문서로 만듦
    documents.append(' '.join(user['tokens']))

# 3. 벡터화 (CountVectorizer)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

# 4. LDA 모델 학습
n_topics = 5  # 토픽 개수는 데이터에 맞게 조정
n_top_words = 10
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_model.fit(X)

# 5. 토픽별 대표 단어 추출
topic_terms = []
for topic_idx, topic in enumerate(lda_model.components_):
    top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    topic_terms.append((f"Topic {topic_idx + 1}", top_features))

# 6. 사용자별 토픽 분포 추출
user_topic_distribution = lda_model.transform(X)
user_topic_df = pd.DataFrame(user_topic_distribution, columns=[f"Topic {i+1}" for i in range(n_topics)])
user_topic_df.insert(0, "username", usernames)

# 7. 결과 출력
print("=== 사용자별 토픽 분포 ===")
print(user_topic_df)
print("\n=== 토픽별 대표 단어 ===")
for topic, words in topic_terms:
    print(f"{topic}: {', '.join(words)}")

user_topic_df.to_json('nlp_results/user_topic_distribution.json', orient='records', force_ascii=False, indent=2)

# 토픽별 대표 단어 저장 (CSV, JSON)
topic_terms_df = pd.DataFrame([
    {"topic": topic, "words": ', '.join(words)}
    for topic, words in topic_terms
])
topic_terms_df.to_json('nlp_results/topic_terms.json', orient='records', force_ascii=False, indent=2)