import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
#파일 경로
file_path = 'C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/20251R0136COSE47101/Kmeans/github_profiles_total_v4.3.csv'
def split_repos(text):
    repos = str(text).split('/')  # 각 repo 구분
    repo_names = []
    descriptions = []
    for repo in repos:
        parts = repo.split('::')
        name = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else ''
        # 빈 문자열이 아닌 경우만 추가
        if name:
            repo_names.append(name)
        if desc:
            descriptions.append(desc)
    return ', '.join(repo_names), ', '.join(descriptions)

'''#3 언어 데이터 정규화
def normalize_language_data(df):
    # 언어 열만 선택
    language_columns = df.columns.difference(['user_ID', 'username','repo_count', 'repo_names', 'description','stack','note'])
    
    # 각 언어 열의 최대값으로 나누어 정규화
    for col in language_columns:
        max_value = df[col].max()
        if max_value > 0:  # 0으로 나누는 것을 방지
            df[col] = df[col] / max_value
    
    return df'''
'''preprocess
1. 언어 합치기. JS+TS=JS, C+C++=C/C++. 이렇게 하고 다시 정규화
2. text열에서 repo_name과 description 분리(구분자 ::)'''

# 2. 인코딩과 errors 옵션을 직접 open에 넣고 처리
with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    df = pd.read_csv(f)
df["JS"]=df[['JavaScript', 'TypeScript']].sum(axis=1)
df["C/C++"]=df[['C', 'C++']].sum(axis=1)
df[['repo_names', 'description']] = df['text'].apply(lambda x: pd.Series(split_repos(x)))
print(df[['repo_names']].head())
print(df[['description']].head())
df.drop(columns=['JavaScript','TypeScript', 'C',"C++","text"], inplace=True)

# 3. 언어 데이터 전처리(TF-IDF, CountVectorizer 등)
# repo_name과 description에서 stopwords, stemming, lemmatization 등을 적용할 수 있지만, 여기서는 간단히 BERT 임베딩을 사용합니다.
# 사전 학습된 BERT 기반 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')  # 작고 빠르면서 정확도도 좋은 모델

# 결측치 처리 (빈 문자열은 임베딩에서 에러 발생할 수 있음)
df['description'] = df['description'].fillna('')
df['repo_names'] = df['repo_names'].fillna('')

# description 열에 대해 문장 임베딩 생성
description_embeddings = model.encode(df['description'].tolist(), show_progress_bar=True)
# 결과는 numpy 배열 -> pandas DataFrame으로 변환
embedding_df = pd.DataFrame(description_embeddings, columns=[f'bert_desc_{i}' for i in range(description_embeddings.shape[1])])

# ② repo_names 임베딩
name_embeddings = model.encode(df['repo_names'].tolist(), show_progress_bar=True)
name_df = pd.DataFrame(name_embeddings, columns=[f'bert_name_{i}' for i in range(name_embeddings.shape[1])])

# 기존 df와 concat
df = pd.concat([df.reset_index(drop=True), name_df,embedding_df], axis=1)

# 저장 경로 설정 (원하는 경로로 변경 가능)
pickle_path = 'C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/github_profiles_with_bert.pkl'

# 피클 형식으로 저장
df.to_pickle(pickle_path)

print(f"BERT 임베딩 포함 데이터프레임이 다음 경로에 저장됨:\n{pickle_path}")

# 확인
print(df.head())