import pandas as pd

# 파일 경로
pkl_path = 'C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/github_profiles_with_bert_processed.pkl'

# 데이터 로딩
df = pd.read_pickle(pkl_path)

# 기본 정보 확인
print("📊 데이터 기본 정보:")
print(f"Shape: {df.shape}")
print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 컬럼 확인
print("\n📋 컬럼 정보:")
print(f"총 컬럼 수: {len(df.columns)}")

# 컬럼 종류별로 분류
basic_cols = [col for col in df.columns if not col.startswith('bert_')]
bert_name_cols = [col for col in df.columns if col.startswith('bert_name_')]
bert_desc_cols = [col for col in df.columns if col.startswith('bert_desc_')]

print(f"기본 컬럼 ({len(basic_cols)}개): {basic_cols}")
print(f"BERT name 임베딩 ({len(bert_name_cols)}개): {bert_name_cols[:5]}...")
print(f"BERT desc 임베딩 ({len(bert_desc_cols)}개): {bert_desc_cols[:5]}...")

# 샘플 데이터 확인
print("\n📝 샘플 데이터:")
print(df[basic_cols].head())

# 특정 사용자의 임베딩 확인
print("\n🔍 첫 번째 사용자의 BERT 임베딩 (처음 5개 값):")
print(f"Repository name 임베딩: {df.iloc[0][bert_name_cols[:5]].values}")
print(f"Description 임베딩: {df.iloc[0][bert_desc_cols[:5]].values}")