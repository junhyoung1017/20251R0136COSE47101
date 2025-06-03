import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import re
from typing import Tuple, List
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
# 파일 경로
file_path = 'C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/20251R0136COSE47101/Kmeans/github_profiles_total_v5.csv'

def split_repos(text: str) -> Tuple[str, str]:
    """
    Repository 텍스트를 이름과 설명으로 분리하는 함수
    """
    if pd.isna(text) or text == '':
        return '', ''
    
    repos = str(text).split('/')  # 각 repo 구분
    repo_names = []
    descriptions = []
    
    for repo in repos:
        parts = repo.split('::')
        name = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else ''
        
        # 빈 문자열이 아닌 경우만 추가
        if name and name != 'nan':
            repo_names.append(name)
        if desc and desc != 'nan':
            descriptions.append(desc)
    
    return ', '.join(repo_names), ', '.join(descriptions)
def process_stack(stack_text: str) -> List[str]:
    """
    Stack 텍스트를 &으로 분리하고 정제하는 함수
    """
    if pd.isna(stack_text) or stack_text == '':
        return []
    
    # &으로 분리하고 각 스택 정제
    stacks = [s.strip() for s in str(stack_text).split('&') if s.strip()]
    
    # 빈 문자열이나 'nan' 제거
    stacks = [s for s in stacks if s and s.lower() != 'nan']
    
    return stacks
def clean_text(text: str) -> str:
    """
    텍스트 전처리 함수 (특수문자 제거, 소문자 변환 등)
    """
    if pd.isna(text) or text == '':
        return ''
    # 특수문자 제거 (영문, 숫자, 공백만 남김)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
    # 연속된 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    # 앞뒤 공백 제거
    text = text.strip()
    return text

def main():
    print("📁 데이터 로딩 중...")
    
    # 데이터 로딩 (인코딩 에러 처리)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            df = pd.read_csv(f)
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return
    
    print(f"✅ 데이터 로딩 완료. 총 {len(df)}개 행")
    print(f"📊 컬럼: {list(df.columns)}")
    
    # 1. 언어 데이터 통합
    print("\n🔄 언어 데이터 통합 중...")
    # JavaScript + TypeScript = JS
    if 'JavaScript' in df.columns and 'TypeScript' in df.columns:
        df["JS"] = df[['JavaScript', 'TypeScript']].sum(axis=1)
        df.drop(columns=['JavaScript', 'TypeScript'], inplace=True)
        print("✅ JavaScript + TypeScript → JS 통합 완료")
    # C + C++ = C/C++
    if 'C' in df.columns and 'C++' in df.columns:
        df["C/C++"] = df[['C', 'C++']].sum(axis=1)
        df.drop(columns=['C', 'C++'], inplace=True)
        print("✅ C + C++ → C/C++ 통합 완료")
    # 2-1. Repository 이름과 설명 분리
    print("\n📝 Repository 텍스트 분리 중...")
    if 'text' in df.columns:
        df[['repo_names', 'description']] = df['text'].apply(lambda x: pd.Series(split_repos(x)))
        df.drop(columns=['text'], inplace=True)
        print("✅ Repository 이름과 설명 분리 완료")
        
        # 분리 결과 확인
        print(f"📋 Repository 이름 샘플:\n{df['repo_names'].head()}")
        print(f"📋 Description 샘플:\n{df['description'].head()}")
   # 2-2. Stack 분리 및 정제 (개선된 버전)
    if 'stack' in df.columns:
        print("\n🔄 Stack 분리 및 정제 중...")
        
        # 원본 stack 데이터 확인
        print(f"📋 원본 Stack 샘플:\n{df['stack'].head()}")
        
        # Stack을 리스트로 분리
        df['stack_list'] = df['stack'].apply(process_stack)
        
        # 분리 결과 확인
        print(f"📋 분리된 Stack 샘플:")
        for i in range(min(5, len(df))):
            print(f"   {i+1}. 원본: '{df['stack'].iloc[i]}' → 분리: {df['stack_list'].iloc[i]}")
        
        # 스택 통계 정보
        all_stacks = []
        for stack_list in df['stack_list']:
            all_stacks.extend(stack_list)
        
        unique_stacks = list(set(all_stacks))
        print(f"📊 전체 고유 스택 수: {len(unique_stacks)}")
        print(f"📊 가장 많이 사용된 스택 상위 10개:")
        
        from collections import Counter
        stack_counts = Counter(all_stacks)
        for stack, count in stack_counts.most_common(10):
            print(f"   {stack}: {count}회")
        
        print("✅ Stack 분리 및 정제 완료")
    # 3. 언어 컬럼 확인
    print("\n📊 언어 데이터 확인 중...")
    # 언어 컬럼 식별 (숫자형 데이터인 컬럼들)
    exclude_columns = {'user_ID', 'username', 'repo_count', 'repo_names', 'description', 'stack','stack_list', 'note'}
    language_columns = [col for col in df.columns if col not in exclude_columns and df[col].dtype in ['int64', 'float64']]
    
    print(f"🎯 언어 컬럼: {language_columns}")
    print("✅ 언어 데이터 확인 완료")
    
    # 4. 텍스트 전처리
    print("\n🧹 텍스트 전처리 중...")
    
    # 결측치 처리 및 텍스트 정제
    df['description'] = df['description'].fillna('').apply(clean_text)
    df['repo_names'] = df['repo_names'].fillna('').apply(clean_text)
    
    # 빈 문자열을 "no description" 또는 "no repository"로 대체 (BERT 임베딩을 위해)
    df['description'] = df['description'].replace('', 'no description available')
    df['repo_names'] = df['repo_names'].replace('', 'no repository name')
    
    print("✅ 텍스트 전처리 완료")
    # 5. BERT 임베딩 생성
    print("\n🤖 BERT 모델 로딩 중...")
    
    try:
        # 사전 학습된 BERT 기반 모델 로드 (다국어 지원 모델로 변경 가능)
        # model = SentenceTransformer('all-MiniLM-L6-v2')  # 빠르고 효과적인 모델  #384차원
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') #768차원
        print("✅ BERT 모델 로딩 완료")
        # Description 임베딩 생성
        print("\n📝 Description 임베딩 생성 중...")
        description_embeddings = model.encode(
            df['description'].tolist(), 
            show_progress_bar=True,
            batch_size=32,  # 메모리 효율성을 위해 배치 크기 설정
            convert_to_numpy=True
        )
        
        # Repository names 임베딩 생성
        print("\n📁 Repository names 임베딩 생성 중...")
        name_embeddings = model.encode(
            df['repo_names'].tolist(), 
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        print("✅ BERT 임베딩 생성 완료")
        
    except Exception as e:
        print(f"❌ BERT 임베딩 생성 실패: {e}")
        return
    
    # 6. 임베딩을 DataFrame으로 변환 및 결합
    print("\n🔗 데이터 결합 중...")
    # Description 임베딩 DataFrame 생성
    embedding_df = pd.DataFrame(
        description_embeddings, 
        columns=[f'bert_desc_{i}' for i in range(description_embeddings.shape[1])]
    )
    # Repository names 임베딩 DataFrame 생성
    name_df = pd.DataFrame(
        name_embeddings, 
        columns=[f'bert_name_{i}' for i in range(name_embeddings.shape[1])]
    )
    
    # 기존 DataFrame과 결합
    df = pd.concat([df.reset_index(drop=True), name_df, embedding_df], axis=1)
    
    print("✅ 데이터 결합 완료")
    print(f"📊 최종 DataFrame 크기: {df.shape}")
    
    # 7. 결과 저장
    print("\n💾 결과 저장 중...")
    
    # 저장 경로 설정
    output_dir = 'C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/pkl_data'
    os.makedirs(output_dir, exist_ok=True)
    
    pickle_path = os.path.join(output_dir, 'github_profiles_with_bert_processed_v2.pkl')
    csv_path = os.path.join(output_dir, 'github_profiles_with_bert_processed.csv')
    
    try:
        # 피클 형식으로 저장 (임베딩 데이터 보존)
        df.to_pickle(pickle_path)
        print(f"✅ 피클 파일 저장 완료: {pickle_path}")
        
        '''# CSV 형식으로도 저장 (일부 컬럼만)
        # BERT 임베딩은 너무 크므로 원본 데이터와 언어 통계만 저장
        basic_columns = [col for col in df.columns if not col.startswith('bert_')]
        df[basic_columns].to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ CSV 파일 저장 완료: {csv_path}")'''
        # Stack 정보도 별도로 저장 (예측 과정에서 활용)
    except Exception as e:
        print(f"❌ 파일 저장 실패: {e}")
        return
    
    # 8. 결과 요약
    print("\n📈 처리 결과 요약:")
    print(f"• 총 사용자 수: {len(df)}")
    print(f"• 언어 컬럼 수: {len(language_columns)}")
    print(f"• Description 임베딩 차원: {description_embeddings.shape[1]}")
    print(f"• Repository names 임베딩 차원: {name_embeddings.shape[1]}")
    print(f"• 최종 피처 수: {df.shape[1]}")
    
    # 샘플 데이터 확인
    print(f"\n📋 최종 데이터 샘플:")
    display_columns = ['username', 'repo_count'] + language_columns[:3] + ['repo_names', 'description']
    available_columns = [col for col in display_columns if col in df.columns]
    print(df[available_columns].head())
    
    print("\n🎉 전처리 완료!")
    # 1. X 구성
    X_lang = df[language_columns].values.astype(np.float32)
    X_desc = df[[col for col in df.columns if col.startswith('bert_desc_')]].values
    X_name = df[[col for col in df.columns if col.startswith('bert_name_')]].values
    X_total = np.concatenate([X_lang, X_name, X_desc], axis=1).astype(np.float32)

    # 2. y 구성
    le = LabelEncoder()
    y_idx = le.fit_transform(df['stack'])  # 또는 가장 대표적인 stack만 고르거나 처리 방식 조정
    def make_onehot(labels, num_classes=None):
        labels = np.array(labels)
        if num_classes is None:
            num_classes = labels.max() + 1
        onehot = np.zeros((len(labels), num_classes), dtype=np.float32)
        onehot[np.arange(len(labels)), labels] = 1.0
        return onehot
    y_onehot = make_onehot(y_idx)

    # 타겟 스택 정의
    target_stacks = ["Android", "Frontend", "ML-Data", "Server", "System", "Visualization", "iOS"]
    
    # stack_list가 포함된 행 중, target 스택만 포함된 리스트 만들기
    df['filtered_stack'] = df['stack_list'].apply(lambda lst: [s for s in lst if s in target_stacks])
    valid_mask = df['filtered_stack'].apply(lambda x: len(x) > 0)
    print(len(df['filtered_stack']),len(df['stack_list']))
    # X와 y 필터링
    X_filtered = X_total[valid_mask.to_numpy()]
    filtered_stack_lists = df.loc[valid_mask, 'filtered_stack'].tolist()
    
    # 원-핫 인코딩
    mlb = MultiLabelBinarizer(classes=target_stacks)
    y_filtered = mlb.fit_transform(filtered_stack_lists)
    
    # 저장
    np.save(os.path.join(output_dir, "X_filtered.npy"), X_filtered)
    np.save(os.path.join(output_dir, "y_filtered.npy"), y_filtered)
    
    print("✅ 필터링된 X/y 저장 완료!")
    print(f"X_filtered shape: {X_filtered.shape}")
    print(f"y_filtered shape: {y_filtered.shape}")
if __name__ == "__main__":
    main()