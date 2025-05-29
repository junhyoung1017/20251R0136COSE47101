import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import LabelEncoder


# CSV 파일 불러오기
file_path = ("github_profiles_total_v4.3.csv")
df = pd.read_csv(file_path, encoding='utf-8')

print("File loaded successfully.")

# 열 이름에서 공백 제거
df.columns = df.columns.str.strip()

language_columns = df.columns[3:19]  # 언어 관련 컬럼들
X_lang = df[language_columns].fillna(0).values  # NaN 방지

# print(X_lang)

# # 상위 몇 줄 확인
# print(df.head())

print("Preprocessing text data...")

# "레포제목 :: 설명 / ..." 형태를 "레포제목: 설명. ..." 형태로 변환하는 함수
def preprocess_text(raw_text):
    if pd.isna(raw_text):
        return ""

    parts = []
    for segment in raw_text.split('/'):
        if '::' in segment:
            split_parts = segment.split('::', 1)
            if len(split_parts) == 2:
                title, desc = split_parts
                parts.append(f"{title.strip()}: {desc.strip()}")
    return '. '.join(parts)


X_text = df['text'].apply(preprocess_text).tolist()

print("Text preprocessing completed.")

# # 상위 5개 문장 확인
# print(X_text[:5])

# BERT 모델을 사용하여 텍스트 임베딩 생성
model = SentenceTransformer('all-mpnet-base-v2')  # 빠르고 성능 균형 좋음
X_text_embed = model.encode(X_text, show_progress_bar=True)

X_total = np.concatenate([X_lang, X_text_embed], axis=1).astype(np.float32)

print("Text embedding completed.")


def make_onehot(labels, num_classes=None):
    """
    정수 인덱스 리스트를 one-hot 벡터로 변환
    labels: [3, 1, 0, 2, ...] 같은 정수 리스트 또는 배열
    num_classes: 클래스 개수. None이면 자동으로 max+1 사용
    """
    labels = np.array(labels)
    if num_classes is None:
        num_classes = int(labels.max()) + 1

    onehot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    onehot[np.arange(labels.shape[0]), labels] = 1.0
    return onehot


le = LabelEncoder()
y_idx = le.fit_transform(df['stack'])
y_onehot = make_onehot(y_idx, num_classes=len(le.classes_))

for idx, stack_name in enumerate(le.classes_):
    print(f"Index {idx}: {stack_name}")

# npz 파일로 결과 저장
np.save("X_total.npy", X_total.astype(np.float32))
np.save("y_onehot.npy", y_onehot.astype(np.float32))

print(y_onehot)  # 상위 5개 원-핫 벡터 확인


# 나중에 불러올 경우
# X_total = np.load("X_total.npy", allow_pickle=True)
# y_onehot = np.load("y_onehot.npy", allow_pickle=True)
# 사용하면 됨
# 이후에 model.fit(X_total, y_onehot, batch_size=32, epochs=10, validation_split=0.2) 로 바로 학습 가능.