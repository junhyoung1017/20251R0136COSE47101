import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 데이터 불러오기
df = pd.read_csv("assets/github_profiles_total_v5.csv")
# 컬럼명 공백 제거
df.columns = df.columns.str.strip()

# 타겟 라벨 전처리 (가장 첫 번째 스택만 사용)
df["main_stack"] = df["stack"].apply(lambda x: x.split("&")[0].strip())

# feature 추출
language_columns = [
    "Assembly",
    "C",
    "C++",
    "C#",
    "Dart",
    "Go",
    "Java",
    "JavaScript",
    "Kotlin",
    "MATLAB",
    "PHP",
    "Python",
    "Ruby",
    "Rust",
    "Scala",
    "Swift",
    "TypeScript",
]
X_num = df[language_columns + ["repo_count"]].fillna(0)

# 텍스트 feature 벡터화
vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
X_text = vectorizer.fit_transform(df["text"].fillna("")).toarray()

# feature 합치기
X = np.hstack([X_num.values, X_text])
y = df["main_stack"]

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Decision Tree(GINI) 모델 학습
clf = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=10)
clf.fit(X_train, y_train)

# 예측 및 평가
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Decision Tree(entropy) 모델 학습
clf = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=10)
clf.fit(X_train, y_train)

# 예측 및 평가
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Random Forest 모델 학습
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# KNN 분류 모델 학습
knn = KNeighborsClassifier(n_neighbors=6)  # k값은 상황에 따라 조정 가능
knn.fit(X_train, y_train)

# 예측 및 평가
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))

# feature scaling (SVM은 거리 기반이므로 꼭 필요!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# SVM 분류 모델 학습
svm = SVC(kernel="poly", random_state=42)  # kernel='rbf', 'poly' 등도 실험 가능
svm.fit(X_train, y_train)

# 예측 및 평가
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))
