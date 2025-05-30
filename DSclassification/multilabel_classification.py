import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import models, layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ===============================
# 다중 레이블 분류를 위한 전체 코드 구조
# ===============================

# 클래스(스택) 이름 정의
stack_names = ["Android", "Frontend", "ML-Data", "Server", "System", "Visualization", "iOS"]

# 데이터 로드 (입력 특징과 다중 레이블 원-핫)
X_total = np.load("X_total_v5.npy", allow_pickle=True)
y_onehot = np.load("y_onehot_v5.npy", allow_pickle=True)

# 입출력 차원 확인
input_dim = X_total.shape[1]
output_dim = y_onehot.shape[1]

# -------------------------------------
# 모델 생성 함수: 다중 레이블용 sigmoid, binary_crossentropy 사용
# -------------------------------------
def create_model(input_dim, output_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_dim, activation='sigmoid')  # 다중 레이블 분류이므로 sigmoid
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# -------------------------------------
# 데이터 분할 (학습/테스트)
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_total, y_onehot, test_size=0.2, random_state=42)

# -------------------------------------
# 모델 학습
# -------------------------------------
model = create_model(input_dim, output_dim)
model.fit(X_train, y_train,
          batch_size=32,
          epochs=25,
          validation_split=0.2,
          verbose=0)

# -------------------------------------
# 예측 (테스트/학습)
# -------------------------------------
X_test = X_test.astype(np.float32)
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# -------------------------------------
# Top-1 Accuracy: 예측 상위 1개가 실제 라벨 중 하나와 겹치면 정답
# -------------------------------------
y_pred_top1 = np.argmax(y_pred, axis=1)
y_true_labels = [np.where(row == 1)[0] for row in y_test]
top1_match = [pred in true for pred, true in zip(y_pred_top1, y_true_labels)]
top1_accuracy = np.mean(top1_match)
print(f"Top-1 Match Accuracy (any match in multi-label): {top1_accuracy:.2%}")

# -------------------------------------
# Top-2 Accuracy: 예측 상위 2개 중 하나라도 실제 라벨에 포함되면 정답
# -------------------------------------
y_pred_top2 = np.argsort(y_pred, axis=1)[:, -2:]
top2_match = [any(pred in true for pred in preds) for preds, true in zip(y_pred_top2, y_true_labels)]
top2_accuracy = np.mean(top2_match)
print(f"Top-2 Match Accuracy (any match in multi-label): {top2_accuracy:.2%}")

# -------------------------------------
# 학습 정확도 (Top-1 기준, multi-label)
# -------------------------------------
y_train_pred_label = np.argmax(y_train_pred, axis=1)
y_train_true_label = [np.where(row == 1)[0] for row in y_train]
train_correct = [pred in true for pred, true in zip(y_train_pred_label, y_train_true_label)]
train_accuracy = np.mean(train_correct)
print(f"Training Accuracy (Top-1 match): {train_accuracy:.2%}")

# -------------------------------------
# Confusion Matrix 및 Classification Report (첫 번째 실제 라벨만 사용)
# -------------------------------------
y_true_label_flat = [labels[0] if len(labels) > 0 else -1 for labels in y_true_labels]

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true_label_flat, y_pred_top1)
print(cm)

# Confusion matrix 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=stack_names, yticklabels=stack_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

print("\nClassification Report (per stack):")
print(classification_report(y_true_label_flat, y_pred_top1, target_names=stack_names, digits=4))

# -------------------------------------
# 전체 데이터셋에 대해 분류 후 새로운 CSV 파일로 저장
# -------------------------------------

# 전체 데이터셋에 대한 예측
y_pred_all = model.predict(X_total)
y_pred_all_top2 = np.argsort(y_pred_all, axis=1)[:, -2:]  # 상위 2개 index

# 인덱스를 스택 이름으로 변환
top2_stack_labels_all = [[stack_names[i] for i in row[::-1]] for row in y_pred_all_top2]
top2_stack_str_all = ['&'.join(labels) for labels in top2_stack_labels_all]

# 기존 CSV 파일 불러오기
df_all = pd.read_csv("github_profiles_total_v5.csv")

# 새로운 column 추가
df_all["top2_predicted_stack"] = top2_stack_str_all

# 저장
df_all.to_csv("github_profiles_total_with_top2.csv", index=False)
print("예측 결과가 github_profiles_total_with_top2.csv 에 저장되었습니다.")