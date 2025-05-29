import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import models, layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


stack_names = ["Android", "Frontend", "ML-Data", "Server", "System", "Visualization", "iOS"]
accuracies = []

X_total = np.load("X_total.npy", allow_pickle=True)
y_onehot = np.load("y_onehot.npy", allow_pickle=True)

input_dim = X_total.shape[1]  # 입력 차원 자동으로 설정
output_dim = y_onehot.shape[1]  # 보통 8


# 모델 생성 함수 정의
def create_model(input_dim, output_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


X_train, X_test, y_train, y_test = train_test_split(X_total, y_onehot, test_size=0.3, random_state=42)

model = create_model(input_dim, output_dim)

model.fit(X_train, y_train,
          batch_size=32,
          epochs=25,
          validation_split=0.2,
          verbose=0)

X_test = X_test.astype(np.float32)
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)
y_train_pred_label = np.argmax(y_train_pred, axis=1)
y_pred_label = np.argmax(y_pred, axis=1)
y_true_label = np.argmax(y_test, axis=1)


accuracy = np.mean(y_pred_label == y_true_label)
print(f"Test Accuracy: {accuracy:.2%}")
train_accuracy = np.mean(y_train_pred_label == np.argmax(y_train, axis=1))
print(f"Training Accuracy: {train_accuracy:.2%}")

# top-2 확률이 높은 인덱스 2개 추출
top2_pred = np.argsort(y_pred, axis=1)[:, -2:]  # 각 행에 대해 상위 2개 index 추출

# 실제 정답 인덱스
true_labels = np.argmax(y_test, axis=1)

# top-2 중 하나라도 정답이면 맞은 것으로 처리
correct_top2 = np.array([true_labels[i] in top2_pred[i] for i in range(len(true_labels))])
top2_accuracy = np.mean(correct_top2)
print(f"Top-2 Accuracy (one match among top 2): {top2_accuracy:.2%}")

class_names = [f"Class {stack_names[i]}" for i in range(output_dim)]
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true_label, y_pred_label)
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
print(classification_report(y_true_label, y_pred_label, target_names=class_names, digits=4))

# Top-2 예측 기준 confusion matrix 생성
top2_confusion = np.zeros((output_dim, output_dim), dtype=int)

for i in range(len(true_labels)):
    for pred in top2_pred[i]:
        top2_confusion[true_labels[i], pred] += 1

# Top-2 confusion matrix 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(top2_confusion, annot=True, fmt='d', cmap='Oranges',
            xticklabels=stack_names, yticklabels=stack_names)
plt.xlabel('Top-2 Predicted')
plt.ylabel('True Label')
plt.title('Top-2 Confusion Matrix')
plt.tight_layout()
plt.show()