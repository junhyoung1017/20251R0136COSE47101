import numpy as np
from tensorflow.keras import models, layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold


stack_names = ["Android", "Frontend", "ML-Data", "Server", "System", "Visualization", "iOS"]
accuracies = []


X_total = np.load("X_total.npy", allow_pickle=True)
y_onehot = np.load("y_onehot.npy", allow_pickle=True)

input_dim = X_total.shape[1]  # 입력 차원 자동으로 설정
output_dim = y_onehot.shape[1]  # 보통 8

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
for train_index, test_index in kfold.split(X_total):
    print(f"\n--- Fold {fold} ---")
    X_train, X_test = X_total[train_index], X_total[test_index]
    y_train, y_test = y_onehot[train_index], y_onehot[test_index]

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              batch_size=32,
              epochs=50,
              validation_split=0.2,
              verbose=0)

    y_pred = model.predict(X_test)
    y_pred_label = np.argmax(y_pred, axis=1)
    y_true_label = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred_label == y_true_label)
    print(f"Fold {fold} Accuracy: {accuracy:.2%}")
    accuracies.append(accuracy)

    class_names = [f"Class {stack_names[i]}" for i in range(output_dim)]
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_label, y_pred_label))
    print("\nClassification Report (per stack):")
    print(classification_report(y_true_label, y_pred_label, target_names=class_names, digits=4))

    fold += 1

print(f"\nAverage Accuracy across {len(accuracies)} folds: {np.mean(accuracies):.2%}")