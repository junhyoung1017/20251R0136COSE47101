# Multilabel Classification Deep Learning Model with Keras

## 사용법
1. multilabel_classification.py 위치 폴더 내에 X_total_v5.npy, y_onehot_v5.npy, github_profiles_total_v5.csv 위치시킴
2. numpy, pandas, matplotlib, seaborn, tensorflow, sklearn 모듈 설치
3. 실행.

## 실행 결과
- 상위에 Top-1 정확도와 Top-2 정확도가 출력
- Training data 에 대한 에측 정확도 출력 (과적합 판단 용)
- Confusion matrix 출력
- Top-2 예측 결과를 기존 CSV 파일의 새로운 column 으로 추가하여 저장