
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, hamming_loss, jaccard_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_preprocessed_data(data_dir):
    """
    전처리된 numpy 데이터를 로드하는 함수
    """
    print("📂 전처리된 데이터 로딩 중...")
    
    # 데이터 파일 경로
    X_train_path = os.path.join(data_dir, "X_train_balanced.npy")
    X_test_path = os.path.join(data_dir, "X_test_balanced.npy") 
    y_train_path = os.path.join(data_dir, "y_train_balanced.npy")
    y_test_path = os.path.join(data_dir, "y_test_balanced.npy")
    metadata_path = os.path.join(data_dir, "metadata_enhanced.pkl")
    class_weights_path = os.path.join(data_dir, "class_weights.pkl")
    
    # 파일 존재 확인
    files_to_check = [X_train_path, X_test_path, y_train_path, y_test_path, metadata_path]
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    # 데이터 로드
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    
    # 메타데이터 로드
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # 클래스 가중치 로드 (있는 경우)
    class_weights = None
    if os.path.exists(class_weights_path):
        with open(class_weights_path, 'rb') as f:
            class_weights = pickle.load(f)
    
    print(f"✅ 데이터 로딩 완료!")
    print(f"📊 데이터 크기:")
    print(f"   • 훈련 데이터: X={X_train.shape}, y={y_train.shape}")
    print(f"   • 테스트 데이터: X={X_test.shape}, y={y_test.shape}")
    print(f"   • 타겟 스택: {metadata['target_stacks']}")
    print(f"   • 총 특성 수: {metadata['total_features']}")
    
    # 클래스 분포 확인
    print(f"\n📊 타겟 스택별 분포 (훈련 세트):")
    for i, stack in enumerate(metadata['target_stacks']):
        count = np.sum(y_train[:, i])
        percentage = (count / len(y_train)) * 100
        print(f"   {stack}: {count}개 ({percentage:.1f}%)")
    
    return X_train, X_test, y_train, y_test, metadata, class_weights

def feature_analysis(X_train, metadata):
    """
    특성 분석 및 정보 출력
    """
    print("\n🔍 특성 분석 중...")
    
    # 특성 타입별 분석
    language_features = len(metadata.get('language_features', []))
    text_features = len(metadata.get('text_features', []))
    embedding_dims = metadata.get('embedding_dims', {})
    
    print(f"📊 특성 구성:")
    print(f"   • 언어 특성: {language_features}개")
    print(f"   • 텍스트 특성: {text_features}개")
    print(f"   • Repository 이름 임베딩: {embedding_dims.get('repo_names', 0)}차원")
    print(f"   • Description 임베딩: {embedding_dims.get('description', 0)}차원")
    
    # 기본 통계
    print(f"\n📈 특성 통계:")
    print(f"   • 평균: {X_train.mean():.4f}")
    print(f"   • 표준편차: {X_train.std():.4f}")
    print(f"   • 최솟값: {X_train.min():.4f}")
    print(f"   • 최댓값: {X_train.max():.4f}")
    
    # 결측치 확인
    nan_count = np.isnan(X_train).sum()
    if nan_count > 0:
        print(f"⚠️ 결측치 발견: {nan_count}개")
        return False
    else:
        print("✅ 결측치 없음")
        return True

def prepare_multilabel_models(class_weights=None):
    """
    멀티라벨 분류를 위한 모델들을 준비하는 함수
    """
    print("\n🤖 멀티라벨 분류 모델 준비 중...")
    
    # 기본 분류기들
    base_classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, 
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'
        ),
        'SVM': SVC(
            random_state=42, 
            probability=True,
            class_weight='balanced'
        )
    }
    
    # 멀티라벨 분류기로 래핑
    multilabel_models = {}
    for name, classifier in base_classifiers.items():
        multilabel_models[name] = MultiOutputClassifier(classifier, n_jobs=-1)
    
    print(f"✅ {len(multilabel_models)}개 멀티라벨 모델 준비 완료")
    return multilabel_models

def multilabel_cross_validation(model, X, y, cv=3):
    """
    멀티라벨 분류를 위한 교차 검증 (평균 라벨 정확도 기준)
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_val_cv)
        
        # 각 라벨별 정확도 계산 후 평균
        label_accuracies = []
        for i in range(y_val_cv.shape[1]):
            acc = accuracy_score(y_val_cv[:, i], y_pred_cv[:, i])
            label_accuracies.append(acc)
        
        mean_label_accuracy = np.mean(label_accuracies)
        scores.append(mean_label_accuracy)
    
    return np.array(scores)

def calculate_exact_match_accuracy(y_true, y_pred):
    """정확한 매칭 정확도 계산 (모든 라벨이 정확해야 함)"""
    return np.mean(np.all(y_pred == y_true, axis=1))

def calculate_topk_accuracy(y_true, y_pred_proba, k_values=[1, 2, 3]):
    """Top-k 정확도 계산"""
    if y_pred_proba is None:
        return {k: 0.0 for k in k_values}
    
    results = {}
    for k in k_values:
        # 각 샘플에 대해 상위 k개 예측
        if len(y_pred_proba) > 0 and hasattr(y_pred_proba[0], '__len__'):
            # MultiOutputClassifier의 경우 각 분류기별 확률을 합치기
            combined_proba = np.zeros((len(y_true), y_true.shape[1]))
            for i in range(y_true.shape[1]):
                combined_proba[:, i] = y_pred_proba[i][:, 1]  # positive class 확률
            y_pred_proba_final = combined_proba
        else:
            y_pred_proba_final = y_pred_proba
        
        top_k_preds = np.argsort(y_pred_proba_final, axis=1)[:, -k:]
        
        # 실제 라벨과 비교
        matches = []
        for i in range(len(y_true)):
            true_labels = set(np.where(y_true[i] == 1)[0])
            pred_labels = set(top_k_preds[i])
            # 교집합이 있으면 성공
            matches.append(len(true_labels & pred_labels) > 0)
        
        accuracy = np.mean(matches)
        results[k] = accuracy
    
    return results

def train_multilabel_models(X_train, y_train, X_test, y_test, models, target_stacks):
    """
    멀티라벨 모델들을 학습하고 평가하는 함수
    """
    print("\n🚀 멀티라벨 모델 학습 및 평가 중...")
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 {name} 학습 중...")
        
        try:
            # 모델 학습
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            # 확률 예측 (가능한 경우)
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                pass
            
            # === 딥러닝 코드와 동일한 평가 지표 계산 ===
            
            # 1. 정확한 매칭 정확도 (Exact Match)
            exact_match_accuracy = calculate_exact_match_accuracy(y_test, y_pred)
            
            # 2. Top-k 정확도
            topk_accuracies = calculate_topk_accuracy(y_test, y_pred_proba, k_values=[1, 2, 3])
            
            # 3. Jaccard Score (딥러닝 코드와 동일)
            jaccard_macro = jaccard_score(y_test, y_pred, average='macro', zero_division=0)
            jaccard_micro = jaccard_score(y_test, y_pred, average='micro', zero_division=0)
            jaccard_samples = jaccard_score(y_test, y_pred, average='samples', zero_division=0)
            
            # 4. Hamming Loss
            hamming = hamming_loss(y_test, y_pred)
            
            # 5. 각 라벨별 정확도 (평균 라벨 정확도)
            label_accuracies = []
            for i in range(y_test.shape[1]):
                acc = accuracy_score(y_test[:, i], y_pred[:, i])
                label_accuracies.append(acc)
            mean_label_accuracy = np.mean(label_accuracies)
            
            # 6. 라벨별 세부 성능 (Precision, Recall, F1)
            label_metrics = []
            for i in range(y_test.shape[1]):
                true_positive = np.sum((y_test[:, i] == 1) & (y_pred[:, i] == 1))
                false_positive = np.sum((y_test[:, i] == 0) & (y_pred[:, i] == 1))
                false_negative = np.sum((y_test[:, i] == 1) & (y_pred[:, i] == 0))
                
                precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
                recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                label_metrics.append({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
            
            # 교차 검증
            try:
                cv_scores = multilabel_cross_validation(model, X_train, y_train, cv=3)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception as e:
                print(f"   ⚠️ 교차 검증 실패: {e}")
                cv_mean, cv_std = 0, 0
            
            # 결과 저장
            results[name] = {
                'model': model,
                # 주요 성능 지표 (딥러닝 코드와 동일)
                'exact_match_accuracy': exact_match_accuracy,
                'top1_accuracy': topk_accuracies[1],
                'top2_accuracy': topk_accuracies[2],
                'top3_accuracy': topk_accuracies[3],
                'jaccard_macro': jaccard_macro,
                'jaccard_micro': jaccard_micro,
                'jaccard_samples': jaccard_samples,
                'hamming_loss': hamming,
                'mean_label_accuracy': mean_label_accuracy,
                # 세부 정보
                'label_accuracies': label_accuracies,
                'label_metrics': label_metrics,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"✅ {name} 완료!")
            print(f"   📊 주요 성능 지표:")
            print(f"     • 정확한 매칭: {exact_match_accuracy:.4f} ({exact_match_accuracy:.2%})")
            print(f"     • Top-1 정확도: {topk_accuracies[1]:.4f} ({topk_accuracies[1]:.2%})")
            print(f"     • Top-2 정확도: {topk_accuracies[2]:.4f} ({topk_accuracies[2]:.2%})")
            print(f"     • Jaccard Score: {jaccard_macro:.4f}")
            print(f"     • 평균 라벨 정확도: {mean_label_accuracy:.4f}")
            print(f"     • 교차 검증: {cv_mean:.4f} (±{cv_std:.4f})")
            
        except Exception as e:
            print(f"❌ {name} 학습 실패: {e}")
            continue
    
    return results

def evaluate_multilabel_results(results, y_test, target_stacks):
    """
    멀티라벨 분류 결과를 평가하고 최고 모델을 선택하는 함수 (딥러닝 코드와 동일한 지표)
    """
    print("\n🏆 멀티라벨 모델 성능 비교:")
    print("-" * 100)
    print(f"{'Model':<20} | {'Exact Match':<12} | {'Top-1':<8} | {'Top-2':<8} | {'Jaccard':<8} | {'Mean Acc':<8}")
    print("-" * 100)
    
    # 성능 비교 출력
    best_score = 0
    best_model_name = None
    
    for name, result in results.items():
        exact_match = result['exact_match_accuracy']
        top1 = result['top1_accuracy']
        top2 = result['top2_accuracy']
        jaccard = result['jaccard_macro']
        mean_acc = result['mean_label_accuracy']
        
        print(f"{name:<20} | {exact_match:<12.4f} | {top1:<8.4f} | {top2:<8.4f} | {jaccard:<8.4f} | {mean_acc:<8.4f}")
        
        # Top-1 정확도를 기준으로 최고 모델 선택 (딥러닝과 비교하기 위해)
        if top1 > best_score:
            best_score = top1
            best_model_name = name
    
    print("-" * 100)
    print(f"\n🥇 최고 성능 모델: {best_model_name}")
    print(f"   Top-1 정확도: {best_score:.4f} ({best_score:.2%})")
    
    # 최고 모델의 상세 평가
    best_result = results[best_model_name]
    
    print(f"\n📊 {best_model_name} 상세 평가:")
    print(f"   🎯 주요 성능 지표:")
    print(f"     • 정확한 매칭 (Exact Match): {best_result['exact_match_accuracy']:.4f} ({best_result['exact_match_accuracy']:.2%})")
    print(f"     • Top-1 정확도: {best_result['top1_accuracy']:.4f} ({best_result['top1_accuracy']:.2%})")
    print(f"     • Top-2 정확도: {best_result['top2_accuracy']:.4f} ({best_result['top2_accuracy']:.2%})")
    print(f"     • Top-3 정확도: {best_result['top3_accuracy']:.4f} ({best_result['top3_accuracy']:.2%})")
    print(f"   📈 추가 지표:")
    print(f"     • Jaccard Score (Macro): {best_result['jaccard_macro']:.4f}")
    print(f"     • Jaccard Score (Micro): {best_result['jaccard_micro']:.4f}")
    print(f"     • Hamming Loss: {best_result['hamming_loss']:.4f}")
    print(f"     • 평균 라벨 정확도: {best_result['mean_label_accuracy']:.4f}")
    
    # 각 라벨별 성능
    print(f"\n📋 스택별 성능:")
    for i, (stack, metrics) in enumerate(zip(target_stacks, best_result['label_metrics'])):
        acc = best_result['label_accuracies'][i]
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']
        print(f"   {stack:>15}: Acc={acc:.3f}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    return best_result['model'], best_model_name, best_result

def visualize_multilabel_results(results, y_test, target_stacks, best_model_name):
    """
    멀티라벨 분류 결과를 시각화하는 함수 (딥러닝 코드와 동일한 지표 포함)
    """
    print("\n📈 결과 시각화 중...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 모델 성능 비교 (Top-1 정확도)
    model_names = list(results.keys())
    top1_scores = [results[name]['top1_accuracy'] for name in model_names]
    exact_match_scores = [results[name]['exact_match_accuracy'] for name in model_names]
    
    axes[0, 0].bar(model_names, top1_scores, alpha=0.7, color='skyblue', label='Top-1')
    axes[0, 0].bar(model_names, exact_match_scores, alpha=0.7, color='lightcoral', label='Exact Match')
    axes[0, 0].set_title('Model Performance (Top-1 vs Exact Match)')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Top-k 정확도 비교 (최고 모델)
    best_result = results[best_model_name]
    topk_values = [best_result['top1_accuracy'], best_result['top2_accuracy'], best_result['top3_accuracy']]
    topk_labels = ['Top-1', 'Top-2', 'Top-3']
    
    axes[0, 1].bar(topk_labels, topk_values, alpha=0.7, color='lightgreen')
    axes[0, 1].set_title(f'{best_model_name} - Top-k Accuracy')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    for i, v in enumerate(topk_values):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 3. Jaccard Score 비교
    jaccard_scores = [results[name]['jaccard_macro'] for name in model_names]
    
    axes[0, 2].bar(model_names, jaccard_scores, alpha=0.7, color='gold')
    axes[0, 2].set_title('Model Performance (Jaccard Score)')
    axes[0, 2].set_ylabel('Jaccard Score')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 최고 모델의 라벨별 정확도
    label_accuracies = best_result['label_accuracies']
    
    axes[1, 0].bar(target_stacks, label_accuracies, alpha=0.7, color='plum')
    axes[1, 0].set_title(f'{best_model_name} - Label-wise Accuracy')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 실제 vs 예측 라벨 분포
    y_pred = best_result['predictions']
    
    true_counts = np.sum(y_test, axis=0)
    pred_counts = np.sum(y_pred, axis=0)
    
    x = np.arange(len(target_stacks))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, true_counts, width, label='True', alpha=0.7, color='steelblue')
    axes[1, 1].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7, color='orange')
    axes[1, 1].set_title('True vs Predicted Label Distribution')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(target_stacks, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 성능 지표 요약 (레이더 차트 대신 바 차트)
    metrics_names = ['Exact Match', 'Top-1', 'Top-2', 'Jaccard', 'Mean Label Acc']
    metrics_values = [
        best_result['exact_match_accuracy'],
        best_result['top1_accuracy'], 
        best_result['top2_accuracy'],
        best_result['jaccard_macro'],
        best_result['mean_label_accuracy']
    ]
    
    axes[1, 2].barh(metrics_names, metrics_values, alpha=0.7, color='lightsteelblue')
    axes[1, 2].set_title(f'{best_model_name} - Performance Summary')
    axes[1, 2].set_xlabel('Score')
    axes[1, 2].grid(True, alpha=0.3)
    for i, v in enumerate(metrics_values):
        axes[1, 2].text(v + 0.01, i, f'{v:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.show()
    
    # 추가: 라벨별 혼동 행렬 시각화
    print("\n📊 라벨별 혼동 행렬:")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, stack in enumerate(target_stacks):
        cm = confusion_matrix(y_test[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{stack}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    # 마지막 축 숨기기
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_best_model(best_model, best_model_name, metadata, data_dir):
    """
    최고 성능 모델을 저장하는 함수
    """
    print(f"\n💾 최고 모델 저장 중...")
    
    model_save_path = os.path.join(data_dir, "best_multilabel_model.pkl")
    
    model_data = {
        'model': best_model,
        'model_name': best_model_name,
        'target_stacks': metadata['target_stacks'],
        'metadata': metadata,
        'model_type': 'multilabel'
    }
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ 모델 저장 완료: {model_save_path}")
    return model_save_path

def main():
    # 데이터 경로 설정
    data_dir = 'C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/pkl_data'
    
    try:
        # 1. 전처리된 데이터 로드
        X_train, X_test, y_train, y_test, metadata, class_weights = load_preprocessed_data(data_dir)
        
        # 2. 특성 분석
        is_valid = feature_analysis(X_train, metadata)
        if not is_valid:
            print("❌ 데이터에 문제가 있습니다. 전처리를 다시 확인해주세요.")
            return
        
        # 3. 멀티라벨 모델 준비
        models = prepare_multilabel_models(class_weights)
        
        # 4. 스케일링 (이미 전처리에서 일부 처리되었지만, 추가 정규화)
        print("\n⚖️ 특성 스케일링 중...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✅ 스케일링 완료")
        
        # 5. 모델 학습 및 평가
        results = train_multilabel_models(
            X_train_scaled, y_train, X_test_scaled, y_test, 
            models, metadata['target_stacks']
        )
        
        if not results:
            print("❌ 모든 모델 학습이 실패했습니다.")
            return
        
        # 6. 결과 평가 및 최고 모델 선택
        best_model, best_model_name, best_result = evaluate_multilabel_results(
            results, y_test, metadata['target_stacks']
        )
        
        # 7. 결과 시각화
        visualize_multilabel_results(
            results, y_test, metadata['target_stacks'], best_model_name
        )
        
        # 8. 최고 모델 저장
        model_save_path = save_best_model(best_model, best_model_name, metadata, data_dir)
        
        # 9. 최종 요약
        print(f"\n🎉 멀티라벨 스택 분류 모델 학습 완료!")
        print(f"📊 최종 결과 요약:")
        print(f"   • 최고 모델: {best_model_name}")
        print(f"   🎯 주요 성능 지표 (딥러닝과 비교용):")
        print(f"     • 정확한 매칭 (Exact Match): {best_result['exact_match_accuracy']:.4f} ({best_result['exact_match_accuracy']:.2%})")
        print(f"     • Top-1 정확도: {best_result['top1_accuracy']:.4f} ({best_result['top1_accuracy']:.2%})")
        print(f"     • Top-2 정확도: {best_result['top2_accuracy']:.4f} ({best_result['top2_accuracy']:.2%})")
        print(f"     • Top-3 정확도: {best_result['top3_accuracy']:.4f} ({best_result['top3_accuracy']:.2%})")
        print(f"     • Jaccard Score: {best_result['jaccard_macro']:.4f}")
        print(f"     • 평균 라벨 정확도: {best_result['mean_label_accuracy']:.4f}")
        print(f"   • 저장 경로: {model_save_path}")
        
        # 성능 비교 가이드
        print(f"\n📈 딥러닝 모델과 성능 비교:")
        print(f"   이 결과를 딥러닝 모델의 다음 지표와 비교하세요:")
        print(f"   - 정확한 매칭 비율 (exact_match_ratio)")
        print(f"   - Top-1, Top-2, Top-3 정확도")
        print(f"   - Jaccard Score (Macro)")
        
        # 사용 가이드
        print(f"\n📖 모델 사용 가이드:")
        print(f"import pickle")
        print(f"with open('{model_save_path}', 'rb') as f:")
        print(f"    model_data = pickle.load(f)")
        print(f"model = model_data['model']")
        print(f"predictions = model.predict(new_X)")
        print(f"# 확률 예측:")
        print(f"probabilities = model.predict_proba(new_X)")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()