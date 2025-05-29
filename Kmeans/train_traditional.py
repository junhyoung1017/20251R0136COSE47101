import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data(pkl_path):
    """
    피클 파일을 로드하고 데이터를 탐색하는 함수
    """
    print("📂 데이터 로딩 중...")
    df = pd.read_pickle(pkl_path)
    
    print(f"✅ 데이터 로딩 완료!")
    print(f"📊 데이터 크기: {df.shape}")
    print(f"💾 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 컬럼 분류
    basic_cols = [col for col in df.columns if not col.startswith('bert_')]
    bert_name_cols = [col for col in df.columns if col.startswith('bert_name_')]
    bert_desc_cols = [col for col in df.columns if col.startswith('bert_desc_')]
    
    print(f"\n📋 컬럼 구성:")
    print(f"• 기본 컬럼: {len(basic_cols)}개")
    print(f"• BERT name 임베딩: {len(bert_name_cols)}개")
    print(f"• BERT description 임베딩: {len(bert_desc_cols)}개")
    
    # 타겟 변수 확인
    if 'stack' in df.columns:
        print(f"\n🎯 타겟 변수 (stack) 분포:")
        stack_counts = df['stack'].value_counts()
        print(stack_counts)
        
        # 결측치 확인
        null_count = df['stack'].isnull().sum()
        print(f"\n❗ 결측치: {null_count}개 ({null_count/len(df)*100:.2f}%)")
        
        return df, basic_cols, bert_name_cols, bert_desc_cols, stack_counts
    else:
        print("❌ 'stack' 컬럼을 찾을 수 없습니다!")
        return df, basic_cols, bert_name_cols, bert_desc_cols, None

def prepare_features(df, basic_cols, bert_name_cols, bert_desc_cols):
    """
    모델 학습을 위한 피처 준비
    """
    print("\n🔧 피처 준비 중...")
    
    # 1. 언어 통계 피처 추출
    exclude_cols = {'user_ID', 'username', 'repo_names', 'description', 'stack', 'note'}
    language_cols = [col for col in basic_cols if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    print(f"🎯 언어 통계 피처: {language_cols}")
    
    # 2. 기본 정보 피처
    basic_info_cols = ['repo_count'] if 'repo_count' in df.columns else []
    
    # 3. 전체 피처 결합
    feature_columns = language_cols + basic_info_cols + bert_name_cols + bert_desc_cols
    
    print(f"📊 총 피처 수: {len(feature_columns)}")
    print(f"• 언어 통계: {len(language_cols)}개")
    print(f"• 기본 정보: {len(basic_info_cols)}개") 
    print(f"• BERT name: {len(bert_name_cols)}개")
    print(f"• BERT desc: {len(bert_desc_cols)}개")
    
    # 피처 데이터 추출
    X = df[feature_columns].copy()
    
    # 결측치 처리
    print(f"\n🔍 결측치 확인:")
    null_counts = X.isnull().sum()
    if null_counts.sum() > 0:
        print(f"결측치가 있는 컬럼: {null_counts[null_counts > 0]}")
        X = X.fillna(0)  # 결측치를 0으로 채움
        print("✅ 결측치를 0으로 처리했습니다.")
    else:
        print("✅ 결측치가 없습니다.")
    
    return X, feature_columns, language_cols

def prepare_target(df):
    """
    타겟 변수 준비
    """
    print("\n🎯 타겟 변수 준비 중...")
    
    # 결측치 제거
    valid_mask = df['stack'].notna()
    print(f"유효한 타겟 데이터: {valid_mask.sum()}개")
    
    if valid_mask.sum() == 0:
        raise ValueError("모든 타겟 값이 결측치입니다!")
    
    # 레이블 인코딩
    le = LabelEncoder()
    y_encoded = le.fit_transform(df.loc[valid_mask, 'stack'])
    
    print(f"✅ 타겟 클래스 수: {len(le.classes_)}")
    print(f"클래스 목록: {list(le.classes_)}")
    
    return y_encoded, valid_mask, le

def feature_selection(X_train, y_train, k=1000):
    """
    중요한 피처 선택 (너무 많은 피처로 인한 과적합 방지)
    """
    print(f"\n🔍 피처 선택 중 (상위 {k}개)...")
    
    if X_train.shape[1] <= k:
        print(f"전체 피처 수({X_train.shape[1]})가 선택 개수({k})보다 적습니다.")
        return X_train, X_train.columns, None
    
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_train, y_train)
    
    # 선택된 피처 이름 가져오기
    selected_features = X_train.columns[selector.get_support()]
    
    print(f"✅ {len(selected_features)}개 피처 선택 완료")
    
    return X_selected, selected_features, selector

def train_models(X_train, y_train, X_test, y_test):
    """
    여러 모델 학습 및 평가
    """
    print("\n🤖 모델 학습 중...")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 {name} 학습 중...")
        
        # 모델 학습
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 평가
        accuracy = accuracy_score(y_test, y_pred)
        
        # 교차 검증
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"✅ {name} 완료!")
        print(f"   테스트 정확도: {accuracy:.4f}")
        print(f"   교차 검증 정확도: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    return results

def evaluate_best_model(results, X_test, y_test, label_encoder):
    """
    최고 성능 모델 평가 및 결과 출력
    """
    print("\n🏆 모델 성능 비교:")
    print("-" * 60)
    
    # 성능 비교
    for name, result in results.items():
        print(f"{name:20} | 테스트: {result['accuracy']:.4f} | CV: {result['cv_mean']:.4f}±{result['cv_std']:.4f}")
    
    # 최고 성능 모델 선택
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_result = results[best_model_name]
    
    print(f"\n🥇 최고 성능 모델: {best_model_name}")
    print(f"   테스트 정확도: {best_result['accuracy']:.4f}")
    
    # 상세 분류 보고서
    print(f"\n📊 {best_model_name} 상세 평가:")
    y_pred = best_result['predictions']
    
    # 클래스 이름으로 변환
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    print("\n분류 보고서:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    return best_result['model'], best_model_name

def visualize_results(results, y_test, label_encoder):
    """
    결과 시각화
    """
    print("\n📈 결과 시각화 중...")
    
    # 1. 모델 성능 비교
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 성능 비교 바 차트
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    cv_means = [results[name]['cv_mean'] for name in model_names]
    
    axes[0, 0].bar(model_names, accuracies, alpha=0.7, label='Test Accuracy')
    axes[0, 0].bar(model_names, cv_means, alpha=0.7, label='CV Mean')
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 최고 성능 모델의 혼동 행렬
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    y_pred = results[best_model_name]['predictions']
    
    # 클래스 이름으로 변환
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    
    axes[0, 1].remove()
    axes[1, 0].remove() 
    axes[1, 1].remove()
    
    # 혼동 행렬을 큰 서브플롯에 그리기
    ax_cm = plt.subplot(2, 2, (2, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_,
                ax=ax_cm)
    ax_cm.set_title(f'{best_model_name} - Confusion Matrix')
    ax_cm.set_ylabel('True Label')
    ax_cm.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()

def main():
    # 파일 경로 설정
    pkl_path = 'C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/github_profiles_with_bert_processed.pkl'
    
    try:
        # 1. 데이터 로드 및 탐색
        df, basic_cols, bert_name_cols, bert_desc_cols, stack_counts = load_and_explore_data(pkl_path)
        
        if stack_counts is None:
            return
        
        # 2. 피처 준비
        X, feature_columns, language_cols = prepare_features(df, basic_cols, bert_name_cols, bert_desc_cols)
        
        # 3. 타겟 준비
        y, valid_mask, label_encoder = prepare_target(df)
        
        # 유효한 데이터만 선택
        X = X.loc[valid_mask].reset_index(drop=True)
        
        print(f"\n📊 최종 데이터 크기: {X.shape}")
        print(f"타겟 데이터 크기: {len(y)}")
        
        # 4. 데이터 분할
        print("\n🔀 데이터 분할 중...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"학습 데이터: {X_train.shape}")
        print(f"테스트 데이터: {X_test.shape}")
        
        # 5. 피처 스케일링 (BERT 벡터는 이미 정규화되어 있지만, 언어 통계는 스케일링 필요)
        print("\n⚖️ 피처 스케일링 중...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # DataFrame 형태로 변환
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # 6. 피처 선택 (선택사항 - 너무 많은 피처가 있을 때)
        if X_train_scaled.shape[1] > 1000:
            X_train_selected, selected_features, selector = feature_selection(X_train_scaled, y_train, k=1000)
            X_test_selected = selector.transform(X_test_scaled)
            
            # DataFrame으로 변환
            X_train_final = pd.DataFrame(X_train_selected, columns=selected_features)
            X_test_final = pd.DataFrame(X_test_selected, columns=selected_features)
        else:
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled
        
        # 7. 모델 학습 및 평가
        results = train_models(X_train_final, y_train, X_test_final, y_test)
        
        # 8. 최고 모델 평가
        best_model, best_model_name = evaluate_best_model(results, X_test_final, y_test, label_encoder)
        
        # 9. 결과 시각화
        visualize_results(results, y_test, label_encoder)
        
        # 10. 모델 저장
        import pickle
        model_save_path = pkl_path.replace('.pkl', '_best_model.pkl')
        
        model_data = {
            'model': best_model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_columns': feature_columns,
            'selected_features': X_train_final.columns.tolist() if 'selector' in locals() else None,
            'selector': selector if 'selector' in locals() else None,
            'model_name': best_model_name
        }
        
        with open(model_save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 최고 모델 저장 완료: {model_save_path}")
        print(f"모델: {best_model_name}")
        print(f"성능: {results[best_model_name]['accuracy']:.4f}")
        
        print("\n🎉 스택 분류 모델 학습 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()