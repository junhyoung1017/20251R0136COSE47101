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

# TensorFlow/Keras 임포트
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    print(f"✅ TensorFlow {tf.__version__} 로드 완료")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("❌ TensorFlow를 설치해주세요: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

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

def create_neural_network(input_dim, num_classes, model_type='basic'):
    """
    딥러닝 모델 생성
    """
    if model_type == 'basic':
        # 기본 피드포워드 네트워크
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
    elif model_type == 'deep':
        # 더 깊은 네트워크
        model = keras.Sequential([
            layers.Dense(1024, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
    elif model_type == 'multi_input':
        # 다중 입력 모델 (언어통계 + BERT 임베딩 분리)
        # 언어 통계 입력
        lang_input = keras.Input(shape=(20,), name='language_stats')  # 언어 피처 수에 맞게 조정
        lang_dense = layers.Dense(64, activation='relu')(lang_input)
        lang_dense = layers.Dropout(0.2)(lang_dense)
        
        # BERT 임베딩 입력
        bert_input = keras.Input(shape=(768,), name='bert_embeddings')  # BERT 임베딩 차원
        bert_dense = layers.Dense(256, activation='relu')(bert_input)
        bert_dense = layers.BatchNormalization()(bert_dense)
        bert_dense = layers.Dropout(0.3)(bert_dense)
        bert_dense = layers.Dense(128, activation='relu')(bert_dense)
        bert_dense = layers.Dropout(0.2)(bert_dense)
        
        # 두 입력 결합
        combined = layers.concatenate([lang_dense, bert_dense])
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.3)(combined)
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.Dropout(0.2)(combined)
        
        output = layers.Dense(num_classes, activation='softmax')(combined)
        
        model = keras.Model(inputs=[lang_input, bert_input], outputs=output)
    
    return model

def train_neural_networks(X_train, y_train, X_val, y_val, num_classes):
    """
    여러 딥러닝 모델 학습
    """
    print("\n🧠 딥러닝 모델 학습 중...")
    
    # 타겟을 원-핫 인코딩
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    
    # 콜백 정의
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [early_stopping, reduce_lr]
    
    dl_results = {}
    
    # 1. 기본 신경망
    print("\n🔄 기본 신경망 학습 중...")
    basic_model = create_neural_network(X_train.shape[1], num_classes, 'basic')
    basic_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_basic = basic_model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )
    
    # 평가
    val_loss, val_acc = basic_model.evaluate(X_val, y_val_cat, verbose=0)
    dl_results['Neural Network (Basic)'] = {
        'model': basic_model,
        'accuracy': val_acc,
        'history': history_basic
    }
    print(f"✅ 기본 신경망 완료! 검증 정확도: {val_acc:.4f}")
    
    # 2. 깊은 신경망
    print("\n🔄 깊은 신경망 학습 중...")
    deep_model = create_neural_network(X_train.shape[1], num_classes, 'deep')
    deep_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_deep = deep_model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )
    
    val_loss, val_acc = deep_model.evaluate(X_val, y_val_cat, verbose=0)
    dl_results['Neural Network (Deep)'] = {
        'model': deep_model,
        'accuracy': val_acc,
        'history': history_deep
    }
    print(f"✅ 깊은 신경망 완료! 검증 정확도: {val_acc:.4f}")
    
    return dl_results

'''def train_traditional_models(X_train, y_train, X_test, y_test):
    """
    전통적인 머신러닝 모델 학습
    """
    print("\n🤖 전통적인 ML 모델 학습 중...")
    
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
    
    return results'''

def evaluate_all_models(ml_results, dl_results, X_test, y_test, label_encoder):
    """
    모든 모델 성능 비교 및 평가
    """
    print("\n🏆 전체 모델 성능 비교:")
    print("-" * 70)
    
    all_results = {}
    
    '''# 전통적인 ML 모델 결과
    for name, result in ml_results.items():
        all_results[name] = result['accuracy']
        print(f"{name:25} | 테스트: {result['accuracy']:.4f} | CV: {result['cv_mean']:.4f}±{result['cv_std']:.4f}")'''
    
    # 딥러닝 모델 결과
    if TENSORFLOW_AVAILABLE and dl_results:
        for name, result in dl_results.items():
            all_results[name] = result['accuracy']
            print(f"{name:25} | 검증: {result['accuracy']:.4f}")
    
    # 최고 성능 모델 선택
    best_model_name = max(all_results.keys(), key=lambda x: all_results[x])
    best_accuracy = all_results[best_model_name]
    
    print(f"\n🥇 최고 성능 모델: {best_model_name}")
    print(f"   정확도: {best_accuracy:.4f}")
    
    # 최고 모델의 상세 평가
    if best_model_name in ml_results:
        # 전통적인 ML 모델
        best_result = ml_results[best_model_name]
        y_pred = best_result['predictions']
        
        # 클래스 이름으로 변환
        y_test_labels = label_encoder.inverse_transform(y_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        
        print(f"\n📊 {best_model_name} 상세 평가:")
        print("\n분류 보고서:")
        print(classification_report(y_test_labels, y_pred_labels))
        
        return ml_results[best_model_name]['model'], best_model_name
    
    else:
        # 딥러닝 모델
        best_model = dl_results[best_model_name]['model']
        
        # 딥러닝 모델 예측
        y_pred_proba = best_model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 클래스 이름으로 변환
        y_test_labels = label_encoder.inverse_transform(y_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        
        print(f"\n📊 {best_model_name} 상세 평가:")
        print("\n분류 보고서:")
        print(classification_report(y_test_labels, y_pred_labels))
        
        return best_model, best_model_name

def visualize_results(ml_results, dl_results, y_test, label_encoder):
    """
    결과 시각화
    """
    print("\n📈 결과 시각화 중...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 모든 모델 성능 비교
    all_models = list(ml_results.keys())
    all_accuracies = [ml_results[name]['accuracy'] for name in all_models]
    
    if TENSORFLOW_AVAILABLE and dl_results:
        dl_models = list(dl_results.keys())
        dl_accuracies = [dl_results[name]['accuracy'] for name in dl_models]
        all_models.extend(dl_models)
        all_accuracies.extend(dl_accuracies)
    
    colors = ['skyblue'] * len(ml_results) + ['lightcoral'] * len(dl_results)
    
    bars = axes[0, 0].bar(range(len(all_models)), all_accuracies, color=colors, alpha=0.7)
    axes[0, 0].set_title('All Models Performance Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(range(len(all_models)))
    axes[0, 0].set_xticklabels(all_models, rotation=45, ha='right')
    
    # 범례 추가
    import matplotlib.patches as mpatches
    ml_patch = mpatches.Patch(color='skyblue', label='Traditional ML')
    dl_patch = mpatches.Patch(color='lightcoral', label='Deep Learning')
    axes[0, 0].legend(handles=[ml_patch, dl_patch])
    
    # 2. 딥러닝 학습 곡선 (첫 번째 모델)
    if TENSORFLOW_AVAILABLE and dl_results:
        first_dl_model = list(dl_results.keys())[0]
        history = dl_results[first_dl_model]['history']
        
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title(f'{first_dl_model} - Learning Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Deep Learning Results', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # 3. 최고 성능 모델의 혼동 행렬
    best_model_name = max(all_models, key=lambda x: 
                         ml_results[x]['accuracy'] if x in ml_results 
                         else dl_results[x]['accuracy'])
    
    if best_model_name in ml_results:
        y_pred = ml_results[best_model_name]['predictions']
    else:
        # 딥러닝 모델의 예측값 계산 필요 (여기서는 임시로 처리)
        y_pred = y_test  # 실제로는 모델 예측값을 사용해야 함
    
    # 클래스 이름으로 변환
    y_test_labels = label_encoder.inverse_transform(y_test)
    if best_model_name in ml_results:
        y_pred_labels = label_encoder.inverse_transform(y_pred)
    else:
        y_pred_labels = y_test_labels  # 임시 처리
    
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    
    # 3,4번 subplot을 합쳐서 혼동 행렬 그리기
    axes[1, 0].remove()
    axes[1, 1].remove()
    ax_cm = plt.subplot(2, 2, (3, 4))
    
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
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
        
        print(f"학습 데이터: {X_train.shape}")
        print(f"검증 데이터: {X_val.shape}")
        print(f"테스트 데이터: {X_test.shape}")
        
        # 5. 피처 스케일링
        print("\n⚖️ 피처 스케일링 중...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 6. 전통적인 ML 모델 학습
        ml_results = train_traditional_models(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # 7. 딥러닝 모델 학습
        dl_results = {}
        if TENSORFLOW_AVAILABLE:
            dl_results = train_neural_networks(X_train_scaled, y_train, X_val_scaled, y_val, len(label_encoder.classes_))
        else:
            print("\n⚠️ TensorFlow가 설치되지 않아 딥러닝 모델을 건너뜁니다.")
        
        # 8. 모든 모델 평가
        best_model, best_model_name = evaluate_all_models(ml_results, dl_results, X_test_scaled, y_test, label_encoder)
        
        # 9. 결과 시각화
        visualize_results(ml_results, dl_results, y_test, label_encoder)
        
        # 10. 모델 저장
        import pickle
        model_save_path = pkl_path.replace('.pkl', '_best_model_with_dl.pkl')
        
        model_data = {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_columns': feature_columns,
            'ml_results': ml_results,
            'dl_results': dl_results if TENSORFLOW_AVAILABLE else None
        }
        
        with open(model_save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 최고 모델 저장 완료: {model_save_path}")
        print(f"모델: {best_model_name}")
        
        if best_model_name in ml_results:
            print(f"성능: {ml_results[best_model_name]['accuracy']:.4f}")
        else:
            print(f"성능: {dl_results[best_model_name]['accuracy']:.4f}")
        
        print("\n🎉 스택 분류 모델 학습 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()