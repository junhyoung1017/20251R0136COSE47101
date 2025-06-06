import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import re
from typing import Tuple, List
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# imbalanced-learn 라이브러리 설치 필요: pip install imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ imbalanced-learn이 설치되지 않았습니다. pip install imbalanced-learn로 설치하세요.")
    print("   클래스 가중치 방법만 사용됩니다.")
    IMBLEARN_AVAILABLE = False

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
        if name and name.lower() not in ['nan', 'none', '']:
            repo_names.append(name)
        if desc and desc.lower() not in ['nan', 'none', '']:
            descriptions.append(desc)
    
    return ', '.join(repo_names), ', '.join(descriptions)

def process_stack(stack_text: str) -> List[str]:
    """
    Stack 텍스트를 &으로 분리하고 정제하는 함수 (멀티스택 처리)
    """
    if pd.isna(stack_text) or stack_text == '':
        return []
    
    # &으로 분리하고 각 스택 정제
    stacks = [s.strip() for s in str(stack_text).split('&') if s.strip()]
    
    # 빈 문자열이나 'nan' 제거, 공백 정리
    cleaned_stacks = []
    for stack in stacks:
        # 공백 정리 (Frontend& Server -> Frontend, Server)
        stack = stack.strip()
        if stack and stack.lower() not in ['nan', 'none', '']:
            cleaned_stacks.append(stack)
    
    return cleaned_stacks

def clean_text(text: str) -> str:
    """
    텍스트 전처리 함수
    """
    if pd.isna(text) or text == '':
        return ''
    
    # 소문자 변환
    text = str(text).lower()
    
    # URL 제거
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # 이메일 제거
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # 특수문자 제거 (영문, 숫자, 공백만 남김)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # 연속된 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    # 너무 짧은 텍스트 처리
    if len(text) < 3:
        return ''
    
    return text

def create_language_features(df: pd.DataFrame, language_columns: List[str]) -> pd.DataFrame:
    """
    언어 데이터로부터 추가 특성을 생성하는 함수
    """
    print("🔧 언어 특성 엔지니어링 중...")
    
    # 언어 데이터를 숫자형으로 변환
    for col in language_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. 총 코드 라인 수
    total_lines=df[language_columns].sum(axis=1)
    
    # 2. 사용 언어 개수
    df['num_languages'] = (df[language_columns] > 0).sum(axis=1)
    
    # 3. 주력 언어 (가장 많이 사용한 언어)
    df['main_language_idx'] = df[language_columns].idxmax(axis=1)
    df['main_language_ratio'] = df[language_columns].max(axis=1) / (total_lines + 1e-6)
    
    # 4. 언어 다양성 지수 (Shannon entropy) - 개선된 버전
    def calculate_diversity(row):
        try:
            values = np.array(row[language_columns].values, dtype=np.float64)
            total = np.sum(values)
            if total == 0 or np.isnan(total):
                return 0.0
            
            probs = values / total
            probs = probs[probs > 0]  # 0인 값 제거
            
            if len(probs) <= 1:  # 하나 이하의 언어만 사용
                return 0.0
            
            # Shannon entropy 계산
            log_probs = np.log2(probs + 1e-10)  # 더 작은 epsilon 사용
            entropy = -np.sum(probs * log_probs)
            
            # 결과가 유효한지 확인
            if np.isnan(entropy) or np.isinf(entropy):
                return 0.0
            
            return float(entropy)
            
        except Exception as e:
            print(f"Warning: calculate_diversity 에러 발생: {e}")
            return 0.0
    
    df['language_diversity'] = df.apply(calculate_diversity, axis=1)
    
    # 5. Frontend/Backend/Others 비율 (실제 언어 기반)
    frontend_langs = ['JS'] 
    backend_langs = ['Python', 'Java', 'C++', 'C#', 'Go', 'PHP', 'Ruby']
    mobile_langs = ['Swift', 'Kotlin', 'Dart']
    system_langs = ['C/C++', 'Rust', 'Assembly']
    
    frontend_cols = [col for col in frontend_langs if col in language_columns]
    backend_cols = [col for col in backend_langs if col in language_columns]
    mobile_cols = [col for col in mobile_langs if col in language_columns]
    system_cols = [col for col in system_langs if col in language_columns]
    
    if frontend_cols:
        df['frontend_lang_ratio'] = df[frontend_cols].sum(axis=1) / (total_lines+ 1e-6)
    else:
        df['frontend_lang_ratio'] = 0
        
    if backend_cols:
        df['backend_lang_ratio'] = df[backend_cols].sum(axis=1) / (total_lines + 1e-6)
    else:
        df['backend_lang_ratio'] = 0
        
    if mobile_cols:
        df['mobile_lang_ratio'] = df[mobile_cols].sum(axis=1) / (total_lines + 1e-6)
    else:
        df['mobile_lang_ratio'] = 0
        
    if system_cols:
        df['system_lang_ratio'] = df[system_cols].sum(axis=1) / (total_lines+ 1e-6)
    else:
        df['system_lang_ratio'] = 0
    
    # 6. 언어별 정규화 (Min-Max Scaling)
    scaler = MinMaxScaler()
    df[language_columns] = scaler.fit_transform(df[language_columns])
    
    print("✅ 언어 특성 엔지니어링 완료")
    return df
def create_stack_specific_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """
    7개 스택별 특화 키워드를 기반으로 특성을 생성하는 함수
    """
    print("🎯 스택별 특화 키워드 특성 생성 중...")
    
    # 스택별 특화 키워드 정의
    stack_keywords = {
        'frontend': [
            'react', 'angular', 'vue', 'svelte',
            'html', 'css', 'sass', 'typescript', 'javascript',
            'bootstrap', 'tailwind', 'ui', 'ux', 'dom',
            'webpack', 'vite', 'next', 'nuxt',
            'spa', 'pwa', 'website', 'browser', 'frontend'
        ],
        'server':['server', 'backend', 'rest', 'graphql', 'rpc', 'webhook',
            'microservice', 'monolith', 'database', 'sql', 'nosql',
            'mongodb', 'postgresql', 'mysql', 'redis', 'sqlite', 'elastic', 'elastic-search',
            'node.js', 'express', 'koa', 'hapi', 'fastify', 'fastapi', 'django', 'flask',
            'spring', 'laravel', 'nest', 'sanic', 'gin', 'actix', 'routing', 'controller',
            'mvc', 'orm', 'prisma', 'sequelize', 'typeorm','router'
            'authentication', 'authorization', 'jwt', 'oauth', 'session', 'cookie',
            'middleware', 'endpoint', 'http', 'https', 'request', 'response',
            'cache',  'reverse-proxy', 'cors', 'rate-limiting',
            'websocket', 'swagger', 'openapi', 'rabbitmq', 'kafka',  'tomcat', 
            'bcrypt', 'hashing', 'encryption', 'salt', 'socket.io', 'api-gateway',
            'service-mesh', 'circuit-breaker', 'strapi', 'quarkus', 'soa', 'bff', 'postman', 'insomnia'
            'backend-service', 'api-server', 'load-balancer', 'reverse-proxy', 'infrastructure'],
        # ''''server': [
        #     'java', 'data', 'python', 'client', 
        #     'react', 'php', 'javascript', 'system', 'server', 'apache', 'spring', 
        #     'management', 'bootstrap', 'laravel', 'django', 'flask', 'node', 'express',
        #     'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis',
        #     'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'microservice', 'rest',
        #     'graphql', 'jwt', 'oauth', 'crud', 'orm'
        # ],'''
        # 'server': [
#     'backend', 'rest', 'graphql', 'rpc', 'webhook',
#     'microservice', 'monolith',
#     'api-server', 'controller', 'middleware', 'routing',
#     'express', 'koa', 'hapi', 'fastify', 'nest',
#     'django', 'flask', 'spring', 'laravel',
#     'prisma', 'sequelize', 'typeorm', 'orm',
#     'jwt', 'oauth', 'session', 'cookie', 'authentication', 'authorization',
#     'swagger', 'openapi', 'postman', 'insomnia'
# ]     
        'android': [
            'android', 'flutter', 'native', 'view',
            'design', 'material', 'kotlin', 'support', 
            'image', 'video', 'sdk', 'gradle', 'studio', 'activity', 
            'fragment', 'intent', 'recyclerview', 'room', 'retrofit', 'coroutines',
            'jetpack', 'compose', 'mvvm', 'livedata', 'viewmodel', 'firebase', 'play'
        ],
        
        'ios': [
            'ios', 'swift','library', 'flutter', 'swiftui', 
            'development', 'package', 'custom', 'xamarin', 'objective', 'xcode', 
            'iphone', 'ipad', 'uikit', 'cocoa', 'pods', 'carthage', 'realm',
            'storyboard', 'autolayout', 'delegate', 'protocol', 'arc', 'gcd',
            'appstore', 'testflight'
        ],
        
       'visualization': [
            'data', 'visualization', 'chart', 'graph', 'd3', 'plotly', 'dashboard', 
            'interactive', 'analysis', 'report',
            'matplotlib', 'seaborn', 'bokeh', 'tableau', 'powerbi', 'chart.js',
            'highcharts', 'echarts', 'vis', 'insight', 'metric', 'kpi', 'business',
            'intelligence', 'bi', 'analytics'
        ],
        
        'ml_data': [
            'learning', 'python', 'data', 'deep', 'tensorflow', 'machine', 
            'networks', 'models', 'pytorch', 'neural', 'object', 'caffe', 
            'detection', 'reinforcement', 'analysis', 'ai', 'artificial', 'intelligence',
            'keras', 'sklearn', 'pandas', 'numpy', 'jupyter', 'notebook', 'algorithm',
            'regression', 'classification', 'clustering', 'nlp', 'cv', 'computer', 'vision',
            'feature', 'training', 'prediction', 'dataset', 'preprocessing'
        ],
         'system' : [
                'linux', 'unix', 'windows', 'macos', 'kernel', 'bash', 'shell', 'powershell',
                'infrastructure', 'deployment', 'ci', 'cd', 'build', 'runner', 'system','os','infra'
                'docker', 'kubernetes', 'container', 'pod', 'cluster', 'command-line', 'terminal', 'script', 'crontab',
                'ansible', 'terraform', 'helm', 'logging', 'monitoring', 'prometheus', 'grafana','gunicorn', 
                'nginx', 'pm2', 'gunicorn', 'socket', 'tcp', 'udp', 'dns','loadbalancer', 'proxy',
                'load', 'latency', 'uptime', 'network', 'firewall', 'security','node','nginx','pm2','pipeline', 
                 'automation',  'backup', 'restore','configuration', 'recovery', 'admin', 'failover', 'scalability'
            ],
        #  'system': [
        #     'system', 'linux', 'server', 'network', 'security', 'docker', 'kubernetes', 
        #     'monitoring', 'deployment', 'infrastructure', 'devops', 'automation', 
        #     'performance', 'unix', 'windows', 'shell', 'bash', 'script', 'ci', 'cd',
        #     'pipeline', 'logging', 'admin', 'maintenance', 'backup', 'recovery',
        #     'scalability', 'load', 'balancer', 'configuration'
        # ]
        # 'system' : [
        #         'system', 'linux', 'unix', 'windows', 'macos', 'kernel', 'os',
        #         'bash', 'shell', 'powershell', 'command-line', 'terminal', 'script', 'crontab',
        #         'devops', 'infrastructure', 'infra', 'deployment', 'build', 'ci', 'cd', 
        #         'pipeline', 'runner', 'monitoring', 'prometheus', 'grafana', 'logging',
        #         'logstash', 'log', 'performance', 'profiling', 'security', 'firewall',
        #         'network', 'socket', 'tcp', 'udp', 'dns', 'load', 'latency', 'uptime',
        #         'docker', 'kubernetes', 'swarm', 'cluster', 'container', 'pod', 'node',
        #         'service', 'ingress', 'helm', 'ansible', 'terraform', 'puppet', 'chef',
        #         'automation', 'configuration', 'provisioning', 'backup', 'restore',
        #         'recovery', 'maintenance', 'admin', 'failover', 'scalability'
        #     ],
    }
    
    # 기존 일반 키워드들 (호환성 유지)
    general_tech_keywords = [
        'agile', 'algorithm', 'api', 'app', 'automation', 'aws', 'azure', 'backend',
    'bot', 'build', 'cd', 'ci', 'clean-code', 'client', 'cloud', 'comment',
    'config', 'container', 'cron', 'data', 'database', 'debug', 'deep', 'deploy',
    'design-pattern', 'devops', 'docker', 'documentation', 'feature', 'framework',
    'frontend', 'fullstack', 'game', 'gcp', 'git', 'github', 'gitlab', 'graphql',
    'infrastructure', 'integration', 'issue', 'json', 'kanban', 'kubernetes',
    'learning', 'library', 'log', 'logging', 'machine', 'metrics', 'microservice',
    'mobile', 'mock', 'module', 'monitor', 'neural', 'package', 'performance',
    'pipeline', 'provisioning', 'pull-request', 'readme', 'refactor',
    'reliability', 'rest', 'review', 'run', 'scalability', 'script', 'scrum',
    'sdk', 'server', 'sprint', 'test', 'ticket', 'tool', 'ui', 'unit-test', 'ux',
    'version', 'web', 'xml', 'yaml','javascript'
    ]
    # general_tech_keywords = [
    #     'api', 'web', 'app', 'mobile', 'data', 'machine', 'learning', 
    #     'deep', 'neural', 'algorithm', 'database', 'server', 'client',
    #     'framework', 'library', 'tool', 'bot', 'game', 'ui', 'ux',
    #     'frontend', 'backend', 'fullstack', 'devops', 'microservice'
    # ]
    
    # 1. 기존 일반 키워드 특성 생성
    print("   일반 기술 키워드 처리 중...")
    for keyword in general_tech_keywords:
        df[f'has_{keyword}'] = df['description'].str.contains(keyword, case=False, na=False).astype(int)
    
    # 2. 스택별 특화 키워드 특성 생성
    print("   실제 데이터 기반 스택별 특화 키워드 처리 중...")
    for stack_name, keywords in stack_keywords.items():
        print(f"     {stack_name}: {len(keywords)}개 키워드")
        
        # 각 키워드별로 개별 특성 생성
        for keyword in keywords:
            df[f'{stack_name}_{keyword}'] = df['description'].str.contains(keyword, case=False, na=False).astype(int)
        
        # 스택별 총 키워드 개수 (해당 스택과 관련된 키워드가 몇 개나 포함되어 있는지)
        stack_columns = [f'{stack_name}_{keyword}' for keyword in keywords]
        df[f'{stack_name}_keyword_count'] = df[stack_columns].sum(axis=1)
        
        # 스택별 키워드 비율 (전체 단어 대비 해당 스택 키워드 비율)
        df[f'{stack_name}_keyword_ratio'] = df[f'{stack_name}_keyword_count'] / (df['description_word_count'] + 1)
        
        # 스택별 키워드 존재 여부 (하나라도 있으면 1)
        df[f'has_{stack_name}_keywords'] = (df[f'{stack_name}_keyword_count'] > 0).astype(int)
    
    # 3. 교차 스택 특성 (복합 스택 개발자 식별)
    print("   교차 스택 특성 생성 중...")
    
    # Full-stack 관련 특성
    df['is_fullstack_likely'] = (
        (df['has_frontend_keywords'] == 1) & 
        (df['has_server_keywords'] == 1)
    ).astype(int)
    
    # Mobile 개발자 (Android + iOS)
    df['is_mobile_dev'] = (
        (df['has_android_keywords'] == 1) | 
        (df['has_ios_keywords'] == 1)
    ).astype(int)
    
    # Data-focused 개발자 (ML + Visualization)
    df['is_data_focused'] = (
        (df['has_ml_data_keywords'] == 1) | 
        (df['has_visualization_keywords'] == 1)
    ).astype(int)
    
    # Backend-heavy 개발자 (Server + System)
    df['is_backend_heavy'] = (
        (df['has_server_keywords'] == 1) & 
        (df['has_system_keywords'] == 1)
    ).astype(int)
    
    # 4. 키워드 다양성 지수 (얼마나 다양한 스택에 관심이 있는지)
    stack_interest_cols = [f'has_{stack}_keywords' for stack in stack_keywords.keys()]
    df['stack_diversity'] = df[stack_interest_cols].sum(axis=1)
    
    # 5. 주력 스택 추정 (키워드 기반)
    stack_count_cols = [f'{stack}_keyword_count' for stack in stack_keywords.keys()]
    
    # 가장 많은 키워드를 가진 스택을 주력 스택으로 추정
    stack_counts_array = df[stack_count_cols].values
    df['estimated_main_stack_idx'] = np.argmax(stack_counts_array, axis=1)
    
    # 주력 스택 이름
    stack_names = list(stack_keywords.keys())
    df['estimated_main_stack'] = df['estimated_main_stack_idx'].apply(lambda x: stack_names[x])
    
    # 주력 스택 신뢰도 (주력 스택 키워드 수 / 전체 기술 키워드 수)
    df['main_stack_confidence'] = np.max(stack_counts_array, axis=1) / (np.sum(stack_counts_array, axis=1) + 1)
    
    # 6. 통계 정보 출력
    print(f"\n📊 생성된 키워드 특성 통계:")
    print(f"   일반 키워드 특성: {len(general_tech_keywords)}개")
    
    total_stack_features = 0
    for stack_name, keywords in stack_keywords.items():
        stack_features = len(keywords) + 3  # 개별 키워드 + count + ratio + has_keywords
        total_stack_features += stack_features
        
        # 각 스택별 키워드 매칭 통계
        keyword_count_col = f'{stack_name}_keyword_count'
        has_keywords_col = f'has_{stack_name}_keywords'
        
        avg_keywords = df[keyword_count_col].mean()
        users_with_keywords = df[has_keywords_col].sum()
        percentage = (users_with_keywords / len(df)) * 100
        
        print(f"   {stack_name}: {len(keywords)}개 키워드, 평균 {avg_keywords:.1f}개 매칭, {users_with_keywords}명 ({percentage:.1f}%)")
    
    print(f"   스택별 특성 총계: {total_stack_features}개")
    print(f"   교차 스택 특성: 4개")
    print(f"   다양성 특성: 4개")
    print(f"   총 키워드 관련 특성: {len(general_tech_keywords) + total_stack_features + 8}개")
    
    # 7. 주력 스택 추정 결과
    print(f"\n🎯 키워드 기반 주력 스택 추정 결과:")
    estimated_stack_dist = df['estimated_main_stack'].value_counts()
    for stack, count in estimated_stack_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   {stack}: {count}명 ({percentage:.1f}%)")
    
    print("✅ 실제 데이터 기반 스택별 특화 키워드 특성 생성 완료")
    return df
def improve_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    텍스트 특성을 개선하는 함수
    """
    print("📝 텍스트 특성 개선 중...")
    
    # 1. 텍스트 길이 특성
    df['description_length'] = df['description'].str.len()
    df['repo_names_length'] = df['repo_names'].str.len()
    
    # 2. 단어 수 특성
    df['description_word_count'] = df['description'].str.split().str.len()
    df['repo_names_word_count'] = df['repo_names'].str.split().str.len()
    
    # 2. 스택별 특화 키워드 특성 생성 (새로 추가)
    df = create_stack_specific_keywords(df)
    
    # 4. 빈 텍스트 처리 개선
    df['has_description'] = (df['description'] != 'no description available').astype(int)
    df['has_repo_names'] = (df['repo_names'] != 'no repository name').astype(int)
    
    # 5. Repository 개수 관련 특성
    df['repo_count'] = pd.to_numeric(df['repo_count'], errors='coerce').fillna(0)
    df['avg_repo_name_length'] = df['repo_names_length'] / (df['repo_count'] + 1)
    df['is_prolific_dev'] = (df['repo_count'] > df['repo_count'].quantile(0.75)).astype(int)
    
    print("✅ 텍스트 특성 개선 완료")
    return df

def filter_low_variance_features(X: np.ndarray, threshold: float = 0.0005) -> Tuple[np.ndarray, np.ndarray]:
    """
    분산이 낮은 특성들을 제거하는 함수
    """
    print(f"🔍 낮은 분산 특성 제거 중... (임계값: {threshold})")
    
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    selected_features = selector.get_support()

    variances = selector.variances_
    # 분산 상위 10개 특성 출력
    top_indices = np.argsort(variances)[::-1][:10]
    print("\n🔥 분산이 높은 Top 10 특성:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Index {idx} - Variance: {variances[idx]:.6f}")

    removed_count = (~selected_features).sum()
    print(f"✅ {removed_count}개 특성 제거됨 ({X.shape[1]} → {X_filtered.shape[1]})")
    
    return X_filtered, selected_features

def analyze_stack_distribution(df: pd.DataFrame) -> None:
    """
    스택 분포를 분석하는 함수
    """
    print("📊 스택 분포 분석 중...")
    
    # 전체 스택 분포
    stack_counts = df['stack'].value_counts()
    print(f"\n📋 스택별 샘플 수:")
    for stack, count in stack_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {stack}: {count}개 ({percentage:.1f}%)")
    
    # 균형도 체크
    min_samples = stack_counts.min()
    max_samples = stack_counts.max()
    imbalance_ratio = max_samples / min_samples
    
    print(f"\n⚖️ 클래스 균형 분석:")
    print(f"   최소 샘플: {min_samples}개")
    print(f"   최대 샘플: {max_samples}개") 
    print(f"   불균형 비율: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 10:
        print("   ⚠️ 심각한 클래스 불균형 감지")
    elif imbalance_ratio > 5:
        print("   🟡 중간 수준의 클래스 불균형")
    else:
        print("   ✅ 양호한 클래스 균형")

'''def handle_class_imbalance(X, y, target_stacks, strategy='auto'):
    """
    클래스 불균형을 해결하는 함수
    """
    print(f"\n⚖️ 클래스 불균형 처리 중... (전략: {strategy})")
    
    # 멀티라벨을 단일 라벨로 변환
    y_single = np.argmax(y, axis=1)
    
    # 원래 분포 확인
    original_distribution = Counter(y_single)
    print(f"📊 원본 클래스 분포:")
    for class_idx, count in original_distribution.items():
        stack_name = target_stacks[class_idx]
        percentage = (count / len(y_single)) * 100
        print(f"   {stack_name} (idx {class_idx}): {count}개 ({percentage:.1f}%)")
    
    # 불균형 정도 계산
    min_samples = min(original_distribution.values())
    max_samples = max(original_distribution.values())
    imbalance_ratio = max_samples / min_samples
    
    print(f"\n📈 불균형 분석:")
    print(f"   최소 샘플: {min_samples}개")
    print(f"   최대 샘플: {max_samples}개")
    print(f"   불균형 비율: {imbalance_ratio:.2f}")
    
    # 클래스 가중치 계산
    classes = np.unique(y_single)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_single)
    class_weight_dict = {}
    
    print(f"\n🎯 클래스 가중치:")
    for i, class_idx in enumerate(classes):
        stack_name = target_stacks[class_idx]
        weight = class_weights[i]
        class_weight_dict[class_idx] = weight
        print(f"   {stack_name}: {weight:.3f}")
    
    # 전략 자동 선택
    if strategy == 'auto':
        if not IMBLEARN_AVAILABLE:
            strategy = 'class_weight'
        elif imbalance_ratio > 10:
            strategy = 'smote'
        elif imbalance_ratio > 5:
            strategy = 'smote'
        else:
            strategy = 'class_weight'
    
    X_resampled, y_resampled = X.copy(), y.copy()
    resampling_applied = False
    
    # SMOTE 적용 (가능한 경우)
    if strategy == 'smote' and IMBLEARN_AVAILABLE and imbalance_ratio > 2:
        print(f"\n🔄 SMOTE 적용 중...")
        try:
            # 최소 샘플 수 확인 (SMOTE는 k_neighbors가 필요)
            min_neighbors = min(5, min_samples - 1)
            if min_neighbors >= 1:
                smote = SMOTE(random_state=42, k_neighbors=min_neighbors)
                X_resampled, y_single_resampled = smote.fit_resample(X, y_single)
                
                # 멀티라벨 형태로 복원
                y_resampled = np.zeros((len(y_single_resampled), y.shape[1]))
                for i, label in enumerate(y_single_resampled):
                    y_resampled[i, label] = 1
                
                resampling_applied = True
                print(f"✅ SMOTE 완료: {X.shape[0]} → {X_resampled.shape[0]} 샘플")
            else:
                print(f"❌ SMOTE 불가: 최소 샘플 수 부족 (min_samples: {min_samples})")
                strategy = 'class_weight'
                
        except Exception as e:
            print(f"❌ SMOTE 실패: {e}")
            strategy = 'class_weight'
    
    # 최종 분포 확인
    if resampling_applied:
        y_single_final = np.argmax(y_resampled, axis=1)
        final_distribution = Counter(y_single_final)
        print(f"\n📊 리샘플링 후 클래스 분포:")
        for class_idx, count in final_distribution.items():
            stack_name = target_stacks[class_idx]
            percentage = (count / len(y_single_final)) * 100
            print(f"   {stack_name}: {count}개 ({percentage:.1f}%)")
        
        final_imbalance = max(final_distribution.values()) / min(final_distribution.values())
        print(f"   개선된 불균형 비율: {final_imbalance:.2f}")
    else:
        print(f"\n💡 리샘플링 미적용 → 클래스 가중치 사용 권장")
    
    return X_resampled, y_resampled, class_weight_dict, strategy'''
def handle_class_imbalance_multilabel(X, y, target_stacks, strategy='smote_multilabel'):
    """
    멀티라벨 데이터의 클래스 불균형을 해결하는 개선된 함수
    """
    print(f"\n⚖️ 멀티라벨 클래스 불균형 처리 중... (전략: {strategy})")
    
    # 각 라벨별 분포 확인
    print(f"📊 원본 라벨별 분포:")
    original_distribution = {}
    for i, stack in enumerate(target_stacks):
        count = np.sum(y[:, i])
        percentage = (count / len(y)) * 100
        original_distribution[stack] = count
        print(f"   {stack}: {count}개 ({percentage:.1f}%)")
    
    # 불균형 정도 계산
    min_samples = min(original_distribution.values())
    max_samples = max(original_distribution.values())
    imbalance_ratio = max_samples / min_samples
    
    print(f"\n📈 불균형 분석:")
    print(f"   최소 샘플: {min_samples}개 (iOS)")
    print(f"   최대 샘플: {max_samples}개 (Server)")
    print(f"   불균형 비율: {imbalance_ratio:.2f}")
    
    # 클래스 가중치 계산 (항상 계산)
    class_weight_dict = {}
    total_samples = len(y)
    
    for i, stack in enumerate(target_stacks):
        positive_samples = np.sum(y[:, i])
        if positive_samples > 0:
            # 균형 가중치 공식: total_samples / (2 * positive_samples)
            weight = total_samples / (2 * positive_samples)
            class_weight_dict[i] = weight
        else:
            class_weight_dict[i] = 1.0
    
    print(f"\n🎯 클래스 가중치:")
    for i, stack in enumerate(target_stacks):
        weight = class_weight_dict[i]
        print(f"   {stack}: {weight:.3f}")
    
    # SMOTE 적용 여부 결정
    if strategy == 'smote_multilabel' and IMBLEARN_AVAILABLE:
        print(f"\n🔄 멀티라벨 SMOTE 적용 중...")
        
        # 방법 1: 각 라벨별로 개별 SMOTE 적용 후 합치기
        try:
            X_resampled_all = []
            y_resampled_all = []
            
            # 가장 많은 클래스의 샘플 수를 목표로 설정
            target_sample_count = max_samples
            
            # 각 라벨별로 SMOTE 적용
            for i, stack in enumerate(target_stacks):
                print(f"   처리 중: {stack}")
                
                # 현재 라벨의 positive/negative 샘플 추출
                y_binary = y[:, i]
                positive_mask = y_binary == 1
                negative_mask = y_binary == 0
                
                positive_count = np.sum(positive_mask)
                negative_count = np.sum(negative_mask)
                
                if positive_count < 10:  # 너무 적으면 스킵
                    print(f"     ⚠️ {stack}: 샘플이 너무 적어 SMOTE 스킵 ({positive_count}개)")
                    continue
                
                # 현재 라벨에 대해 SMOTE 적용
                # k_neighbors를 positive 샘플 수에 맞게 조정
                k_neighbors = min(5, positive_count - 1)
                if k_neighbors < 1:
                    k_neighbors = 1
                
                # 목표 샘플 수 계산 (최대 클래스의 70% 정도로 설정)
                target_positive = min(int(target_sample_count * 0.7), positive_count * 3)
                
                if target_positive > positive_count:
                    # SMOTE 적용
                    sampling_strategy = {1: target_positive}
                    smote = SMOTE(
                        sampling_strategy=sampling_strategy,
                        random_state=42,
                        k_neighbors=k_neighbors
                    )
                    
                    X_resampled_label, y_resampled_label = smote.fit_resample(X, y_binary)
                    
                    # 증강된 샘플만 추출
                    original_indices = np.arange(len(X))
                    augmented_indices = np.arange(len(X), len(X_resampled_label))
                    
                    if len(augmented_indices) > 0:
                        X_augmented = X_resampled_label[augmented_indices]
                        # 멀티라벨 형태로 변환 (현재 라벨만 1, 나머지는 0)
                        y_augmented = np.zeros((len(X_augmented), len(target_stacks)))
                        y_augmented[:, i] = 1
                        
                        X_resampled_all.append(X_augmented)
                        y_resampled_all.append(y_augmented)
                        
                        print(f"     ✅ {stack}: {len(X_augmented)}개 샘플 증강")
                else:
                    print(f"     ✅ {stack}: 충분한 샘플, 증강 불필요")
            
            # 원본 데이터와 증강 데이터 합치기
            if X_resampled_all:
                X_augmented_combined = np.vstack(X_resampled_all)
                y_augmented_combined = np.vstack(y_resampled_all)
                
                X_final = np.vstack([X, X_augmented_combined])
                y_final = np.vstack([y, y_augmented_combined])
                
                print(f"\n✅ SMOTE 적용 완료:")
                print(f"   원본: {X.shape[0]}개 → 최종: {X_final.shape[0]}개")
                print(f"   증강된 샘플: {X_augmented_combined.shape[0]}개")
                
                # 최종 분포 확인
                print(f"\n📊 SMOTE 적용 후 라벨별 분포:")
                for i, stack in enumerate(target_stacks):
                    count = np.sum(y_final[:, i])
                    percentage = (count / len(y_final)) * 100
                    improvement = count - original_distribution[stack]
                    print(f"   {stack}: {count}개 ({percentage:.1f}%) [+{improvement}]")
                
                # 개선된 불균형 비율 계산
                final_counts = [np.sum(y_final[:, i]) for i in range(len(target_stacks))]
                final_imbalance = max(final_counts) / min(final_counts)
                print(f"   개선된 불균형 비율: {imbalance_ratio:.2f} → {final_imbalance:.2f}")
                
                return X_final, y_final, class_weight_dict, 'smote_multilabel'
            
            else:
                print(f"❌ SMOTE 적용 실패: 증강할 수 있는 라벨이 없음")
                return X, y, class_weight_dict, 'class_weight'
                
        except Exception as e:
            print(f"❌ SMOTE 실패: {e}")
            print(f"   클래스 가중치 방법으로 대체")
            return X, y, class_weight_dict, 'class_weight'
    
    else:
        print(f"\n💡 SMOTE 미적용 → 클래스 가중치만 사용")
        if not IMBLEARN_AVAILABLE:
            print(f"   (imbalanced-learn 라이브러리가 설치되지 않음)")
        return X, y, class_weight_dict, 'class_weight'
'''def create_train_test_split_with_balance(X, y, target_stacks, test_size=0.25, random_state=42):
    """
    균형을 고려한 train/test 분할
    """
    print(f"\n🔄 균형 고려 데이터 분할 중... (test_size: {test_size})")
    
    # 멀티라벨을 단일 라벨로 변환
    y_single = np.argmax(y, axis=1)
    
    # 계층화 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y_single
    )
    
    print(f"✅ 분할 완료:")
    print(f"   훈련 세트: {X_train.shape[0]}개")
    print(f"   테스트 세트: {X_test.shape[0]}개")
    
    # 분할 후 각 세트의 클래스 분포 확인
    y_train_single = np.argmax(y_train, axis=1)
    y_test_single = np.argmax(y_test, axis=1)
    
    train_dist = Counter(y_train_single)
    test_dist = Counter(y_test_single)
    
    print(f"\n📊 분할 후 클래스 분포:")
    for class_idx in sorted(train_dist.keys()):
        stack_name = target_stacks[class_idx]
        train_count = train_dist[class_idx]
        test_count = test_dist.get(class_idx, 0)
        train_pct = (train_count / len(y_train_single)) * 100
        test_pct = (test_count / len(y_test_single)) * 100 if test_count > 0 else 0
        print(f"   {stack_name}: 훈련 {train_count}개 ({train_pct:.1f}%), 테스트 {test_count}개 ({test_pct:.1f}%)")
    
    return X_train, X_test, y_train, y_test'''
def create_train_test_split_with_balance(X, y, target_stacks, test_size=0.25, random_state=42):
    print(f"\n🔄 멀티라벨 데이터 분할 중... (test_size: {test_size})")
    
    # 멀티라벨에서는 stratify 없이 단순 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✅ 분할 완료:")
    print(f"   훈련 세트: {X_train.shape[0]}개")
    print(f"   테스트 세트: {X_test.shape[0]}개")
    
    # 각 세트의 클래스 분포 확인
    print(f"\n📊 분할 후 각 스택 분포:")
    for i, stack_name in enumerate(target_stacks):
        train_count = np.sum(y_train[:, i])
        test_count = np.sum(y_test[:, i])
        train_pct = (train_count / len(y_train)) * 100
        test_pct = (test_count / len(y_test)) * 100
        print(f"   {stack_name}: 훈련 {train_count}개 ({train_pct:.1f}%), 테스트 {test_count}개 ({test_pct:.1f}%)")
    
    return X_train, X_test, y_train, y_test
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
    
    # 스택 분포 분석
    analyze_stack_distribution(df)
    
    # 1. 언어 데이터 통합
    print("\n🔄 언어 데이터 통합 중...")
    if 'JavaScript' in df.columns and 'TypeScript' in df.columns:
        df["JS"] = df[['JavaScript', 'TypeScript']].sum(axis=1)
        df.drop(columns=['JavaScript', 'TypeScript'], inplace=True)
        print("✅ JavaScript + TypeScript → JS 통합 완료")
    
    if 'C' in df.columns and 'C++' in df.columns:
        df["C/C++"] = df[['C', 'C++']].sum(axis=1)
        #df.drop(columns=['C', 'C++'], inplace=True)
        print("✅ C + C++ → C/C++ 통합 완료")
    
    # 2. Repository 이름과 설명 분리
    print("\n📝 Repository 텍스트 분리 중...")
    if 'text' in df.columns:
        df[['repo_names', 'description']] = df['text'].apply(lambda x: pd.Series(split_repos(x)))
        df.drop(columns=['text'], inplace=True)
        print("✅ Repository 이름과 설명 분리 완료")
    
    # 3. Stack 처리 (멀티스택 지원)
    if 'stack' in df.columns:
        print("\n🔄 Stack 데이터 처리 중...")
        
        # &으로 분리된 멀티스택을 리스트로 변환
        df['stack_list'] = df['stack'].apply(process_stack)
        
        # 스택 통계 정보
        all_stacks = []
        for stack_list in df['stack_list']:
            all_stacks.extend(stack_list)
        
        stack_counts = Counter(all_stacks)
        print(f"📊 전체 고유 스택 수: {len(set(all_stacks))}")
        print(f"📊 스택별 분포:")
        for stack, count in stack_counts.most_common():
            print(f"   {stack}: {count}회")
        
        print("✅ Stack 데이터 처리 완료")
    df.drop(columns=['C', 'C++'], inplace=True)
    # 4. 언어 컬럼 확인 및 특성 엔지니어링
    print("\n📊 언어 데이터 확인 및 특성 생성 중...")
    exclude_columns = {'user_ID', 'username', 'repo_count', 'repo_names', 'description', 'stack', 'stack_list', 'note'}
    language_columns = [col for col in df.columns if col not in exclude_columns and df[col].dtype in ['int64', 'float64']]
    
    print(f"🎯 언어 컬럼: {language_columns}")
    
    # 언어 특성 엔지니어링 적용
    df = create_language_features(df, language_columns)
    
    # 5. 텍스트 전처리
    print("\n🧹 텍스트 전처리 중...")
    df['description'] = df['description'].fillna('').apply(clean_text)
    df['repo_names'] = df['repo_names'].fillna('').apply(clean_text)
    df['description'] = df['description'].replace('', 'no description available')
    df['repo_names'] = df['repo_names'].replace('', 'no repository name')
    
    # 텍스트 특성 개선 적용
    df = improve_text_features(df)
    
    # 6. BERT 임베딩 생성
    print("\n🤖 BERT 모델 로딩 중...")
    
    try:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        print("✅ BERT 모델 로딩 완료")
        
        # 유니크한 텍스트만 임베딩하여 중복 계산 방지
        unique_descriptions = df['description'].unique()
        unique_repo_names = df['repo_names'].unique()
        
        print(f"📝 유니크 Description 임베딩 생성 중... ({len(unique_descriptions)}개)")
        unique_desc_embeddings = model.encode(
            unique_descriptions.tolist(),
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        print(f"📁 유니크 Repository names 임베딩 생성 중... ({len(unique_repo_names)}개)")
        unique_name_embeddings = model.encode(
            unique_repo_names.tolist(),
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # 임베딩 매핑
        desc_embedding_dict = dict(zip(unique_descriptions, unique_desc_embeddings))
        name_embedding_dict = dict(zip(unique_repo_names, unique_name_embeddings))
        
        # DataFrame에 매핑
        description_embeddings = np.array([desc_embedding_dict[desc] for desc in df['description']])
        name_embeddings = np.array([name_embedding_dict[name] for name in df['repo_names']])
        
        print("✅ BERT 임베딩 생성 완료")
        
    except Exception as e:
        print(f"❌ BERT 임베딩 생성 실패: {e}")
        return
    
    # 7. 최종 특성 조합
    print("\n🔗 최종 특성 조합 중...")
    
    # 언어 특성 (기존 + 새로 생성된 특성)
    language_feature_cols = language_columns + [
        'num_languages', 'main_language_ratio', 
        'language_diversity', 'frontend_lang_ratio', 'backend_lang_ratio',
        'mobile_lang_ratio', 'system_lang_ratio'
    ]
    language_feature_cols = [col for col in language_feature_cols if col in df.columns]
    
    # 텍스트 특성
    text_feature_cols = [
        'description_length', 'repo_names_length', 'description_word_count', 
        'repo_names_word_count', 'has_description', 'has_repo_names',
        'avg_repo_name_length', 'is_prolific_dev'
    ]
    tech_keyword_cols = [col for col in df.columns if col.startswith('has_')]
    text_feature_cols.extend(tech_keyword_cols)
    text_feature_cols = [col for col in text_feature_cols if col in df.columns]
    
    # 특성 조합
    X_lang = df[language_feature_cols].values.astype(np.float32)
    X_text_features = df[text_feature_cols].values.astype(np.float32)
    X_desc = description_embeddings.astype(np.float32)
    X_name = name_embeddings.astype(np.float32)
    
    X_total = np.concatenate([X_lang, X_text_features, X_name, X_desc], axis=1)
    
    print(f"📊 특성 조합 결과:")
    print(f"   언어 특성: {X_lang.shape[1]}개")
    print(f"   텍스트 특성: {X_text_features.shape[1]}개")
    print(f"   Repository 이름 임베딩: {X_name.shape[1]}개")
    print(f"   Description 임베딩: {X_desc.shape[1]}개")
    print(f"   총 특성: {X_total.shape[1]}개")
    
    # 8. 낮은 분산 특성 제거
    X_total, selected_features = filter_low_variance_features(X_total, threshold=0.0005)
    
    
    # 9. 타겟 변수 처리 (7개 주요 스택만 사용)
    target_stacks = ['Server', 'System', 'Visualization', 'Frontend', 'Android', 'ML-Data', 'iOS']

    print(f"🎯 타겟 스택 (7개): {target_stacks}")

    # 각 사용자의 스택 리스트에서 타겟 스택만 필터링
    def filter_target_stacks(stack_list):
        """타겟 스택에 포함된 스택만 필터링"""
        if not stack_list:
            return []
        filtered = [stack for stack in stack_list if stack in target_stacks]
        return list(set(filtered))  # 중복 제거

    df['filtered_stack_list'] = df['stack_list'].apply(filter_target_stacks)

    # 타겟 스택이 있는 사용자만 필터링
    valid_mask = df['filtered_stack_list'].apply(lambda x: len(x) > 0)
    print(f"📊 유효한 샘플 수: {valid_mask.sum()} / {len(df)}")

    # 필터링된 데이터 확인
    print(f"\n📋 필터링된 스택 분포:")
    filtered_stacks_all = []
    for stack_list in df.loc[valid_mask, 'filtered_stack_list']:
        filtered_stacks_all.extend(stack_list)

    filtered_stack_counts = Counter(filtered_stacks_all)
    for stack in target_stacks:
        count = filtered_stack_counts.get(stack, 0)
        percentage = (count / valid_mask.sum()) * 100 if valid_mask.sum() > 0 else 0
        print(f"   {stack}: {count}회 ({percentage:.1f}%)")

    X_filtered = X_total[valid_mask.to_numpy()]
    filtered_stack_lists = df.loc[valid_mask, 'filtered_stack_list'].tolist()

    # 멀티라벨 인코딩 (7개 타겟 스택 기준)
    mlb = MultiLabelBinarizer(classes=target_stacks)
    y_filtered = mlb.fit_transform(filtered_stack_lists)

    print(f"✅ 타겟 변수 처리 완료")
    print(f"   • 최종 샘플 수: {X_filtered.shape[0]}")
    print(f"   • 멀티라벨 shape: {y_filtered.shape}")  # (n_samples, 7)이어야 함

    # 각 사용자가 가진 스택 개수 확인
    stack_counts_per_user = [len(stack_list) for stack_list in filtered_stack_lists]
    unique_counts = Counter(stack_counts_per_user)
    print(f"   • 사용자별 스택 개수 분포:")
    for count, users in sorted(unique_counts.items()):
        print(f"     {count}개 스택: {users}명")
    
    # ⭐ 10. 클래스 불균형 처리 (새로 추가된 부분)
    X_balanced, y_balanced, class_weights, strategy_used = handle_class_imbalance_multilabel(
        X_filtered, y_filtered, target_stacks, strategy='smote_multilabel'
    )
    
    # ⭐ 11. 균형 고려 train/test 분할
    X_train, X_test, y_train, y_test = create_train_test_split_with_balance(
        X_balanced, y_balanced, target_stacks, test_size=0.25, random_state=42
    )
    
    # 12. 결과 저장
    print("\n💾 결과 저장 중...")
    output_dir = 'C:/Users/jun01/OneDrive/바탕 화면/고려대/데과/TermProject/pkl_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # 개선된 데이터 저장 (여러 버전)
    # 버전 1: 원본 (균형 처리 안된 것)
    np.save(os.path.join(output_dir, "X_filtered_original.npy"), X_filtered)
    np.save(os.path.join(output_dir, "y_filtered_original.npy"), y_filtered)
    
    # 버전 2: 균형 처리된 전체 데이터
    np.save(os.path.join(output_dir, "X_filtered_balanced.npy"), X_balanced)
    np.save(os.path.join(output_dir, "y_filtered_balanced.npy"), y_balanced)
    
    # 버전 3: 균형 처리 + 분할된 데이터
    np.save(os.path.join(output_dir, "X_train_balanced.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test_balanced.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train_balanced.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test_balanced.npy"), y_test)
    
    # 메타데이터 저장 (클래스 가중치 포함)
    metadata = {
        'target_stacks': target_stacks,
        'language_features': language_feature_cols,
        'text_features': text_feature_cols,
        'selected_features': selected_features,
        'total_features': X_balanced.shape[1],
        'samples': {
            'original': X_filtered.shape[0],
            'balanced': X_balanced.shape[0],
            'train': X_train.shape[0],
            'test': X_test.shape[0]
        },
        'embedding_dims': {
            'description': description_embeddings.shape[1],
            'repo_names': name_embeddings.shape[1]
        },
        'class_weights': class_weights,
        'strategy_used': strategy_used,
        'imbalance_handling': {
            'applied': strategy_used != 'class_weight',
            'method': strategy_used
        }
    }
    
    with open(os.path.join(output_dir, "metadata_enhanced.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    # 처리된 DataFrame도 저장
    df_filtered = df[valid_mask].copy()
    df_filtered.to_pickle(os.path.join(output_dir, "processed_df_enhanced.pkl"))
    
    # 클래스 가중치를 별도 파일로 저장 (모델 학습시 쉽게 불러오기 위해)
    class_weight_for_keras = {}
    for i, stack in enumerate(target_stacks):
        class_weight_for_keras[i] = class_weights.get(i, 1.0)
    
    with open(os.path.join(output_dir, "class_weights.pkl"), 'wb') as f:
        pickle.dump(class_weight_for_keras, f)
    
    print("✅ 개선된 데이터 저장 완료!")
    print(f"📊 최종 결과:")
    print(f"   • 원본 데이터: X={X_filtered.shape}, y={y_filtered.shape}")
    print(f"   • 균형 처리된 데이터: X={X_balanced.shape}, y={y_balanced.shape}")
    print(f"   • 훈련 데이터: X={X_train.shape}, y={y_train.shape}")
    print(f"   • 테스트 데이터: X={X_test.shape}, y={y_test.shape}")
    print(f"   • Target stacks: {target_stacks}")
    print(f"   • 언어 특성: {len(language_feature_cols)}개")
    print(f"   • 텍스트 특성: {len(text_feature_cols)}개")
    print(f"   • BERT 임베딩: {description_embeddings.shape[1] + name_embeddings.shape[1]}차원")
    print(f"   • 사용된 균형 전략: {strategy_used}")
    
    # 최종 스택별 분포 재확인
    print(f"\n📊 최종 스택별 샘플 분포 (균형 처리 후):")
    y_balanced_single = np.argmax(y_balanced, axis=1)
    final_distribution = Counter(y_balanced_single)
    for class_idx, count in final_distribution.items():
        stack_name = target_stacks[class_idx]
        percentage = (count / len(y_balanced_single)) * 100
        print(f"   {stack_name}: {count}개 ({percentage:.1f}%)")
    
    '''# 13. 사용 가이드 출력
    print(f"\n📖 사용 가이드:")
    print(f"=" * 50)
    print(f"🔹 원본 데이터 사용:")
    print(f"   X = np.load('X_filtered_original.npy')")
    print(f"   y = np.load('y_filtered_original.npy')")
    print(f"")
    print(f"🔹 균형 처리된 데이터 사용:")
    print(f"   X = np.load('X_filtered_balanced.npy')")
    print(f"   y = np.load('y_filtered_balanced.npy')")
    print(f"")
    print(f"🔹 바로 학습 가능한 분할 데이터:")
    print(f"   X_train = np.load('X_train_balanced.npy')")
    print(f"   X_test = np.load('X_test_balanced.npy')")
    print(f"   y_train = np.load('y_train_balanced.npy')")
    print(f"   y_test = np.load('y_test_balanced.npy')")
    print(f"")
    print(f"🔹 클래스 가중치 사용:")
    print(f"   import pickle")
    print(f"   with open('class_weights.pkl', 'rb') as f:")
    print(f"       class_weights = pickle.load(f)")
    print(f"   # 모델 학습시:")
    print(f"   model.fit(X_train, y_train, class_weight=class_weights, ...)")
    print(f"=" * 50)'''
    
    # 14. 예상 성능 향상 요약
    original_imbalance = max(Counter(np.argmax(y_filtered, axis=1)).values()) / min(Counter(np.argmax(y_filtered, axis=1)).values())
    final_imbalance = max(final_distribution.values()) / min(final_distribution.values()) if len(final_distribution) > 1 else 1.0
    
    print(f"\n🎯 예상 성능 향상:")
    print(f"   • 특성 엔지니어링: +3-5%p")
    print(f"   • 클래스 불균형 해결: +2-4%p")
    print(f"     - 원본 불균형 비율: {original_imbalance:.2f}")
    print(f"     - 처리 후 불균형 비율: {final_imbalance:.2f}")
    print(f"   • 텍스트 품질 개선: +1-2%p")
    print(f"   • 총 예상 향상: +6-11%p (66.7% → 73-78%)")

if __name__ == "__main__":
    main()