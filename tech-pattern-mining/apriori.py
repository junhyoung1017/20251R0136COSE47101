import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 데이터 불러오기
file_path = 'C:\\Users\\gse07\\Downloads\\github_profiles_total_v5.csv'
df = pd.read_csv(file_path, encoding='utf-8')
df.columns = df.columns.str.strip()

# 사용할 컬럼
language_columns = ['Assembly', 'C', 'C++', 'C#', 'Dart', 'Go', 'Java', 'JavaScript',
                    'Kotlin', 'MATLAB', 'PHP', 'Python', 'Ruby', 'Rust', 'Scala', 'Swift', 'TypeScript']

# threshold 기반 이진화
threshold = 0.2
df_bin = df[language_columns].applymap(lambda x: 1 if x >= threshold else 0)

# Apriori
frequent_itemsets = apriori(df_bin, min_support=0.005, use_colnames=True)
frequent_itemsets_multi = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) >= 2)]

# Association Rule
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage']]
rules = rules.sort_values(by='lift', ascending=False)

# 결과 출력
print(frequent_itemsets_multi)
print("\nTop Association Rules:")
print(rules.head(20))
