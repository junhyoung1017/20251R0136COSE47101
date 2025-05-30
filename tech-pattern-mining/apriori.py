import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# ① csv 불러오기
file_path = 'C:\\Users\\gse07\\Downloads\\github_profiles_total_v5.csv'
df = pd.read_csv(file_path, encoding='utf-8')

df.columns = df.columns.str.strip()

# ② 이진화
language_columns = ['Assembly', 'C', 'C++', 'C#', 'Dart', 'Go', 'Java', 'JavaScript',
                    'Kotlin', 'MATLAB', 'PHP', 'Python', 'Ruby', 'Rust', 'Scala', 'Swift', 'TypeScript']

df_bin = df[language_columns].applymap(lambda x: 1 if x > 0 else 0)

# ③ Apriori 적용
frequent_itemsets = apriori(df_bin, min_support=0.008, use_colnames=True)

# ④ 2개 이상 itemset만 필터링
frequent_itemsets_multi = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) >= 2)]
print(frequent_itemsets_multi)

# ⑤ Association Rule
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print("Association Rules:")
print(rules[['antecedents', 'consequents', 'confidence']])
