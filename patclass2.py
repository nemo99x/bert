import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# 1.csv 読み込み（1行目がヘッダーとして存在）
df = pd.read_csv('1.csv')

# 欠損値を空文字列で埋める
df['Abstract'] = df['Abstract'].fillna("").astype(str)
df['Inventors'] = df['Inventors'].fillna("").astype(str)

# AbstractとInventorsを連結したテキストを生成
df['combo_text'] = df['Abstract'] + ' Inventor: ' + df['Inventors']

# BERTベースで文をベクトル化
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
combo_embeddings = model.encode(df['combo_text'].tolist())

# クラスタ数指定
n_clusters = 10  # 適宜調整
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(combo_embeddings)
df['group'] = clusters

# クラスタごとのcombo_textを結合
cluster_texts = []
for i in range(n_clusters):
    texts = df[df['group'] == i]['combo_text'].tolist()
    cluster_texts.append(' '.join(texts))

# TF-IDFで代表技術用語などの特徴単語抽出
vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
tfidf_matrix = vectorizer.fit_transform(cluster_texts)
feature_names = vectorizer.get_feature_names_out()

cluster_keywords = []
for row in tfidf_matrix.toarray():
    top_idx = row.argmax()
    cluster_keywords.append(feature_names[top_idx])

df['group_name'] = df['group'].map({i: kw for i, kw in enumerate(cluster_keywords)})

# 6列目（新しい分類キー）に「代表技術名＋代表発明者」などを記入する例
def make_group_id(i):
    # そのグループの代表的発明者名（最多登場）
    inventor_list = df[df['group'] == i]['Inventors']
    rep_inventor = inventor_list.value_counts().idxmax()
    return f"{cluster_keywords[i]} / {rep_inventor}"

df['final_group'] = df['group'].map(make_group_id)

# 2.csv用出力
output_columns = ['Display Key', 'Application Date', 'Inventors', 'Abstract', 'group_name', 'final_group']
output_columns = [col for col in output_columns if col in df.columns]
df_out = df[output_columns]
df_out.to_csv('2.csv', index=False, header=False)

# 追加インポート（不要ならコメントアウトしてOK）
import re

# 技術カテゴリと代表的なキーワードの辞書（小文字化推奨）
category_keywords = {
    'Material': ['material', 'composition', 'alloy', 'substrate', 'film', 'coating', 'polymer', 'metal', 'ceramic'],
    'Circuit': ['circuit', 'transistor', 'amplifier', 'semiconductor', 'chip', 'ic', 'signal', 'voltage', 'current'],
    'Structure': ['structure', 'framework', 'layer', 'design', 'architecture', 'assembly', 'configuration', 'mechanical'],
    'Manufacture': ['manufacture', 'fabrication', 'process', 'method', 'forming', 'production', 'machining', 'deposition'],
    'Protocol': ['protocol', 'algorithm', 'encryption', 'communication', 'processing', 'data', 'network', 'software', 'signal processing'],
    'Idea': ['idea', 'concept', 'approach', 'methodology', 'theory', 'principle']
}

# Abstractからカテゴリ判定関数
def classify_category(text):
    text_lower = text.lower()  # 小文字化
    matched_categories = []
    for category, keywords in category_keywords.items():
        for kw in keywords:
            # 単語単位マッチが良いので簡易には正規表現で単語境界を使う
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                matched_categories.append(category)
                break  # 1カテゴリで1単語見つかれば十分次へ
    if not matched_categories:
        return 'Unknown'
    # 複数該当した場合は優先度順にソートし1つだけ返す（必要に応じ複数返してもOK）
    priority_order = ['Material', 'Circuit', 'Structure', 'Manufacture', 'Protocol', 'Idea']
    for p in priority_order:
        if p in matched_categories:
            return p
    return matched_categories[0]

# 7列目に新規追加
df['tech_category'] = df['Abstract'].fillna("").astype(str).apply(classify_category)

# 出力カラムにtech_category追加（7列目）
output_columns = ['Display Key', 'Application Date', 'Inventors', 'Abstract', 'group_name', 'final_group', 'tech_category']
output_columns = [col for col in output_columns if col in df.columns]

df_out = df[output_columns]

# CSV出力（ヘッダーなし、インデックスなし）
df_out.to_csv('2.csv', index=False, header=False)
