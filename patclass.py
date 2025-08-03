import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# 1.csv 読み込み（1行目がヘッダーとして存在する想定）
df = pd.read_csv('1.csv')

# inventor列がなければ存在確認後除外してください
print("Columns in CSV:", df.columns.tolist())
# 例: ['Display Key', 'Application Date', 'Abstract']

# 欠損値を空文字列に変換し、strに変換しておく（BERT入力用）
df['Abstract'] = df['Abstract'].fillna("").astype(str)

# 請求項テキストをBERTでベクトル化
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
claim_embeddings = model.encode(df['Abstract'].tolist())

# クラスタ数指定
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(claim_embeddings)
df['group'] = clusters

# 各クラスタの請求項テキストを結合
cluster_texts = []
for i in range(n_clusters):
    texts = df[df['group'] == i]['Abstract'].tolist()
    cluster_texts.append(' '.join(texts))

# TF-IDFで特徴単語抽出（英語ストップワード除去）
vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
tfidf_matrix = vectorizer.fit_transform(cluster_texts)
feature_names = vectorizer.get_feature_names_out()

cluster_keywords = []
for row in tfidf_matrix.toarray():
    top_idx = row.argmax()
    cluster_keywords.append(feature_names[top_idx])

# group_name列を作成
df['group_name'] = df['group'].map({i: kw for i, kw in enumerate(cluster_keywords)})

# 出力用データフレーム
# inventor列が無ければ外す
output_columns = ['Display Key', 'Abstract', 'Application Date', 'group_name']

# inventors列があれば次のように追加
# output_columns = ['Display Key', 'Abstract', 'Application Date', 'inventor', 'group_name']

# 存在しない列は除外しておく
output_columns = [col for col in output_columns if col in df.columns]

df_out = df[output_columns]

# 2.csvへ書き込み（ヘッダーなし、インデックスなし）
df_out.to_csv('2.csv', index=False, header=False)
