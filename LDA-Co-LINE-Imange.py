import pandas as pd
import re
import itertools
import networkx as nx
from collections import Counter
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nodevectors import Node2Vec
import matplotlib.pyplot as plt
from matplotlib import cm  # 互換性のあるカラーマップ取得方法

nltk.download("punkt")
nltk.download("stopwords")

# ---------- STEP 1: データの読み込みと前処理 ----------
df = pd.read_csv("test3.csv", encoding="utf-8")  # "abstract"列が必要
docs = df["abstract"].dropna().astype(str).tolist()

stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words and len(w) > 2]
    return tokens

tokenized_docs = [preprocess(doc) for doc in docs]

# ---------- STEP 2: LDAによるトピック抽出 ----------
dictionary = corpora.Dictionary(tokenized_docs)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=8, passes=10, random_state=42)

# 各トピックの上位500語を取得
top_terms_by_topic = {}
top_terms_set = set()

for topic_id in range(8):
    top_terms = lda_model.show_topic(topic_id, topn=500)
    top_terms_by_topic[topic_id] = [word for word, _ in top_terms]
    top_terms_set.update(top_terms_by_topic[topic_id])

# ---------- STEP 3: 共起ネットワークの構築 ----------
pair_counter = Counter()

for tokens in tokenized_docs:
    topic_terms_in_doc = [t for t in tokens if t in top_terms_set]
    for a, b in itertools.combinations(sorted(set(topic_terms_in_doc)), 2):
        pair_counter[(a, b)] += 1

# networkxグラフ作成
G = nx.Graph()
for (a, b), weight in pair_counter.items():
    G.add_edge(a, b, weight=weight)

print(f"🧩 グラフのノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}")

# ---------- STEP 4: Node2Vecによるノード埋め込み ----------
node2vec_model = Node2Vec(
    n_components=64,
    walklen=30,
    epochs=20,
    return_weight=1.0,
    neighbor_weight=1.0,
    threads=4,
    w2vparams={'window': 5, 'min_count': 1}
)

node2vec_model.fit(G)

# 学習されたノードのみ埋め込み取得
valid_nodes = [node for node in G.nodes() if node in node2vec_model.model.wv.key_to_index]
embeddings = {node: node2vec_model.predict(node) for node in valid_nodes}

# ---------- STEP 5: 可視化 ----------
term_to_topic = {}
for topic_id, terms in top_terms_by_topic.items():
    for term in terms:
        if term not in term_to_topic:
            term_to_topic[term] = topic_id

x_coords, y_coords, colors, labels = [], [], [], []

# ✅ カラーマップ（互換性あり・最大10色）
color_map = cm.get_cmap('tab10')

for node in valid_nodes:
    vec = embeddings[node]
    x_coords.append(vec[0])
    y_coords.append(vec[1])
    topic_id = term_to_topic.get(node, -1)
    colors.append(color_map((topic_id if topic_id >= 0 else 0) % 10))  # tab10の10色ループ
    labels.append(f"{node} (T{topic_id})" if topic_id >= 0 else node)

# プロット
plt.figure(figsize=(12, 8))
scatter = plt.scatter(x_coords, y_coords, c=colors, alpha=0.7)
plt.title("LDA + Co-occurrence + Node2Vec Embedding")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")


# ラベル表示（任意でコメント解除）
for i, label in enumerate(labels):
    if i % 10 == 0:  # 10個につき1つだけ表示
        plt.text(x_coords[i], y_coords[i], label, fontsize=8, alpha=0.6)

plt.tight_layout()
plt.show()

with open("vosviewer_network.txt", "w", encoding="utf-8") as f:
    f.write("Source\tTarget\tWeight\n")
    for (a, b), weight in pair_counter.items():
        if a in valid_nodes and b in valid_nodes and weight > 0:
            f.write(f"{a}\t{b}\t{weight}\n")
