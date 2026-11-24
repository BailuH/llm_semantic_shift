import argparse
import csv
import json
import pickle
from pathlib import Path

import jieba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_DIR = Path("data/processed")
STOPWORDS_DIR = Path("data/stopwords")
TABLES_DIR = Path("outputs/tables")
PERIOD = "2026"  # 相似检索在 T3 语料上演示

DOMAIN_WORDS = [
    "算法", "大模型", "内卷", "躺平", "消费降级", "灵活就业",
    "性别角色", "工作稳定", "社会信任", "技术焦虑", "人工智能",
    "短视频", "直播带货", "远程办公", "自然语言处理", "大语言模型",
    "语言模型", "机器学习", "深度学习", "生成式",
]


def _space_tokenizer(s):
    """与 03_tokenize 一致；TF-IDF 向量化器在 03 以 __main__ 运行时存了该函数引用，
    此处同名定义以保证 pickle 反序列化时能在 __main__ 找到它。"""
    return s.split(" ")

QUERIES = [
    "大语言模型是深度学习背景下发展起来的一类神经网络语言模型，能够完成文本生成、翻译、问答等任务。",
    "推荐算法通过用户行为数据预测偏好，被广泛用于短视频、电商与内容分发平台。",
    "灵活就业与远程办公改变了传统的工作模式，劳动者面临收入不稳定与社会保障缺失的压力。",
]


def load_stopwords():
    sw = set()
    merged = STOPWORDS_DIR / "stopwords_merged.txt"
    if merged.exists():
        sw = set(merged.read_text(encoding="utf-8").splitlines())
    return sw


def tokenize_query(text, stopwords):
    for w in DOMAIN_WORDS:
        jieba.add_word(w)
    toks = [w for w in jieba.lcut(text, cut_all=False)
            if len(w) > 1 and w not in stopwords]
    return " ".join(toks)


def load_t3_corpus():
    pkl = PROCESSED_DIR / f"tfidf_{PERIOD}.pkl"
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    vec = data["vectorizer"]
    matrix = data["matrix"]

    jsonl = PROCESSED_DIR / f"zhwiki_{PERIOD}.jsonl"
    titles, snippets = [], []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            titles.append(doc.get("title", ""))
            snippets.append(doc.get("text", "")[:160])
    return vec, matrix, titles, snippets


def retrieve(query, vec, matrix, topk=5):
    q_vec = vec.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    idx = sims.argsort()[::-1][:topk]
    return idx, sims


def main():
    print("=" * 60)
    print("第 6 步：TF-IDF 余弦相似文本检索（T3 语料）")
    print("=" * 60)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n加载 T3({PERIOD}) TF-IDF 与语料 ...")
    vec, matrix, titles, snippets = load_t3_corpus()
    print(f"  矩阵: {matrix.shape}, 文档数: {len(titles)}")

    stopwords = load_stopwords()

    all_rows = []
    for qi, q in enumerate(QUERIES, 1):
        q_tok = tokenize_query(q, stopwords)
        print(f"\n查询 {qi}: {q[:50]}...")
        print(f"  分词后: {q_tok}")
        idx, sims = retrieve(q_tok, vec, matrix)
        for rank, i in enumerate(idx, 1):
            row = {
                "query_id": qi, "query": q, "rank": rank,
                "doc_idx": int(i), "title": titles[i],
                "cosine": round(float(sims[i]), 4),
                "snippet": snippets[i],
            }
            all_rows.append(row)
            print(f"  Top{rank} [{row['cosine']:.3f}] {row['title']} | {row['snippet'][:60]}")

    out = TABLES_DIR / "similar_t3.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "query", "rank", "doc_idx",
                                          "title", "cosine", "snippet"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n保存: {out} ({len(all_rows)} 行)")
    print("\n完成。")


if __name__ == "__main__":
    main()
