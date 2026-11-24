import argparse
import csv
import json
import os
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import SEED

PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("outputs/tables")
PERIODS = ["2021", "2022", "2026"]
STOPWORDS_PATH = Path("data/stopwords/stopwords_merged.txt")

DOMAIN_WORDS = [
    "算法", "大模型", "内卷", "躺平", "消费降级", "灵活就业",
    "性别角色", "工作稳定", "社会信任", "技术焦虑", "人工智能",
    "短视频", "直播带货", "远程办公", "自然语言处理", "大语言模型",
    "语言模型", "机器学习", "深度学习", "生成式",
]

_STOPWORDS = None


def _space_tokenizer(s):
    return s.split(" ")


def _worker_init(stopwords):
    global _STOPWORDS
    _STOPWORDS = set(stopwords)
    for w in DOMAIN_WORDS:
        jieba.add_word(w)


def _tokenize_batch(batch):
    sw = _STOPWORDS
    out = []
    for text in batch:
        toks = [w for w in jieba.lcut(text, cut_all=False)
                if len(w) > 1 and w not in sw]
        out.append(" ".join(toks))
    return out


def parallel_tokenize(texts, n_workers, batch_size=500):
    """多进程分词，返回空格分隔的词串列表"""
    chunks = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    result = []
    with Pool(n_workers, initializer=_worker_init,
              initargs=(_load_stopwords(),)) as pool:
        for batch_result in pool.imap(_tokenize_batch, chunks):
            result.extend(batch_result)
    return result


def _load_stopwords():
    if STOPWORDS_PATH.exists():
        return STOPWORDS_PATH.read_text(encoding="utf-8").splitlines()
    return []


def load_balanced(max_per_period):
    random.seed(SEED)
    texts, labels, titles = [], [], []
    for period in PERIODS:
        path = PROCESSED_DIR / f"zhwiki_{period}.jsonl"
        if not path.exists():
            print(f"  [警告] 缺少 {path}")
            continue
        docs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                t = doc.get("text", "")
                if t:
                    docs.append((doc.get("title", ""), t))
        if max_per_period and len(docs) > max_per_period:
            docs = random.sample(docs, max_per_period)
        for title, t in docs:
            texts.append(t)
            labels.append(period)
            titles.append(title)
        print(f"  {period}: 采样 {len(docs)} 篇")
    return texts, labels, titles


def main():
    parser = argparse.ArgumentParser(description="时段三分类（MultinomialNB / LinearSVC）")
    parser.add_argument("--max-per-period", type=int, default=80000,
                        help="每时段均衡采样的文档上限（0=全量）")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    print("=" * 60)
    print("第 7 步：时段分类建模（三分类）")
    print("=" * 60)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n加载并均衡采样语料 (max_per_period={args.max_per_period}) ...")
    texts, labels, titles = load_balanced(args.max_per_period)
    n = len(texts)
    print(f"  总计 {n} 篇")

    cnt = Counter(labels)
    baseline = max(cnt.values()) / n
    print(f"  类别分布: {dict(cnt)}")
    print(f"  多数类基线 accuracy = {baseline:.4f}")

    # 分层划分 8:2（在分词前划分，保证同一文档不跨集）
    X_train, X_test, y_train, y_test, titles_train, titles_test = train_test_split(
        texts, labels, titles, test_size=0.2, random_state=SEED, stratify=labels,
    )
    print(f"  训练集 {len(X_train)} / 测试集 {len(X_test)}")

    # 多进程分词（训练集 + 测试集分别分词，但训练集先 fit TF-IDF）
    print(f"\n多进程分词 ({args.workers} 进程) ...")
    print("  分词训练集 ...")
    X_train_tok = parallel_tokenize(X_train, args.workers)
    print("  分词测试集 ...")
    X_test_tok = parallel_tokenize(X_test, args.workers)

    # TF-IDF 仅在训练集上 fit（防泄露），用轻量 space tokenizer
    print("\n拟合 TF-IDF (仅训练集) ...")
    vec = TfidfVectorizer(
        tokenizer=_space_tokenizer, token_pattern=None,
        max_df=0.85, min_df=5, max_features=50000,
        ngram_range=(1, 1), lowercase=False,
    )
    X_train_vec = vec.fit_transform(X_train_tok)
    X_test_vec = vec.transform(X_test_tok)
    print(f"  词表大小: {len(vec.vocabulary_)}, 矩阵: {X_train_vec.shape}")

    results = {}
    for name, clf in [
        ("MultinomialNB", MultinomialNB()),
        ("LinearSVC", LinearSVC(random_state=SEED, max_iter=10000)),
    ]:
        print(f"\n=== {name} ===")
        clf.fit(X_train_vec, y_train)
        pred = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, pred)
        print(f"Accuracy: {acc:.4f}  (基线 {baseline:.4f}, 提升 {acc-baseline:+.4f})")
        rep = classification_report(y_test, pred, digits=4)
        print(rep)
        cm = confusion_matrix(y_test, pred, labels=PERIODS)
        print("混淆矩阵 (行=真实, 列=预测, 顺序 2021/2022/2026):")
        print(cm)
        results[name] = {"acc": acc, "pred": pred, "cm": cm, "report": rep}

    best = "LinearSVC" if "LinearSVC" in results else list(results)[0]
    pred = results[best]["pred"]
    cm = results[best]["cm"]

    cm_path = TABLES_DIR / "confusion_matrix.csv"
    with open(cm_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["actual"] + [f"pred_{p}" for p in PERIODS])
        for i, p in enumerate(PERIODS):
            w.writerow([p] + cm[i].tolist())
    print(f"\n混淆矩阵已保存: {cm_path}")

    rep_path = TABLES_DIR / "classification_summary.csv"
    with open(rep_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "baseline", "improvement"])
        for name, r in results.items():
            w.writerow([name, f"{r['acc']:.4f}", f"{baseline:.4f}",
                        f"{r['acc']-baseline:+.4f}"])
    print(f"分类汇总已保存: {rep_path}")

    # 错分分析
    print(f"\n=== 错分分析（基于 {best}）===")
    feature_names = vec.get_feature_names_out()
    mis_idx = [i for i, (yt, yp) in enumerate(zip(y_test, pred)) if yt != yp]
    print(f"  错分样本数: {len(mis_idx)}")
    shown = 0
    mis_rows = []
    for i in mis_idx:
        if shown >= 5:
            break
        yt, yp = y_test[i], pred[i]
        row = X_test_vec[i]
        if row.nnz == 0:
            continue
        cols = row.nonzero()[1]
        vals = row.data
        order = vals.argsort()[::-1][:8]
        top_feats = [(feature_names[cols[k]], round(float(vals[k]), 3))
                     for k in order]
        print(f"\n  [样本 {shown+1}] 真实={yt} 预测={yp} | 标题: {titles_test[i][:30]}")
        print(f"    Top TF-IDF 词: {top_feats}")
        mis_rows.append({
            "sample": shown + 1, "true": yt, "pred": yp,
            "title": titles_test[i][:40],
            "top_features": "; ".join(f"{w}({v})" for w, v in top_feats),
        })
        shown += 1

    mis_path = TABLES_DIR / "misclassification_analysis.csv"
    with open(mis_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample", "true", "pred",
                                          "title", "top_features"])
        w.writeheader()
        w.writerows(mis_rows[:2])
    print(f"\n错分样本已保存: {mis_path}")

    offdiag = {}
    for i, a in enumerate(PERIODS):
        for j, p in enumerate(PERIODS):
            if i != j and cm[i, j] > 0:
                offdiag[f"{a}->{p}"] = int(cm[i, j])
    if offdiag:
        worst = max(offdiag, key=offdiag.get)
        print(f"\n混淆最严重方向: {worst} ({offdiag[worst]} 例)")
    print("\n完成。")


if __name__ == "__main__":
    main()
