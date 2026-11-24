import argparse
import json
import pickle
import re
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from utils import SEED

PROCESSED_DIR = Path("data/processed")
STOPWORDS_DIR = Path("data/stopwords")
TFIDF_TOPK = 20
BATCH_SIZE = 2000

# 锚定词 / 领域词，提前加入 jieba 词典锁定为单一词元
DOMAIN_WORDS = [
    "算法", "大模型", "内卷", "躺平", "消费降级", "灵活就业",
    "性别角色", "工作稳定", "社会信任", "技术焦虑", "人工智能",
    "短视频", "直播带货", "远程办公", "自然语言处理", "大语言模型",
    "语言模型", "机器学习", "深度学习", "生成式",
]

# 句子切分（Word2Vec 按句训练）
SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")

# 维基百科特有的高频噪声词
WIKI_NOISE_WORDS = {
    "参见", "参考文献", "外部链接", "编辑", "分类", "目录", "注释",
    "出处", "来源", "拓展", "延伸", "阅读", "相关", "条目", "页面",
    "维基", "百科", "维基百科", "stub", "协调", "世界", "基金",
}

STOPWORD_URLS = {
    "hit_stopwords.txt": "https://raw.githubusercontent.com/goto456/stopwords/master/hit_stopwords.txt",
    "baidu_stopwords.txt": "https://raw.githubusercontent.com/goto456/stopwords/master/baidu_stopwords.txt",
}

# 子进程共享的停用词表，由 initializer 注入
_STOPWORDS = None


def _space_tokenizer(s):
    """TF-IDF 用的占位分词器（输入已是空格分隔的词串）"""
    return s.split(" ")


def _worker_init(stopwords):
    global _STOPWORDS
    _STOPWORDS = set(stopwords)
    for w in DOMAIN_WORDS:
        jieba.add_word(w)


def _process_batch(batch):
    """处理一批 JSONL 行，返回 (n_lines, 句子字符串列表, 文档token串列表, raw_counter, filtered_counter)"""
    sw = _STOPWORDS
    n_lines = len(batch)
    sent_lines = []
    doc_strs = []
    vocab_raw = Counter()
    vocab_filt = Counter()
    sp = SENT_SPLIT_RE.split
    jlcut = jieba.lcut
    for line in batch:
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = doc.get("text", "")
        if not text:
            doc_strs.append("")
            continue
        doc_toks = []
        for sent in sp(text):
            sent = sent.strip()
            if len(sent) < 2:
                continue
            raw = jlcut(sent, cut_all=False)
            toks = []
            for w in raw:
                vocab_raw[w] += 1
                if len(w) > 1 and w not in sw:
                    toks.append(w)
                    vocab_filt[w] += 1
            if toks:
                sent_lines.append(" ".join(toks))
                doc_toks.extend(toks)
        doc_strs.append(" ".join(doc_toks))
    return n_lines, sent_lines, doc_strs, vocab_raw, vocab_filt


def download_stopwords():
    """下载哈工大 + 百度停用词表到 data/stopwords/"""
    STOPWORDS_DIR.mkdir(parents=True, exist_ok=True)
    stopwords = set()
    import urllib.request

    for fname, url in STOPWORD_URLS.items():
        dst = STOPWORDS_DIR / fname
        if dst.exists() and dst.stat().st_size > 0:
            print(f"  [跳过] 已存在 {dst}")
        else:
            print(f"  下载 {url}")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    dst.write_bytes(r.read())
            except Exception as e:
                print(f"  [警告] 下载失败 ({e})，跳过 {fname}")
                continue

        for line in dst.read_text(encoding="utf-8", errors="ignore").splitlines():
            w = line.strip()
            if w:
                stopwords.add(w)

    stopwords |= WIKI_NOISE_WORDS
    merged = STOPWORDS_DIR / "stopwords_merged.txt"
    merged.write_text("\n".join(sorted(stopwords)), encoding="utf-8")
    print(f"  合并停用词表: {len(stopwords)} 个 -> {merged}")
    return stopwords


def iter_batches(lines, n):
    batch = []
    for ln in lines:
        batch.append(ln)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def tokenize_period(period, stopwords, n_workers):
    in_path = PROCESSED_DIR / f"zhwiki_{period}.jsonl"
    out_tokens = PROCESSED_DIR / f"zhwiki_{period}_tokens.txt"
    if not in_path.exists():
        print(f"  [错误] 输入文件不存在: {in_path}")
        return [], Counter()

    n_docs = sum(1 for _ in open(in_path, encoding="utf-8"))
    print(f"  输入: {in_path} ({n_docs} 条文档), {n_workers} 进程")

    doc_strs_all = []
    vocab_raw = Counter()
    vocab_filt = Counter()
    t0 = time.time()

    with open(in_path, encoding="utf-8") as f_in, \
         open(out_tokens, "w", encoding="utf-8") as fout:
        pbar = tqdm(total=n_docs, desc=f"  分词 {period}", unit="doc")
        with Pool(n_workers, initializer=_worker_init, initargs=(list(stopwords),)) as pool:
            for n_lines, sent_lines, doc_strs, vraw, vfilt in pool.imap(
                _process_batch, iter_batches(f_in, BATCH_SIZE)
            ):
                for s in sent_lines:
                    fout.write(s + "\n")
                doc_strs_all.extend(doc_strs)
                vocab_raw += vraw
                vocab_filt += vfilt
                pbar.update(n_lines)
        pbar.close()

    elapsed = time.time() - t0
    print(f"  分词完成: {len(doc_strs_all)} 文档 / 耗时 {elapsed:.0f}s")
    print(f"  过滤前词表大小: {len(vocab_raw)}")
    print(f"  过滤后词表大小: {len(vocab_filt)}")
    print(f"  输出: {out_tokens}")
    return doc_strs_all, vocab_filt


def fit_tfidf(period, doc_token_strs, vocab_filtered):
    if not doc_token_strs:
        print(f"  [跳过] {period} 无文档，跳过 TF-IDF")
        return

    vec = TfidfVectorizer(
        tokenizer=_space_tokenizer,
        token_pattern=None,
        max_df=0.85,
        min_df=5,
        max_features=50000,
        ngram_range=(1, 1),
        lowercase=False,
    )

    print(f"  拟合 TF-IDF ({period}) ...")
    t0 = time.time()
    tfidf_matrix = vec.fit_transform(doc_token_strs)
    elapsed = time.time() - t0
    print(f"  TF-IDF 矩阵: {tfidf_matrix.shape} / 耗时 {elapsed:.0f}s")

    feature_names = vec.get_feature_names_out()
    avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_idx = avg_tfidf.argsort()[::-1][:TFIDF_TOPK]

    print(f"\n  === {period} Top-{TFIDF_TOPK} TF-IDF 关键词 ===")
    print(f"  {'rank':<5}{'word':<14}{'avg_tfidf':<10}")
    top_rows = []
    for rank, i in enumerate(top_idx, 1):
        w = feature_names[i]
        v = float(avg_tfidf[i])
        print(f"  {rank:<5}{w:<14}{v:<10.6f}")
        top_rows.append((rank, w, v))

    print(f"\n  === {period} 锚定词词频核查 ===")
    anchor_words = [
        "人工智能", "算法", "模型", "自然语言处理", "大语言模型",
        "机器学习", "深度学习", "工作", "消费", "信任",
    ]
    print(f"  {'anchor':<12}{'in_vocab':<10}{'doc_freq':<12}{'corpus_freq':<12}")
    for w in anchor_words:
        in_vocab = w in vec.vocabulary_
        df = 0
        if in_vocab:
            col = vec.vocabulary_[w]
            df = int((tfidf_matrix[:, col] > 0).sum())
        cf = vocab_filtered.get(w, 0)
        print(f"  {w:<12}{str(in_vocab):<10}{df:<12}{cf:<12}")

    out_pkl = PROCESSED_DIR / f"tfidf_{period}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({"vectorizer": vec, "matrix": tfidf_matrix}, f)
    print(f"\n  保存: {out_pkl}")

    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    top_csv = tables_dir / f"tfidf_top{TFIDF_TOPK}_{period}.csv"
    with open(top_csv, "w", encoding="utf-8") as f:
        f.write("rank,word,avg_tfidf,period\n")
        for rank, w, v in top_rows:
            f.write(f"{rank},{w},{v:.6f},{period}\n")
    print(f"  保存: {top_csv}")


def main():
    parser = argparse.ArgumentParser(description="jieba 分词 + 停用词过滤 + TF-IDF 特征工程")
    parser.add_argument(
        "period", nargs="?", default="all",
        choices=["2021", "2022", "2026", "all"],
        help="时段标签，默认 all",
    )
    import os
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
        help="并行进程数",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("第 3 步：中文分词与 TF-IDF 特征工程")
    print("=" * 60)

    print("\n[1/3] 准备停用词表 ...")
    stopwords = download_stopwords()

    print(f"\n[2/3] 初始化 jieba（{len(DOMAIN_WORDS)} 个领域词）")
    print(f"  并行进程数: {args.workers}")

    periods = ["2021", "2022", "2026"]
    for period in periods:
        if args.period != "all" and period != args.period:
            continue
        print(f"\n[3/3] 处理 {period} 时段 ...")
        doc_token_strs, vocab_filtered = tokenize_period(period, stopwords, args.workers)
        fit_tfidf(period, doc_token_strs, vocab_filtered)

    print("\n完成。")


if __name__ == "__main__":
    main()
