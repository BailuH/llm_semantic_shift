import argparse
import os
import time
from pathlib import Path

from gensim.models import Word2Vec
from tqdm import tqdm

from utils import SEED

PROCESSED_DIR = Path("data/processed")

# 训练参数（三时段完全一致，写进报告 6.2）
VECTOR_SIZE = 200
WINDOW = 8
MIN_COUNT = 10
SG = 1
EPOCHS = 10
WORKERS = max(1, (os.cpu_count() or 2) - 2)

# 多 seed 稳健近邻（5.3）：5 次训练，出现 >=3 次视为稳健近邻
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

# 锚定词（5.1），核查是否在词表中、词频是否足够
ANCHOR_WORDS = [
    "人工智能", "算法", "模型", "自然语言处理",
    "工作", "消费", "信任",
]


class WikiSentences:
    """流式读取 tokens.txt，每行一个句子（空格分隔词元）。gensim 每轮 epoch 重新迭代。"""

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                toks = line.split()
                if toks:
                    yield toks


def count_sentences(path):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def train_one(period, seed, n_sentences, workers):
    tokens_path = PROCESSED_DIR / f"zhwiki_{period}_tokens.txt"
    if not tokens_path.exists():
        print(f"  [错误] 缺少 {tokens_path}")
        return None

    sentences = WikiSentences(tokens_path)
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        workers=workers,
        seed=seed,
        epochs=EPOCHS,
    )
    out_kv = PROCESSED_DIR / f"w2v_{period}_seed{seed}.kv"
    model.wv.save(str(out_kv))
    # 同时保存完整模型，便于后续复现
    out_model = PROCESSED_DIR / f"w2v_{period}_seed{seed}.model"
    model.save(str(out_model))
    return model


def train_period(period, seeds, workers):
    tokens_path = PROCESSED_DIR / f"zhwiki_{period}_tokens.txt"
    if not tokens_path.exists():
        print(f"  [跳过] {tokens_path} 不存在")
        return

    n_sent = count_sentences(tokens_path)
    print(f"  输入: {tokens_path} ({n_sent} 句)")

    first_kv = None
    for seed in seeds:
        t0 = time.time()
        model = train_one(period, seed, n_sent, workers)
        if model is None:
            continue
        if first_kv is None:
            first_kv = model.wv
        elapsed = time.time() - t0
        print(f"  seed={seed} 词表大小={len(model.wv)} 耗时={elapsed:.0f}s "
              f"-> w2v_{period}_seed{seed}.kv")

    # 锚定词核查（基于第一个 seed 的词表）
    if first_kv is not None:
        print(f"\n  === {period} 锚定词核查 (seed={seeds[0]}) ===")
        print(f"  {'anchor':<12}{'in_vocab':<10}{'count':<10}")
        for w in ANCHOR_WORDS:
            ok = w in first_kv.key_to_index
            cnt = int(first_kv.get_vecattr(w, "count")) if ok else 0
            print(f"  {w:<12}{str(ok):<10}{cnt:<10}")
    print()


def main():
    parser = argparse.ArgumentParser(description="分时段独立训练 Word2Vec（多 seed 稳健近邻）")
    parser.add_argument(
        "period", nargs="?", default="all",
        choices=["2021", "2022", "2026", "all"],
        help="时段标签，默认 all",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="训练用 seed 列表（默认 42-46，5 次）",
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    print("=" * 60)
    print("第 4 步：分时段 Word2Vec 训练")
    print("=" * 60)
    print(f"  参数: vector_size={VECTOR_SIZE} window={WINDOW} min_count={MIN_COUNT} "
          f"sg={SG} epochs={EPOCHS} workers={args.workers}")
    print(f"  seeds: {args.seeds}")

    periods = ["2021", "2022", "2026"]
    for period in periods:
        if args.period != "all" and period != args.period:
            continue
        print(f"\n处理 {period} 时段 ...")
        train_period(period, args.seeds, args.workers)

    print("完成。")


if __name__ == "__main__":
    main()
