import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

from opencc import OpenCC
from tqdm import tqdm

# 假设 utils.py 中定义了 SEED，如果没有请注释掉下面这行并取消注释硬编码
# from utils import SEED
SEED = 42

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
MIN_LEN = 200

# 初始化繁简转换器（全局只加载一次）
CC = OpenCC("t2s")

PERIOD_MAP = {
    "2021": INTERIM_DIR / "raw_2021",
    "2022": INTERIM_DIR / "raw_2022",
    "2026": INTERIM_DIR / "raw_2026",
}


def to_halfwidth(s):
    """全角字符转半角"""
    out = []
    for ch in s:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def clean_text(s):
    """统一文本清洗：全角转半角 + 繁转简 + 压缩空白"""
    if not s:
        return ""
    s = to_halfwidth(s)
    s = CC.convert(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def postprocess_period(period):
    raw_dir = PERIOD_MAP[period]
    if not raw_dir.exists():
        print(f"[错误] 目录不存在: {raw_dir}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"zhwiki_{period}.jsonl"

    json_files = sorted(raw_dir.rglob("wiki_*"))
    if not json_files:
        print(f"  [错误] 未找到 wiki_* 文件: {raw_dir}")
        return

    # 预先统计总行数（跳过空文件）
    n_total = 0
    for jf in json_files:
        try:
            n_total += sum(1 for _ in open(jf, encoding="utf-8"))
        except Exception:
            print(f"  [警告] 跳过无法读取的文件: {jf}")
    print(f"  输入: {raw_dir}/ ({len(json_files)} 个文件, 共 {n_total} 条)")

    seen_hashes = set()
    total_read = 0
    total_short = 0
    total_dup = 0
    total_kept = 0
    total_json_err = 0
    total_blank = 0
    total_len = 0
    t0 = time.time()

    json_err_samples = []

    with open(out_path, "w", encoding="utf-8") as out_f:
        pbar = tqdm(total=n_total, desc=f"  后处理 {period}", unit="条")

        for jf in json_files:
            with open(jf, encoding="utf-8") as f_jf:
                for line in f_jf:
                    line = line.strip()
                    if not line:
                        total_blank += 1
                        continue
                    
                    total_read += 1
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        total_json_err += 1
                        if len(json_err_samples) < 5:
                            json_err_samples.append(line[:200])
                        continue

                    # 【修复 1】核心逻辑必须缩进在 for 循环内部
                    # 【修复 2】使用 .get() 防止缺少 text 字段导致 KeyError
                    text = clean_text(doc.get("text", ""))
                    if not text or len(text) < MIN_LEN:
                        total_short += 1
                        pbar.update(1)
                        continue

                    h = hashlib.md5(text.encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        total_dup += 1
                        pbar.update(1)
                        continue
                    seen_hashes.add(h)

                    # 【修复 3】title 也经过 clean_text，保证繁简和全半角统一
                    title = clean_text(doc.get("title", ""))

                    rec = {
                        "doc_id": doc.get("id", ""),
                        "title": title,
                        "text": text,
                        "period": period,
                        "source": "zhwiki",
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_kept += 1
                    total_len += len(text)
                    pbar.update(1)

        pbar.close()

    elapsed = time.time() - t0
    avg_len = total_len / total_kept if total_kept else 0
    print(f"\n  后处理对比 ({period}):")
    print(f"    总行数:       {n_total:>8}")
    print(f"    空行跳过:     {total_blank:>8}")
    print(f"    JSON 错误:    {total_json_err:>8}")
    print(f"    短文本丢弃:   {total_short:>8}")
    print(f"    重复去除:     {total_dup:>8}")
    print(f"    保留:         {total_kept:>8}")
    print(f"    平均正文长度: {avg_len:>7.0f}")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  输出: {out_path}")

    if json_err_samples:
        print(f"\n  JSON 解析错误样本 (共 {total_json_err} 条):")
        for i, s in enumerate(json_err_samples):
            print(f"    [{i}] {s}...")

    # 【修复 4】使用蓄水池采样代替 readlines()，防止大文件 OOM
    samples = []
    random.seed(SEED)
    with open(out_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(samples) < 2:
                samples.append(line)
            else:
                j = random.randint(0, i)
                if j < 2:
                    samples[j] = line
    
    if samples:
        print("\n  随机样本预览:")
        for s in samples:
            doc = json.loads(s)
            title = doc["title"][:30] if doc["title"] else ""
            text = doc["text"][:120] if doc["text"] else ""
            print(f"    [{title}] {text}...\n")


def main():
    parser = argparse.ArgumentParser(description="中文维基百科后处理（全角→半角 + 繁→简 + 去重）")
    parser.add_argument(
        "period", nargs="?", default="all",
        choices=["2021", "2022", "2026", "all"],
        help="时段标签，默认 all"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("第 2 步：后处理（半角 + 繁简 + 去重）")
    print("=" * 60)

    for period in ["2021", "2022", "2026"]:
        if args.period != "all" and period != args.period:
            continue
        print(f"\n处理 {period} 时段 ...")
        postprocess_period(period)

    print("\n完成。")


if __name__ == "__main__":
    main()