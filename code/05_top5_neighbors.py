import argparse
import csv
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("outputs/tables")

# 锚定词（5.1）：技术核心 + 社会对照
ANCHOR_WORDS = [
    "人工智能", "算法", "模型", "自然语言处理",
    "工作", "消费", "信任",
]
PERIODS = ["2021", "2022", "2026"]
PERIOD_LABELS = {"2021": "T1", "2022": "T2", "2026": "T3"}

TOPN = 5          # 最终近邻数
CAND_K = 15       # 每个 seed 取候选数
# 稳健近邻阈值：当有 5 个 seed 时 >=3，少 seed 时按比例
def robust_threshold(n_seeds):
    return max(1, int(np.ceil(n_seeds * 0.6)))


def load_seed_models(period):
    """加载某时段全部 seed 的 KeyedVectors，按 seed 升序"""
    kvs = []
    for p in sorted(PROCESSED_DIR.glob(f"w2v_{period}_seed*.kv")):
        kvs.append((p.stem, KeyedVectors.load(str(p), mmap="r")))
    return kvs


def robust_neighbors(anchor, seed_models, topn=TOPN, cand_k=CAND_K):
    """聚合多 seed 的近邻，返回稳健近邻列表 [(neighbor, avg_cos, count), ...]"""
    if not seed_models:
        return None, {}

    # 每个邻居在各 seed 的 top-cand_k 中的余弦值与出现次数
    cand_cos = {}
    cand_count = {}
    in_top5_count = {}  # 在每个 seed 的 top5 中出现次数

    for _, kv in seed_models:
        if anchor not in kv.key_to_index:
            return None, {}
        sims = kv.most_similar(anchor, topn=cand_k)
        top5_set = {w for w, _ in sims[:topn]}
        for w, s in sims:
            cand_cos.setdefault(w, []).append(float(s))
            cand_count[w] = cand_count.get(w, 0) + 1
            if w in top5_set:
                in_top5_count[w] = in_top5_count.get(w, 0) + 1

    n_seeds = len(seed_models)
    thr = robust_threshold(n_seeds)
    robust = []
    for w, counts in cand_count.items():
        # 稳健：在 cand_k 中出现 >= thr 次（用 top5 出现次数更严格，这里用 cand_k 出现）
        if counts >= thr:
            avg_cos = float(np.mean(cand_cos[w]))
            robust.append((w, avg_cos, in_top5_count.get(w, 0)))
    # 排序：先按 top5 出现次数降序，再按平均余弦降序
    robust.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return robust[:topn], {"n_seeds": n_seeds, "threshold": thr}


def build_long_table(results):
    """results: {anchor: {period: (robust_list, meta)}}"""
    rows = []
    for anchor in ANCHOR_WORDS:
        per = results.get(anchor, {})
        for period in PERIODS:
            plist, meta = per.get(period, (None, {}))
            if not plist:
                rows.append({
                    "anchor": anchor, "period": period,
                    "rank": "", "neighbor": "(不在词表)", "avg_cosine": "",
                    "top5_appear_count": "", "n_seeds": meta.get("n_seeds", ""),
                })
                continue
            for rank, (w, cos, cnt) in enumerate(plist, 1):
                rows.append({
                    "anchor": anchor, "period": period,
                    "rank": rank, "neighbor": w,
                    "avg_cosine": round(cos, 4),
                    "top5_appear_count": cnt,
                    "n_seeds": meta.get("n_seeds", ""),
                })
    out = TABLES_DIR / "top5_neighbors_by_period.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "anchor", "period", "rank", "neighbor",
            "avg_cosine", "top5_appear_count", "n_seeds",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"  长表已保存: {out} ({len(rows)} 行)")


def build_comparison(results):
    """并排对比表（含标注），每个锚定词一行组"""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "top5_comparison.csv"
    rows = []
    md_lines = ["# 锚定词 Top-5 近邻三时段并排对比\n"]

    for anchor in ANCHOR_WORDS:
        per = results.get(anchor, {})
        tops = {}
        for period in PERIODS:
            plist, _ = per.get(period, (None, {}))
            tops[period] = [w for w, _, _ in plist] if plist else []

        t1, t2, t3 = set(tops["2021"]), set(tops["2022"]), set(tops["2026"])
        md_lines.append(f"\n## 锚定词：{anchor}\n")
        md_lines.append("| rank | T1 (2021) | T2 (2022) | T3 (2026) | 标注 |")
        md_lines.append("|---|---|---|---|---|")

        max_len = max(len(tops[p]) for p in PERIODS) if tops else 0
        for rank in range(max_len):
            cells = {p: tops[p][rank] if rank < len(tops[p]) else "" for p in PERIODS}
            marks = []
            for p in PERIODS:
                w = cells[p]
                if not w:
                    continue
                # 持续：三时段都出现
                if w in t1 and w in t2 and w in t3:
                    marks.append(f"●{w}")
                elif w in t3 and w not in t1:
                    marks.append(f"▲{w}")
                elif w in t1 and w not in t3:
                    marks.append(f"▼{w}")
            mark_str = " / ".join(marks) if marks else ""
            md_lines.append(
                f"| {rank+1} | {cells['2021']} | {cells['2022']} | {cells['2026']} | {mark_str} |"
            )
            rows.append({
                "anchor": anchor, "rank": rank + 1,
                "T1": cells["2021"], "T2": cells["2022"], "T3": cells["2026"],
                "mark": mark_str,
            })

        # 描述性辅助：T1 与 T3 Top-5 重合个数（描述性，不做推断）
        overlap = len(t1 & t3)
        md_lines.append(f"\n*T1 与 T3 Top-5 重合个数：{overlap}/5（描述性统计，不构成偏移量）*\n")

    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["anchor", "rank", "T1", "T2", "T3", "mark"])
        w.writeheader()
        w.writerows(rows)
    print(f"  对比表已保存: {out} ({len(rows)} 行)")

    md_path = TABLES_DIR / "top5_comparison.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  对比表(Markdown)已保存: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="锚定词 Top-5 稳健近邻提取与并排对比")
    args = parser.parse_args()

    print("=" * 60)
    print("第 5 步：锚定词 Top-5 近邻提取与对比")
    print("=" * 60)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for anchor in ANCHOR_WORDS:
        results[anchor] = {}
        print(f"\n锚定词: {anchor}")
        for period in PERIODS:
            models = load_seed_models(period)
            if not models:
                print(f"  [{period}] 无模型，跳过")
                results[anchor][period] = (None, {})
                continue
            rlist, meta = robust_neighbors(anchor, models)
            if rlist is None:
                print(f"  [{period}] 不在词表 (threshold={meta.get('threshold')})")
                results[anchor][period] = (None, meta)
                continue
            results[anchor][period] = (rlist, meta)
            top5_str = ", ".join(f"{w}({c:.3f}, {k}/{meta['n_seeds']})"
                                 for w, c, k in rlist)
            print(f"  [{period}] 阈值>={meta['threshold']}/{meta['n_seeds']}: {top5_str}")

    print("\n生成表格 ...")
    build_long_table(results)
    build_comparison(results)
    print("\n完成。")


if __name__ == "__main__":
    main()
