import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import seaborn as sns

PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("outputs/tables")
FIG_DIR = Path("outputs/figures")
PERIODS = ["2021", "2022", "2026"]
PERIOD_LABEL = {"2021": "T1 (2021)", "2022": "T2 (2022)", "2026": "T3 (2026)"}

# 中文字体
for fp in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
           "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"]:
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False


def collect_stats():
    """流式统计各时段文档数与正文长度（采样以加速长度分布）"""
    counts, lengths = {}, {}
    for p in PERIODS:
        path = PROCESSED_DIR / f"zhwiki_{p}.jsonl"
        c = 0
        lens_sample = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                t = doc.get("text", "")
                c += 1
                if i % 20 == 0:  # 采样 5% 估计长度分布
                    lens_sample.append(len(t))
        counts[p] = c
        lengths[p] = lens_sample
    return counts, lengths


def fig_doc_counts(counts):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [PERIOD_LABEL[p] for p in PERIODS]
    vals = [counts[p] for p in PERIODS]
    bars = ax.bar(labels, vals, color=["#4C72B0", "#55A868", "#C44E52"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01,
                f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("文档数")
    ax.set_title("各时段文档数量")
    ax.set_ylim(0, max(vals) * 1.12)
    fig.tight_layout()
    out = FIG_DIR / "fig1_doc_counts.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  保存 {out}")


def fig_text_length(lengths):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [lengths[p] for p in PERIODS]
    labels = [PERIOD_LABEL[p] for p in PERIODS]
    parts = ax.violinplot(data, showmedians=True)
    for body, col in zip(parts["bodies"], ["#4C72B0", "#55A868", "#C44E52"]):
        body.set_facecolor(col)
        body.set_alpha(0.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_ylabel("正文长度（字符数，5% 采样）")
    ax.set_title("各时段正文长度分布")
    # 标注中位数
    for i, d in enumerate(data, 1):
        ax.text(i, max(d) * 0.95, f"中位数={int(np.median(d))}", ha="center", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "fig2_text_length.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  保存 {out}")


def fig_top5_heatmap():
    """锚定词 Top-5 三时段并排，颜色标注持续/新出现/消失"""
    cmp_csv = TABLES_DIR / "top5_comparison.csv"
    if not cmp_csv.exists():
        print("  [跳过] 缺少 top5_comparison.csv")
        return
    rows = list(csv.DictReader(open(cmp_csv, encoding="utf-8")))
    # 按 anchor 分组
    anchors = []
    cur = None
    groups = {}
    for r in rows:
        a = r["anchor"]
        groups.setdefault(a, []).append(r)
    anchor_list = list(groups.keys())
    n_anchors = len(anchor_list)

    fig, axes = plt.subplots(n_anchors, 1, figsize=(9, 2.2 * n_anchors))
    if n_anchors == 1:
        axes = [axes]

    def cell_color(words, all_t1, all_t3, all_t2):
        # 为该单元格（一个 rank 的三时段词）着色
        return None

    for ax, anchor in zip(axes, anchor_list):
        prows = groups[anchor]
        ranks = [int(r["rank"]) for r in prows]
        max_rank = max(ranks)
        t1 = [r["T1"] for r in prows]
        t2 = [r["T2"] for r in prows]
        t3 = [r["T3"] for r in prows]
        table_data = list(zip(t1, t2, t3))
        set_t1, set_t2, set_t3 = set(t1) & set(t3), set(t1) & set(t2) & set(t3), set(t1) & set(t3)
        # 着色矩阵：对每个 cell
        colors = []
        for r in range(max_rank):
            row_colors = []
            for ci, colvals in enumerate([t1, t2, t3]):
                w = colvals[r] if r < len(colvals) else ""
                if not w:
                    row_colors.append("#EEEEEE")
                elif w in set(t1) and w in set(t2) and w in set(t3):
                    row_colors.append("#A8D8A8")  # 持续 绿
                elif ci == 2 and w not in set(t1):  # T3 新出现
                    row_colors.append("#F4B860")  # 橙
                elif ci == 0 and w not in set(t3):  # T1 消失
                    row_colors.append("#E88C8C")  # 红
                else:
                    row_colors.append("#FFFFFF")
            colors.append(row_colors)

        ax.axis("off")
        ax.set_title(f"锚定词：{anchor}", fontsize=12, fontweight="bold", loc="left")
        tbl = ax.table(cellText=table_data,
                       rowLabels=[str(i + 1) for i in range(max_rank)],
                       colLabels=["T1 (2021)", "T2 (2022)", "T3 (2026)"],
                       cellColours=colors, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
    fig.tight_layout()
    out = FIG_DIR / "fig3_top5_neighbors.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存 {out}")

    # 图例说明
    print("  颜色说明: 绿=持续(三时段均在Top5), 橙=新出现(T3进入), 红=消失(T1有T3无)")


def main():
    print("=" * 60)
    print("第 8 步：探索性分析与可视化")
    print("=" * 60)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] 统计文档数与正文长度 ...")
    counts, lengths = collect_stats()
    for p in PERIODS:
        print(f"  {p}: {counts[p]} 篇, 长度中位数 {int(np.median(lengths[p]))}")

    print("\n[2/3] 绘图 1 文档数量柱状图 ...")
    fig_doc_counts(counts)
    print("[2/3] 绘图 2 正文长度分布 ...")
    fig_text_length(lengths)

    print("\n[3/3] 绘图 3 锚定词 Top-5 变化 ...")
    fig_top5_heatmap()
    print("\n完成。")


if __name__ == "__main__":
    main()
