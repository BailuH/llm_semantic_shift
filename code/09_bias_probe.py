"""偏见探针实验（《人机对齐》附录）：经典性别类比 + WEAT 式关联测试。

探索性附录，不构成总体推断。方法学口径：
- 不计算跨时段相似度差值，只看 s(w) 的方向和排序变化（与方案 6.2 一致）
- 单边关联分数 s(w)，非完整 WEAT 统计检验（词数不足不做 t 检验）
- 必给替代解释：维基编辑群体构成变化、词频不平衡、w2v 训练随机性
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from gensim.models import KeyedVectors

from utils import SEED

PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("outputs/tables")
FIG_DIR = Path("outputs/figures")
PERIODS = ["2021", "2022", "2026"]
PERIOD_LABEL = {"2021": "T1 (2021)", "2022": "T2 (2022)", "2026": "T3 (2026)"}
SEEDS = [42, 43]

# 中文字体
for fp in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
           "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"]:
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 词集定义
# ============================================================

GENDER_MALE = ["男人", "男性", "男孩", "父亲", "儿子", "丈夫", "哥哥"]
GENDER_FEMALE = ["女人", "女性", "女孩", "母亲", "女儿", "妻子", "姐姐"]

ANALOGY_VERIFICATION = [
    ("国王", "男人", "女人", "王后"),
]
ANALOGY_BIAS_WORDS = [
    "医生", "护士", "工程师", "程序员", "教授", "司机",
    "秘书", "教师", "科学家", "建筑师", "保姆", "厨师",
]

WEAT_GROUPS = {
    "职业-高端": ["医生", "律师", "工程师", "教授", "科学家", "程序员",
                "总裁", "博士", "建筑师", "院士"],
    "职业-服务": ["护士", "教师", "保姆", "秘书", "服务员", "清洁工",
                "厨师", "司机", "售货员", "理发师"],
    "理工科": ["数学", "物理", "化学", "计算机", "算法", "编程",
              "工程", "数据", "技术", "机器"],
    "文科": ["文学", "历史", "艺术", "音乐", "哲学", "语言",
            "诗歌", "绘画", "舞蹈", "写作"],
    "AI相关": ["人工智能", "算法", "模型", "机器学习", "深度学习", "自然语言处理"],
    "家庭词": ["家庭", "家务", "育儿", "厨房", "家庭主妇", "主妇",
              "缝纫", "照料"],
}
LLM_ONLY_WORDS = ["大语言模型"]  # 仅 T3 存在


def load_models(period):
    """加载某时段全部 seed 的 KeyedVectors"""
    kvs = []
    for seed in SEEDS:
        p = PROCESSED_DIR / f"w2v_{period}_seed{seed}.kv"
        if p.exists():
            kvs.append((seed, KeyedVectors.load(str(p), mmap="r")))
    return kvs


def avg_cos(kv, target, group):
    """目标词与词集的平均余弦"""
    sims = []
    for w in group:
        if w in kv.key_to_index and target in kv.key_to_index:
            sims.append(float(kv.similarity(target, w)))
    return sum(sims) / len(sims) if sims else None


def s_score(kv, target, male_set, female_set):
    """s(w) = mean_cos(w, male) - mean_cos(w, female)。正=偏男，负=偏女"""
    sm = avg_cos(kv, target, male_set)
    sf = avg_cos(kv, target, female_set)
    if sm is None or sf is None:
        return None, sm, sf
    return sm - sf, sm, sf


def tag_bias(s):
    if s is None:
        return "NA"
    if s > 0.02:
        return "偏男"
    if s < -0.02:
        return "偏女"
    return "中性"


# ============================================================
# 模块 1：经典性别类比
# ============================================================

def module_analogy():
    print("\n[模块1] 经典性别类比任务")
    rows = []
    for period in PERIODS:
        models = load_models(period)
        for seed, kv in models:
            # 验证类比
            for a, b, c, expected in ANALOGY_VERIFICATION:
                if all(w in kv.key_to_index for w in [a, b, c]):
                    res = kv.most_similar(positive=[a, c], negative=[b], topn=3)
                    top3 = "; ".join(f"{w}({s:.3f})" for w, s in res)
                    hit = expected in [w for w, _ in res]
                    print(f"  [{period} seed{seed}] {a}-{b}+{c} = {top3}"
                          f"  期望={expected} {'✓' if hit else '✗'}")
                    rows.append({
                        "period": period, "seed": seed,
                        "type": "验证", "X": a, "a-b+c": f"{a}-{b}+{c}",
                        "top3": top3, "expected": expected, "hit": hit,
                    })

            # 偏见诊断类比：双向
            for X in ANALOGY_BIAS_WORDS:
                if X not in kv.key_to_index:
                    continue
                if all(w in kv.key_to_index for w in ["男人", "女人"]):
                    # X - 男人 + 女人 → ?
                    res_f = kv.most_similar(positive=[X, "女人"],
                                            negative=["男人"], topn=3)
                    top3_f = "; ".join(f"{w}({s:.3f})" for w, s in res_f)
                    # X - 女人 + 男人 → ?
                    res_m = kv.most_similar(positive=[X, "男人"],
                                            negative=["女人"], topn=3)
                    top3_m = "; ".join(f"{w}({s:.3f})" for w, s in res_m)
                    rows.append({
                        "period": period, "seed": seed,
                        "type": "偏见诊断", "X": X,
                        "a-b+c": f"{X}-男人+女人",
                        "top3": top3_f, "expected": "", "hit": "",
                    })
                    rows.append({
                        "period": period, "seed": seed,
                        "type": "偏见诊断", "X": X,
                        "a-b+c": f"{X}-女人+男人",
                        "top3": top3_m, "expected": "", "hit": "",
                    })

    # 跨 seed 一致性标注
    by_key = {}
    for r in rows:
        if r["type"] != "偏见诊断":
            continue
        key = (r["period"], r["X"], r["a-b+c"])
        by_key.setdefault(key, []).append(r["top3"])
    for r in rows:
        if r["type"] != "偏见诊断":
            r["consistent"] = ""
            continue
        key = (r["period"], r["X"], r["a-b+c"])
        tops = by_key[key]
        r["consistent"] = "是" if len(set(tops)) == 1 else "否"

    out = TABLES_DIR / "bias_analogy.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "period", "seed", "type", "X", "a-b+c",
            "top3", "expected", "hit", "consistent"])
        w.writeheader()
        w.writerows(rows)
    print(f"  保存: {out} ({len(rows)} 行)")
    return rows


# ============================================================
# 模块 2：WEAT 式关联测试
# ============================================================

def module_weat():
    print("\n[模块2] WEAT 式关联测试")
    rows = []
    for period in PERIODS:
        models = load_models(period)
        for group, words in WEAT_GROUPS.items():
            for w in words:
                s_vals, sm_vals, sf_vals = [], [], []
                for seed, kv in models:
                    if w not in kv.key_to_index:
                        continue
                    s, sm, sf = s_score(kv, w, GENDER_MALE, GENDER_FEMALE)
                    if s is not None:
                        s_vals.append(s)
                        sm_vals.append(sm)
                        sf_vals.append(sf)
                if not s_vals:
                    rows.append({
                        "word": w, "group": group, "period": period,
                        "s_diff": "", "s_male": "", "s_female": "",
                        "tag": "不在词表", "llm_only": False,
                        "n_seeds": 0,
                    })
                    continue
                s_mean = float(np.mean(s_vals))
                rows.append({
                    "word": w, "group": group, "period": period,
                    "s_diff": round(s_mean, 4),
                    "s_male": round(float(np.mean(sm_vals)), 4),
                    "s_female": round(float(np.mean(sf_vals)), 4),
                    "tag": tag_bias(s_mean),
                    "llm_only": False,
                    "n_seeds": len(s_vals),
                })

    # 模块 3：LLM 原生词（仅 T3）
    print("\n[模块3] LLM 原生词偏见（仅 T3）")
    for w in LLM_ONLY_WORDS:
        models = load_models("2026")
        s_vals, sm_vals, sf_vals = [], [], []
        for seed, kv in models:
            if w not in kv.key_to_index:
                continue
            s, sm, sf = s_score(kv, w, GENDER_MALE, GENDER_FEMALE)
            if s is not None:
                s_vals.append(s)
                sm_vals.append(sm)
                sf_vals.append(sf)
        if s_vals:
            s_mean = float(np.mean(s_vals))
            print(f"  {w} (T3): s={s_mean:+.4f} "
                  f"男={np.mean(sm_vals):.4f} 女={np.mean(sf_vals):.4f} "
                  f"{tag_bias(s_mean)}")
            rows.append({
                "word": w, "group": "AI相关", "period": "2026",
                "s_diff": round(s_mean, 4),
                "s_male": round(float(np.mean(sm_vals)), 4),
                "s_female": round(float(np.mean(sf_vals)), 4),
                "tag": tag_bias(s_mean), "llm_only": True,
                "n_seeds": len(s_vals),
            })
        else:
            print(f"  {w}: T3 不在词表")

    # 补齐：LLM 原生词在 T1/T2 标记为不存在
    for w in LLM_ONLY_WORDS:
        for period in ["2021", "2022"]:
            rows.append({
                "word": w, "group": "AI相关", "period": period,
                "s_diff": "", "s_male": "", "s_female": "",
                "tag": "不在词表", "llm_only": True, "n_seeds": 0,
            })

    out = TABLES_DIR / "bias_weat.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "word", "group", "period", "s_diff", "s_male", "s_female",
            "tag", "llm_only", "n_seeds"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  保存: {out} ({len(rows)} 行)")

    # 打印 T1 vs T3 对比摘要
    print("\n  === T1 vs T3 偏见移动摘要（按 s_diff 排序）===")
    by_word = {}
    for r in rows:
        by_word.setdefault(r["word"], {})[r["period"]] = r
    print(f"  {'word':<10}{'group':<12}{'T1 s':<10}{'T3 s':<10}"
          f"{'Δ(T3-T1)':<12}{'方向'}")
    moved = []
    for w, per in by_word.items():
        r1 = per.get("2021")
        r3 = per.get("2026")
        if not r1 or not r3:
            continue
        if r1["s_diff"] == "" or r3["s_diff"] == "":
            continue
        d1 = float(r1["s_diff"])
        d3 = float(r3["s_diff"])
        delta = d3 - d1
        direction = "偏男↑" if delta > 0.01 else "偏女↑" if delta < -0.01 else "稳定"
        moved.append((w, r1["group"], d1, d3, delta, direction))
    moved.sort(key=lambda x: x[4])
    for w, g, d1, d3, delta, direction in moved:
        print(f"  {w:<10}{g:<12}{d1:<10.4f}{d3:<10.4f}"
              f"{delta:<+12.4f}{direction}")
    return rows


# ============================================================
# 模块 4：可视化
# ============================================================

def module_visualize(weat_rows):
    print("\n[模块4] 可视化 fig4_bias_trend.png")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 选取代表性词：每组选 |s| 最大的 3 个（基于 T1）
    by_word = {}
    for r in weat_rows:
        by_word.setdefault(r["word"], {})[r["period"]] = r
    by_group = {}
    for w, per in by_word.items():
        r1 = per.get("2021")
        if not r1 or r1["s_diff"] == "":
            continue
        by_group.setdefault(r1["group"], []).append((w, abs(float(r1["s_diff"]))))
    selected = []
    for g, lst in by_group.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        selected.extend([w for w, _ in lst[:3]])
    # 加入 LLM 原生词
    selected.append("大语言模型")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = {
        "职业-高端": "#C44E52", "职业-服务": "#55A868",
        "理工科": "#4C72B0", "文科": "#8172B2",
        "AI相关": "#CCB974", "家庭词": "#64B5CD",
    }
    group_of = {r["word"]: r["group"]
                for r in weat_rows if r["group"]}

    for w in selected:
        xs, ys = [], []
        for i, p in enumerate(PERIODS):
            r = by_word.get(w, {}).get(p)
            if r and r["s_diff"] != "":
                xs.append(i)
                ys.append(float(r["s_diff"]))
        if not xs:
            continue
        g = group_of.get(w, "AI相关")
        ax.plot(xs, ys, "o-", label=f"{w} ({g})",
                color=colors.get(g, "gray"), markersize=5, alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.axhline(0.02, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axhline(-0.02, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(PERIODS)))
    ax.set_xticklabels([PERIOD_LABEL[p] for p in PERIODS])
    ax.set_ylabel("s(w) = cos(w,男) − cos(w,女)")
    ax.set_title("性别偏见关联得分历时趋势（正=偏男，负=偏女）")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8,
              borderaxespad=0)
    ax.text(0.01, 0.02, "虚线=中性阈值 ±0.02", transform=ax.transAxes,
            fontsize=8, color="gray")
    fig.tight_layout()
    out = FIG_DIR / "fig4_bias_trend.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("第 9 步（附录）：词向量偏见探针")
    print("=" * 60)
    print("  基于已有 Word2Vec 模型，不重新训练")
    print(f"  性别轴: 男{len(GENDER_MALE)}词 / 女{len(GENDER_FEMALE)}词")
    print(f"  时段: {PERIODS}, seeds: {SEEDS}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    analogy_rows = module_analogy()
    weat_rows = module_weat()
    module_visualize(weat_rows)

    print("\n完成。")


if __name__ == "__main__":
    main()
