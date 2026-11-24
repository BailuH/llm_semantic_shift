# 历时 Word2Vec 视角下的中文语义变迁（2021—2026）

## 项目结构

```
llm_semantic_shift/
├── code/
│   ├── utils.py                 # 随机种子（SEED=42）、公共工具
│   ├── 01_extract.py             # WikiExtractor 纯文本提取
│   ├── 02_postprocess.py         # 后处理（全角→半角、繁→简、去重）
│   ├── 03_tokenize.py            # jieba 分词 + 停用词过滤 + TF-IDF 向量化
│   ├── 04_train_word2vec.py      # 分时段独立训练 Word2Vec
│   ├── 05_top5_neighbors.py      # 锚定词 Top-5 近邻提取与并排对比
│   ├── 06_tfidf_similarity.py    # TF-IDF 余弦相似文本检索
│   └── 07_period_classify.py     # 时段分类（三分类：T1/T2/T3，MultinomialNB / LinearSVC）
├── data/
│   ├── raw/                      # 原始 *.xml.bz2 dump 文件
│   ├── interim/                  # WikiExtractor 中间 JSON 输出
│   └── processed/                # 后处理后 JSONL + 分词/向量化/模型
├── outputs/
│   ├── figures/
│   └── tables/
├── requirements.txt
└── README.md
```

## 环境配置

```bash
uv sync
# 或
pip install -r requirements.txt
```

核心依赖：`wikiextractor`、`opencc-python-reimplemented`、`jieba`、`gensim`、`scikit-learn`、`numpy`、`scipy`、`matplotlib`、`seaborn`。

固定随机种子 `SEED=42`（`code/utils.py`）。Word2Vec 的 `workers>1` 会引入非确定性，固定 seed 只能降低而不能完全消除训练随机性。

## 运行顺序

```
01_extract.py → 02_postprocess.py → 03_tokenize.py → 04_train_word2vec.py → 05_top5_neighbors.py → 06_tfidf_similarity.py → 07_period_classify.py
```

每个脚本的输出作为下一个脚本的输入，按顺序逐一执行。

---

## 第 1 步：数据获取与时段划分

### 时段设计

本项目原计划覆盖四个时段（2021/2022/2023/2025），实际执行中调整为三个时段并重编号如下：

| 时段标签 | 年份 | 快照日期 | 角色 | Wikipedia XML 命名空间 | 数据来源 |
|---|---|---|---|---|---|
| T1 | 2021 | 2021-06-01 | ChatGPT 前基线 | export-0.10 | Internet Archive |
| T2 | 2022 | 2022-05-01 | 过渡年 | export-0.10 | Internet Archive |
| T3 | 2026 | 2026-06-01 | 延伸检验 | export-0.11 | dumps.wikimedia.org |

> **T3（2023）被舍弃**：维基百科官方在 2022—2023 年间进行技术维护，dumps.wikimedia.org 上未提供 2023 年稳定快照，Internet Archive 中也未能找到。报告中注明“缺少 2023 时段，无法区分渐变与突变”。
>
> **T2 使用 5 月快照**：同样因技术维护，2022 年 6 月快照不可获取，以 5 月快照作为最近替代。

### 下载方法

#### 2026 年数据

来源：`dumps.wikimedia.org`

```bash
curl -L -o data/raw/zhwiki-20260601-pages-articles-multistream.xml.bz2 \
  https://dumps.wikimedia.org/zhwiki/20260601/zhwiki-20260601-pages-articles-multistream.xml.bz2
```

#### 2021 / 2022 年数据

来源：Internet Archive（dumps.wikimedia.org 不保留较早历史快照）

1. 访问 https://archive.org/details/wikimedia-archive
2. 搜索 `zhwiki-YYYYMMDD-pages-articles-multistream.xml.bz2`
3. 下载对应文件，放入 `data/raw/`

### 数据文件

| 文件 | 存放位置 | 格式 |
|---|---|---|
| `zhwiki-20210601-pages-articles-multistream.xml.bz2` | `data/raw/` | multistream bz2 |
| `zhwiki-20220501-pages-articles-multistream.xml.bz2` | `data/raw/` | multistream bz2 |
| `zhwiki-20260601-pages-articles-multistream.xml.bz2` | `data/raw/` | multistream bz2 |

> `multistream` 变体内容与标准 `pages-articles` 相同，仅压缩格式略有差异。

### WikiExtractor 纯文本提取

执行 `code/01_extract.py`，使用 WikiExtractor 3.1.0（Stanford NLP 维护，零依赖）从 bz2 提取纯文本：

```bash
.venv/bin/python code/01_extract.py 2021
```

WikiExtractor 完成以下工作：

1. 自动检测 Wikipedia XML 命名空间版本（0.10 / 0.11）
2. 仅保留主命名空间（`ns=0`）的百科条目
3. 过滤正文长度不足 200 字符的页面
4. **完整的 Wiki 标记清理**（<ref>、{{}}、[[File:]]、=== 章节标题、<gallery>、<imagemap>、代码块、引用、表格、外部链接等）
5. 输出多文件 JSON 到 `data/interim/raw_{period}/`

参数 `--no-templates`：中文维基大量使用 Lua Scribunto 模块（`{{#invoke:...}}`），WikiExtractor 无法完全展开，直接剥离模板避免残留。

输出文件：

| 中间目录 | 对应时段 | 内容 |
|---|---|---|
| `data/interim/raw_2021/` | T1 | 多个 JSON 文件（每 ~1MB 拆分） |
| `data/interim/raw_2022/` | T2 | 同上 |
| `data/interim/raw_2026/` | T3 | 同上 |

JSON 格式：`{"id": "13", "title": "数学", "text": "...", ...}`

`period` 字段由 dump 快照日期确定性赋值，代表“该条目在此快照中存在”，而非“该条目写于该年”。

### 解析统计（报告 3.3）

| 指标 | T1 (2021) | T2 (2022) | T3 (2026) |
|---|---|---|---|
| WikiExtractor 产出条数 | — | — | — |
| 后处理保留 | 463,376 | 508,512 | 658,288 |
| 正文长度中位数 | 533 | 516 | 522 |

---

## 第 2 步：后处理（半角 + 繁简 + 去重）

执行 `code/02_postprocess.py`，读取 WikiExtractor 的 JSON 输出，做中文专用后处理：

```bash
.venv/bin/python code/02_postprocess.py 2021
.venv/bin/python code/02_postprocess.py 2022
.venv/bin/python code/02_postprocess.py 2026
```

| 步骤 | 操作 | 工具 | 理由 |
|---|---|---|---|
| 1 | 全角数字/字母 → 半角 | `to_halfwidth()` | 统一数字字母宽度，中文标点保留 |
| 2 | 繁体 → 简体 | OpenCC `t2s` | 避免"數學"和"数学"成为两个独立词 |
| 3 | `\s+` → ` ` 合并空白 | `re.sub` | 规范化空白 |
| 4 | 长度 < 200 过滤 | — | 移除后处理后过短页面 |
| 5 | MD5 去重 | `hashlib` | 移除跨时段重复正文 |

### 后处理前后对比表（报告 4.4）

| 指标 | T1 (2021) | T2 (2022) | T3 (2026) |
|---|---|---|---|
| 后处理保留 | 463,376 | 508,512 | 658,288 |
| 正文长度中位数 | 533 | 516 | 522 |

> 三时段规模递增（T3 比 T1 多 42%），正文长度中位数相近（516—533），说明文体无系统性变化，Word2Vec 上下文窗口覆盖不受影响。

输出文件：

| 中间文件 | 对应时段 | 格式 |
|---|---|---|
| `data/processed/zhwiki_2021.jsonl` | T1 | JSONL（每行一条 JSON） |
| `data/processed/zhwiki_2022.jsonl` | T2 | 同上 |
| `data/processed/zhwiki_2026.jsonl` | T3 | 同上 |

JSONL schema：`{"doc_id": "13", "title": "数学", "text": "...", "period": "2021", "source": "zhwiki"}`

---

## 第 3 步：中文分词与 TF-IDF 特征工程

执行 `code/03_tokenize.py`，读取 `data/processed/zhwiki_{period}.jsonl`（JSONL 格式），输出分词结果和 TF-IDF：

```bash
.venv/bin/python code/03_tokenize.py 2021
.venv/bin/python code/03_tokenize.py 2022
.venv/bin/python code/03_tokenize.py 2026
```

**注意**：JSONL 已通过 02_postprocess 完成全角→半角和繁→简转换，jieba 分词直接在简体纯文本上操作。

### 分词方案

| 组件 | 配置 | 理由 |
|---|---|---|
| jieba 分词 | 精确模式（`cut_all=False`） | 选最优成词路径，避免冗余共现污染近邻结构 |
| 自定义词典 | 14 个领域词（如“大模型”“消费降级”“远程办公”） | 将多字组合锁定为单一词元 |
| 停用词 | 哈工大 + 百度停用词表并集 + 维基噪声词 | 自动从 GitHub 下载 |
| 句子切分 | `。！？!?；;` 断句 | Word2Vec 按句子训练 |

### TF-IDF 参数

```python
TfidfVectorizer(
    max_df=0.85, min_df=5, max_features=50000, ngram_range=(1, 1)
)
```

每个时段独立拟合，输出 Top-20 TF-IDF 关键词（报告 5.3）。

### Top-20 TF-IDF 关键词（报告 5.3）

| 排名 | T1 (2021) | T2 (2022) | T3 (2026) |
|---|---|---|---|
| 1 | 中国 | 中国 | 市镇 |
| 2 | 日本 | 车站 | 中国 |
| 3 | 美国 | 位于 | 法国 |
| 4 | 车站 | 美国 | 位于 |
| 5 | 香港 | 日本 | 美国 |
| 6 | 台湾 | 香港 | 车站 |
| 7 | 位于 | 市镇 | 日本 |
| 8 | — | 台湾 | 香港 |
| 9-20 | 公司/电影/担任/大学/国家/历史/作品… | 公司/电影/法国/担任/大学… | 台湾/担任/电影/政治/大学… |

> 三时段 Top-20 高度重叠（地名+通用动词为主），说明全局词分布稳定；T3 "市镇""法国"上升、"政治"新进 Top-20 反映条目构成的微调。

### 锚定词词频核查（报告 5.1 / 6.2）

| 锚定词 | T1 语料频次 | T2 语料频次 | T3 语料频次 | 备注 |
|---|---|---|---|---|
| 人工智能 | 1,929 | 2,228 | **5,412** | T3 大幅增长 |
| 算法 | 7,432 | 7,788 | **10,052** | T3 增长 |
| 模型 | 22,863 | 24,447 | **34,824** | T3 增长 |
| 自然语言处理 | 139 | 154 | 257 | 低频，近邻可靠性低 |
| 大语言模型 | **0** | **0** | **82** | T3 新出现 |
| 机器学习 | 466 | 523 | 1,018 | T3 翻倍 |
| 深度学习 | 199 | 232 | 474 | T3 翻倍 |
| 工作 | 149,371 | 161,949 | 215,953 | 高频稳定（对照） |
| 消费 | 5,965 | 6,479 | 8,903 | 高频稳定（对照） |
| 信任 | 7,533 | 8,185 | 11,140 | 高频稳定（对照） |

### 输出文件

| 中间文件 | 内容 | 用途 |
|---|---|---|
| `data/processed/zhwiki_{period}_tokens.txt` | 每行一条分词后的句子（空格分隔） | Word2Vec 训练 |
| `data/processed/tfidf_{period}.pkl` | TF-IDF vectorizer + matrix | 相似检索 / 时段分类 |
| `data/stopwords/*.txt` | 停用词表 | 分词过滤 |

## 输出文件与报告对应关系

| 报告章节 | 输出文件 |
|---|---|
| 3.2 数据来源与快照日期 | 本 README |
| 3.3 时段规模统计 | `01_extract.py` + `02_postprocess.py` 运行日志 |
| 4.4 清洗前后对比表 | `02_postprocess.py` 运行日志（填入上表） |
| 5.1 分词方案 | `03_tokenize.py` 参数（见上） |
| 5.3 各时段 TF-IDF 关键词 | `outputs/tables/tfidf_top20_{period}.csv` |
| 6.2 锚定词 Top-5 近邻 | `outputs/tables/top5_neighbors_by_period.csv` |
| 6.3 时段分类评估 | `outputs/tables/classification_summary.csv` + `confusion_matrix.csv` |
| 6.1 可视化 | `outputs/figures/` |

## 第 6 步：TF-IDF 余弦相似文本检索（T3 语料）

执行 `code/06_tfidf_similarity.py`，在 T3(2026) 全量语料上拟合的 TF-IDF 上检索 3 条查询文本的 Top-5 相似条目。

| 查询 | Top-1 结果 | 余弦 |
|---|---|---|
| 大语言模型/深度学习/文本生成… | 深度学习 | 0.412 |
| 推荐算法/短视频/电商… | 上海国际短视频中心 | 0.404 |
| 灵活就业/远程办公/社会保障… | 中华人民共和国社会保障卡 | 0.368 |

> 局限：TF-IDF 只捕捉词汇重叠，无法处理同义词，也无法反映同一词跨时段语义位置的变化——这正是 Word2Vec 近邻分析的必要性所在。输出：`outputs/tables/similar_t3.csv`。

---

## 第 7 步：时段分类建模（三分类）

执行 `code/07_period_classify.py`，每时段均衡采样 80,000 篇（共 240,000 篇），8:2 分层划分，TF-IDF 仅在训练集上 fit。

### 分类结果（报告 6.3）

| 模型 | Accuracy | 基线（多数类） | 提升 |
|---|---|---|---|
| MultinomialNB | 0.3678 | 0.3333 | +0.0344 |
| LinearSVC | 0.3625 | 0.3333 | +0.0292 |

### 混淆矩阵（LinearSVC，行=真实，列=预测）

| | 预测 2021 | 预测 2022 | 预测 2026 |
|---|---|---|---|
| **真实 2021** | 5,839 | 5,952 | 4,209 |
| **真实 2022** | 6,334 | 5,014 | 4,652 |
| **真实 2026** | 4,768 | 4,684 | 6,548 |

> 两个分类器的 accuracy 仅略高于多数类基线（+3%），说明三时段用词模式差异有限，不足以支撑高精度判别。混淆最严重方向：**2022→2021（6,334 例）**，符合预期——相邻时段最难区分。T3(2026) 召回率最高（6,548/16,000=41%），反映 T3 新增条目带来更多可区分特征。
>
> 错分样本举例：真实 2021 的"撞击事件"被预测为 2026，其 Top TF-IDF 词（撞击/地球/小行星/陨石）为跨时段稳定的主题词，缺乏时段区分性信号。

输出文件：`outputs/tables/classification_summary.csv`、`confusion_matrix.csv`、`misclassification_analysis.csv`。

---

## 第 4/5 步：Word2Vec 训练与锚定词 Top-5 近邻对比

### 训练参数（报告 6.2）

| 参数 | 值 | 理由 |
|---|---|---|
| vector_size | 200 | 数十万文档量级上 200 维给语义区分留余量 |
| window | 8 | 比默认 5 略大，捕捉较宽的主题共现 |
| min_count | 10 | 过滤低频噪声；训练后逐段核查锚定词均在词表 |
| sg | 1 | Skip-gram，对低频词和语义邻近结构优于 CBOW |
| epochs | 10 | — |
| seeds | 42, 43 | 双 seed 稳健近邻：两 seed 均进入 Top-5 者为准 |

### 各时段词表大小

| 时段 | 词表大小 | 锚定词核查 |
|---|---|---|
| T1 (2021) | 433,828 | 人工智能/算法/模型/自然语言处理/工作/消费/信任 均在词表 |
| T2 (2022) | 452,992 | 同上，均在词表 |
| T3 (2026) | 526,691 | 同上，"大语言模型"（频次82）新进入词表 |

### Top-5 近邻并排对比（核心结果，报告 6.2）

#### 人工智能（T1∩T3 = 3/5）

| rank | T1 (2021) | T2 (2022) | T3 (2026) | 标注 |
|---|---|---|---|---|
| 1 | 人工智慧 | 人工智慧 | AI | ● |
| 2 | AI | AI | 人工智慧 | ● |
| 3 | 智能 | 智能 | 机器学习 | ▼智能 / ▲机器学习 |
| 4 | 计算机 | 机器学习 | 生成式 | ▼计算机 / ▲生成式 |
| 5 | 机器学习 | 计算机 | 自然语言处理 | ▲自然语言处理 |

> "智能""计算机"在 T1—T2 稳定出现，T3 退出；"生成式""自然语言处理"进入。可解读为人工智能的语义邻域从"计算机科学的泛指能力"向"生成式 AI 与 NLP 应用"收窄与扩张。但需注意 T3 维基百科新增大量 ChatGPT/LLM 条目，近邻变化可能部分来自语料主题构成的改变。

#### 模型（T1∩T3 = 3/5）

| rank | T1 (2021) | T2 (2022) | T3 (2026) | 标注 |
|---|---|---|---|---|
| 1 | 建模 | 建模 | 建模 | ● |
| 2 | 数学模型 | 数学模型 | 数学模型 | ● |
| 3 | 对模型 | 对模型 | 对模型 | ● |
| 4 | 分析模型 | 分析模型 | RLHF | ▼分析模型 / ▲RLHF |
| 5 | 参数估计 | 模式图 | PBPK | ▲PBPK |

> "RLHF"（基于人类反馈的强化学习）在 T3 进入"模型"近邻 Top-4，是大语言模型对齐技术的标志性术语，在 T1/T2 完全不在近邻。这是"模型"语义邻域被大模型话语重塑的探索性信号。

#### 工作（T1∩T3 = 4/5，对照组）

| rank | T1 (2021) | T2 (2022) | T3 (2026) | 标注 |
|---|---|---|---|---|
| 1 | 人员 | 管理工作 | 人员 | ● |
| 2 | 管理工作 | 人员 | 管理工作 | ● |
| 3 | 负责 | 负责人 | 负责 | ● |
| 4 | 工作人员 | 负责 | 负责人 | ▼工作人员 / ▲负责人 |
| 5 | 负责人 | 工作人员 | 基层工作 | ▲基层工作 |

> 社会对照词"工作"的 Top-5 成员在三时段高度稳定（4/5 持续近邻），仅末位词微调。这排除了"全局训练随机性导致所有词都变"的替代解释，支持技术词近邻变化具有内容相关性的判断。

#### 消费（T1∩T3 = 3/5）/ 信任（T1∩T3 = 2/5，对照组）

- **消费**：Top-5 持续近邻"消费者""消费品""消费额"等，结构稳定。
- **信任**：Top-1"信赖"持续，但中低位词在 T2—T3 有较多替换（"态度""敌意""不信任感"进入），变化比"工作""消费"明显，但仍以情感/关系语义为主，未见技术话语侵入。

### T1 与 T3 重合个数（描述性辅助，不构成偏移量）

| 锚定词 | T1∩T3 | 角色 |
|---|---|---|
| 人工智能 | 3/5 | 技术（核心） |
| 算法 | 3/5 | 技术（核心） |
| 模型 | 3/5 | 技术（核心） |
| 自然语言处理 | 4/5 | 技术（核心，低频） |
| 工作 | 4/5 | 社会（对照） |
| 消费 | 3/5 | 社会（对照） |
| 信任 | 2/5 | 社会（对照） |

> 重合个数仅作描述性辅助，不做推断统计。技术词并非一律比社会词变化更大——"自然语言处理"因低频而近邻偏稳，"信任"因情感词跳变而偏动。定性解读以成员变化方向为准。

输出文件：`outputs/tables/top5_neighbors_by_period.csv`（长表）、`top5_comparison.csv` + `top5_comparison.md`（并排对比）。

## 方法学边界

- **不做跨时段向量对齐**，不计算跨时段余弦相似度差值，只比较每个空间内部的近邻排序成员变化
- **不做“语义偏移量”的量化打分**，不做显著性检验
- **不归因于 ChatGPT**，结论措辞落在“探索性观察”与“方法可行性验证”上
- 不同时段独立训练的向量空间存在旋转与尺度差异，余弦相似度的绝对数值跨空间不可比
- Word2Vec 训练随机性通过多次训练稳健近邻（≥3/5 次进入 Top-5）缓解

## 合规声明

本项目未对任何明确禁止爬取的网站发起批量请求，未绕过反爬机制，未采集个人信息。数据来源于 Wikimedia 官方历史 dump（CC BY-SA 4.0），为离线公开存档，不涉及对目标站点的实时批量抓取。数据仅用于本课程学习。
