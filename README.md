# bank-rag · A 股银行财报检索问答

第四周作业（方向 A）。语料为 12 家 A 股上市银行的 2026 年半年度报告。

## 进度

- [x] 第一步：从巨潮资讯网批量下载 12 家银行 2026H1 半年报 PDF
- [x] 第二步：PDF 文本提取 + 表格还原 + 切分落盘（含繁体统一转简体）
- [x] 第三步：建索引（Qwen3-Embedding 向量 + BM25 + search.py 自检）
- [ ] 第四步：检索问答

## 环境与运行

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

下载财报（默认只跑试点两家）：

```bash
.venv/Scripts/python.exe download_reports.py            # 招商银行 + 平安银行
.venv/Scripts/python.exe download_reports.py --all      # 全部 12 家
.venv/Scripts/python.exe download_reports.py --codes 600036,601398
.venv/Scripts/python.exe download_reports.py --all --force   # 忽略已存在文件重下
```

产物：`files/{代码}_{简称}_2026H1.pdf`，元数据 `manifest.json`。
脚本输出已固定为 UTF-8，Windows 下无需手动 `chcp`。

提取文本 + 表格 + 切块（逐家落盘，可重复跑，已完成的自动跳过）：

```bash
.venv/Scripts/python.exe extract_chunks.py              # 全量，12 家
.venv/Scripts/python.exe extract_chunks.py --codes 600036   # 只跑一家
.venv/Scripts/python.exe extract_chunks.py --force      # 忽略 extract/ 重编
.venv/Scripts/python.exe extract_chunks.py --merge-only # 只合并 + 出报告
.venv/Scripts/python.exe extract_chunks.py --sample 5   # 抽样自检
```

产物：`extract/{代码}.json`（逐家中间产物）、`chunks.jsonl`（2,553 块）、
`extraction_report.md`（质检报告，**入库**）。

建索引（向量 + BM25）：

```bash
.venv/Scripts/python.exe build_index.py                 # 2553 块约 4 分钟（纯 CPU）
.venv/Scripts/python.exe build_index.py --limit 20      # 冒烟测试
.venv/Scripts/python.exe build_index.py --block 100     # 调小断点间隔
.venv/Scripts/python.exe search.py "招商银行 2026 年上半年净息差是多少？"
.venv/Scripts/python.exe search.py --top 3 "问题一" "问题二"
.venv/Scripts/python.exe search.py                      # 交互模式
```

产物：`index/embeddings.npy`（float32，2553×512）、`index/meta.jsonl`、
`index/bm25_tokens.jsonl`、`index/index_info.json`。
模型权重走 `HF_ENDPOINT=https://hf-mirror.com`，缓存在项目内 `.hf_cache/`。

## 数据来源

巨潮资讯网（cninfo.com.cn）公开公告查询接口。摸索出的几个关键点，避免以后重复踩坑：

1. `stock` 参数**必须**写成 `"代码,orgId"`，只传代码会返回 0 条。orgId 通过
   `POST /new/information/topSearch/detailOfQuery`（表单参数 `keyWord=<代码>`）获取，
   取 `keyBoardList[]` 中 `category == "A股"` 的那一项的 `orgId`。
2. `column` 必须与上市地一致：沪市 `sse`，深市 `szse`。
3. `category=category_bndbg_szsh` 即「半年度报告」，**沪深通用**，一个值即可。
4. 每家在 2026 全年区间内正好返回 2 条：**正文 + 摘要**。各家标题格式不统一
   （「招商银行股份有限公司2026年半年度报告」vs「工商银行2026半年度报告」，
   有的不带「年/度」），因此过滤规则为「必须含『半年度报告』或『半年报』」
   并排除 `摘要/英文/更正/补充/提示性/公告/修订` 等词，再按日期 + 体积取最大。
5. PDF 实际地址为 `http://static.cninfo.com.cn/` + 公告的 `adjunctUrl`。

12 家银行的 orgId 已实测取得，作为接口临时故障时的兜底值内置在脚本的
`ORG_ID_FALLBACK` 中。

## 已知问题

### 招商银行半年报正文为繁体

招商银行 A+H 两地上市，其在巨潮 A 股「半年度报告」类目下披露的正文
（`600036_招商银行_2026H1.pdf`）**全篇为繁体中文**——封面写作
「2026 年半年度報告 A 股股票代碼：600036」。经核查该股 2026 半年报类目下
只有这一份正文 + 一份摘要，没有对应的简体版本，因此**保留使用**该文件。

**处理方式**：在第二步文本提取环节统一做繁 → 简转换，使用 `zhconv`
（已装进项目 `.venv`）。

```python
import zhconv
text = zhconv.convert(text, "zh-cn")   # 淨利息收入 -> 净利息收入
```

**原因**：语料中混入繁体将同时拖累 embedding 与 BM25 的中文匹配效果——
繁体字形与简体字形的 token 不一致，同一概念无法命中；BM25 尤其严重，
「淨利息收入」查不到「净利息收入」。统一转简体后语料才是同分布的。

### 向量模型没用课件推荐的 Qwen3-Embedding-0.6B

**课件推荐 `Qwen/Qwen3-Embedding-0.6B`（1024 维）**，本项目实际用的是
**`BAAI/bge-small-zh-v1.5`（512 维）**，原因是本机无 GPU、Qwen3 全量编码需 2~4 小时，
要先把全流程跑通。

排查过程（这一步值得记，因为它不是简单的「模型大所以慢」）：

1. 第一版直接跑，实测 **22 秒/块，2553 块要 11 小时**。
2. 先怀疑注意力实现，eager 换 sdpa 只快 1.4 倍（23 → 16 秒/块），不是主因。
3. 打印权重 dtype 才发现根因：**Qwen3-Embedding 仓库里存的是 bfloat16**，
   `SentenceTransformer` 默认照单全收，而本机 CPU 是 Alder Lake、
   **没有 AVX512-BF16**，torch 只能走慢速兜底——吞吐被钉在 59 token/s。
4. 显式转 float32 后 **261 token/s**，快 4.4 倍，但全量仍需 2~4 小时。

```python
SentenceTransformer(name, device="cpu", model_kwargs={"dtype": torch.float32})
```

顺带扫出来的配置（8 块样本，取两遍最快值；这台机器上）：

| 配置 | 秒/块 | 2553 块 |
|---|---|---|
| bge-small-zh fp32 batch=16 | **0.08** | **约 4 分钟** |
| Qwen3 fp32 batch=4 | 3.2 | 2.3 小时 |
| Qwen3 bf16 batch=4 | 15.8 | 11.2 小时 |
| Qwen3 fp32 batch=8 | 6.9 | 4.9 小时 |
| Qwen3 fp32 + int8 动态量化 batch=16 | 7.3 | 5.2 小时 |

即：**Qwen3 在这台机器上 batch 要小（4），int8 动态量化反而更慢**。
换 bge-small-zh 后快约 40 倍，代价见下一条。

如果之后要换回课件指定的 1024 维 Qwen3，把 `build_index.py` 里的
`MODEL_NAME` / `DIM` 改回去即可——断点指纹里带了模型名，换模型会自动作废旧断点，
不会拿 512 维的旧向量拼新索引。

### bge-small-zh-v1.5 上下文只有 512 token，长块会被截断

`bge-small-zh-v1.5` 的位置编码上限是 512 token，而本批块平均约 1,066 token
（中位数，最长 2,203）——**2,553 块里有 2,548 块（99.8%）会被截断**，
平均只有前 48% 进入向量。

**这是为跑通全流程接受的已知取舍**，应对方式是两路互补：

- **向量路**：入库文本带了 `《{简称}2026年半年度报告 · {章节}》` 抬头，
  被截断时保留的是「章节开头」——恰好是结论、总额、小计所在的位置。
- **BM25 路**：jieba 分词 + `BM25Okapi` 覆盖**全文不截断**，块尾部的内容
  （尤其是长表格的后半段）只在这条路上能被召回。

`index_info.json` 里记了 `max_seq_length` / `truncated_chunks` / `token_len_median`，
每次建索引都会重新统计并打印，换模型或改切块参数后能立刻看出截断比例的变化。

### BM25 单独用时分不清是哪家银行

三个自检问题里，**BM25 的首条命中三题全部是错公司**。例如问「招商银行 2026 年上半年
净息差是多少」，BM25 首条给的是**平安银行** p47（「本集团净息差 1.80%」）。

不是分词问题，是排序问题：

1. 每家 200 多块的入库文本都带抬头 `《{简称}2026年半年度报告 · {章节}》`，
   于是「招商银行」在 2,553 块里出现在约 240 块中——**IDF 拉不开差距**，
   它几乎不能把候选集收窄到目标公司。
2. `BM25Okapi` 默认 `b=0.75`，长度归一化很重。本批块平均 666 个 jieba 词，
   而「净息差」在短块（如平安 p47 那样的一小节）里词频密度高，
   长块的得分被长度压下去，于是**密集提到关键词的短块压过了目标公司的分析页**。

**这是第四步要做混合检索 + 公司元数据过滤的直接理由**，不是 bug，要在评测里量化。
向量路在同一批问题上表现相反：三题里两题首条就命中正确公司和指标
（招商银行净息差 → 管讨 p15-16；工商银行手续费及佣金 → 财务报告 p23-25，
原文即「上半年，手续费及佣金收入 765.91 亿元，同比增加 24.29 亿元，增长 3.3%」）。

## 目录结构

```
bank-rag/
├── download_reports.py    # 第一步：批量下载脚本
├── extract_chunks.py      # 第二步：文本/表格提取 + 章节识别 + 切块
├── build_index.py         # 第三步：建向量 + BM25 索引
├── search.py              # 第三步：检索自检（两路 top-k 并排）
├── jieba_userdict.txt     # BM25 自定义词典（银行名 + 财务术语）
├── manifest.json          # 下载清单（代码/标题/日期/地址/文件名/大小/页数）
├── extraction_report.md   # 第二步质检报告（入库，结论页素材）
├── requirements.txt
├── files/                 # 财报 PDF（已 gitignore，可由脚本重新下载）
├── extract/               # 逐家提取中间产物（已 gitignore）
├── chunks.jsonl           # 2,553 块（已 gitignore，可重新生成）
├── index/                 # 向量 + BM25 索引（已 gitignore，可重新生成）
├── .hf_cache/             # Qwen3-Embedding 权重缓存（已 gitignore，1.2GB）
└── .venv/                 # 虚拟环境（已 gitignore）
```

## 未纳入版本控制

`.env`、`files/`、`.venv/`、`extract/`、`chunks.jsonl`、`index/`、`.hf_cache/`
见 `.gitignore`。不入库的都是体积大且可由脚本完整复现的：`files/` 靠
`download_reports.py --all`，`extract/` 与 `chunks.jsonl` 靠 `extract_chunks.py`，
`index/` 靠 `build_index.py`。入库的是**无法自动重算的结论与产物清单**——
`manifest.json`（便于核对语料一致性）和 `extraction_report.md`（质检结论）。
