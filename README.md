# bank-rag · A 股银行财报检索问答

第四周作业（方向 A）。语料为 12 家 A 股上市银行的 2026 年半年度报告。

## 进度

- [x] 第一步：从巨潮资讯网批量下载 12 家银行 2026H1 半年报 PDF
- [ ] 第二步：PDF 文本提取 + 繁体统一转简体
- [ ] 第三步：切分、索引（embedding + BM25）
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

## 目录结构

```
bank-rag/
├── download_reports.py    # 第一步：批量下载脚本
├── manifest.json          # 下载清单（代码/标题/日期/地址/文件名/大小/页数）
├── requirements.txt
├── files/                 # 财报 PDF（已 gitignore，可由脚本重新下载）
└── .venv/                 # 虚拟环境（已 gitignore）
```

## 未纳入版本控制

`.env`、`files/`、`.venv/` 见 `.gitignore`。`files/` 不入库是因为体积大且可由
`download_reports.py --all` 完整复现；`manifest.json` 入库，便于核对语料一致性。
