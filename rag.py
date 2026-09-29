#!/usr/bin/env python3
"""第四步：混合检索（向量 + BM25 + RRF 融合 + 公司过滤）。

为什么这么设计——第三步自检暴露的两个问题：

1. **BM25 单独用时分不清公司。** 问「招商银行 2026 年上半年净息差是多少」，
   BM25 首条给的是平安银行（抬头让公司名 IDF 拉不开差距，加上 BM25Okapi 的
   长度归一化重罚长块，密集提到关键词的短块压过目标公司的分析页）。
   对策：**公司过滤**——问题里出现哪家银行的简称/代码，就先在该公司的子集里
   检索一轮；同时保留一轮全局检索兜底，防止简称识别错了把正确答案挡掉。

2. **两路各有盲区。** 向量路只吃前 512 token（bge-small-zh 的位置编码上限），
   块尾部长表的内容进不了向量；BM25 覆盖全文但排序粗糙。
   对策：**RRF 融合**——两路各取 top-10，按名次的倒数和合并，
   只用名次不用原始分数，省得把余弦相似度和 BM25 分数这两个不同量纲硬凑。

3. **同一个指标的问法和报表用词对不上。** 问「净息差」，招商银行报表里写的是
   「净利息收益率」「净利差」，字面上一个都对不上，BM25 直接落空。
   对策：**口径/同义词扩展**（`SYNONYMS` + `expand_query`）——命中就追加同义词，
   并且把这个对应关系一并告诉生成模型（见 ask.py 的「口径说明」），
   否则模型会以「材料里没这个词」为由拒答。

注意：第 1 条的问题最终是靠**公司过滤**修好的，不是靠 RRF——实测 RRF 单独上
反而比只用向量更差（README「第四步」有消融表）。三者管的是不同的事。

用法见 ask.py（命令行）与 app.py（Streamlit 页面）。
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import numpy as np  # noqa: E402
import jieba  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

INDEX_DIR = ROOT / "index"
CHUNKS_PATH = ROOT / "chunks.jsonl"
USERDICT = ROOT / "jieba_userdict.txt"

RRF_K = 60          # RRF 平滑常数，文献常用 60
K_EACH = 10         # 每路先各取 top-10
TOP_N = 6           # 融合后交给生成模型的块数

# 路名 -> 存名次的字段名（名字不统一，别在拼字符串时拼错）
RANK_KEY = {"vector": "vec_rank", "bm25": "bm_rank"}

# 12 家银行 + 常见简称。简称优先于全称匹配（更长的先匹配），代码也认。
BANKS = [
    ("601398", "工商银行", ["中国工商银行", "工商银行", "工行", "ICBC"]),
    ("601939", "建设银行", ["中国建设银行", "建设银行", "建行", "CCB"]),
    ("601288", "农业银行", ["中国农业银行", "农业银行", "农行", "ABC"]),
    ("601988", "中国银行", ["中国银行", "中行", "BOC"]),
    ("601328", "交通银行", ["交通银行", "交行", "BOCOM"]),
    ("600036", "招商银行", ["招商银行", "招行", "CMB"]),
    ("601166", "兴业银行", ["兴业银行", "兴业", "兴行"]),
    ("600000", "浦发银行", ["上海浦东发展银行", "浦发银行", "浦发", "SPDB"]),
    ("601998", "中信银行", ["中信银行", "中信", "CNCB"]),
    ("600016", "民生银行", ["中国民生银行", "民生银行", "民生"]),
    ("601818", "光大银行", ["中国光大银行", "光大银行", "光大"]),
    ("000001", "平安银行", ["平安银行", "平安"]),
]

CODE2NAME = {code: name for code, name, _ in BANKS}

# 口径/同义词表：同一个指标，问法和财报用词经常对不上（问「净息差」，
# 招商银行报表里写的是「净利息收益率」「净利差」）。命中就把同义词追加到查询里，
# 让 BM25 也能捞到那些块（向量路本来就能跨过字面差异，加了也不亏）。
#
# 只收本项目实测对不上的口径，不做通用词表——通用近义词表容易引入噪声，
# 而且这一层解决的是「财报术语不统一」，不是「语言表达多样」。
SYNONYMS = {
    "净息差": ("净利息收益率", "净利差", "NIM"),
    "净利息收益率": ("净息差", "净利差"),
    "净利差": ("净息差", "净利息收益率"),
    "手续费及佣金净收入": ("手续费及佣金收入", "中间业务收入"),
    "归母净利润": ("归属于本行股东的净利润", "归属于母公司股东的净利润"),
    "利息净收入": ("净利息收入",),
    "不良贷款率": ("不良率", "不良贷款比率"),
    "拨备覆盖率": ("贷款拨备覆盖率", "拨备率"),
    "资本充足率": ("资本充足水平",),
    "成本收入比": ("成本收入比率",),
    "营业收入": ("营收",),
}


def expand_query(question: str, enabled: bool = True) -> tuple[str, list[str]]:
    """把问题里命中的指标名扩上同义词，返回 (用于检索的查询, 新增的词)。

    只追加不替换：原词是用户真实问法，往往是块里出现频率最高的那个，
    换掉它反而会丢分。新增的词只是补一路召回。
    """
    if not enabled:
        return question, []
    added: list[str] = []
    for term, syns in SYNONYMS.items():
        if term not in question:
            continue
        for s in syns:
            if s not in question and s not in added:
                added.append(s)
    if not added:
        return question, []
    return f"{question}（{'、'.join(added)}）", added


def load_env(path: Path = ROOT / ".env") -> None:
    """读 .env（已在 .gitignore）。已存在的环境变量优先，不覆盖。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def detect_company(question: str) -> tuple[str | None, str | None]:
    """问题里出现哪家银行的代码或简称就返回 (代码, 简称)，否则 (None, None)。

    先认代码（600036 这种不会误伤），再按别名长度从长到短匹配，
    避免「中国银行」被短别名先命中成别的行。
    """
    for code, name, _ in BANKS:
        if code in question:
            return code, name
    hits = []
    for code, name, aliases in BANKS:
        for alias in aliases:
            if alias in question:
                hits.append((len(alias), code, name))
    if not hits:
        return None, None
    hits.sort(reverse=True)          # 最长的别名优先
    return hits[0][1], hits[0][2]


class Retriever:
    """索引加载一次，之后每次提问只算分数。"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.info = self._read_json(INDEX_DIR / "index_info.json")
        self.meta = [json.loads(l) for l in
                     (INDEX_DIR / "meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        self.tokens = [json.loads(l)["tokens"] for l in
                       (INDEX_DIR / "bm25_tokens.jsonl").read_text(encoding="utf-8").splitlines()
                       if l.strip()]
        self.emb = np.load(INDEX_DIR / "embeddings.npy")
        # meta 里只有正文前 100 字，送模型的必须是全文，所以回读 chunks.jsonl
        self.texts = [json.loads(l)["text"] for l in
                      CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(self.meta) == len(self.tokens) == len(self.texts) == self.emb.shape[0], \
            "索引与 chunks.jsonl 行数不一致，重建索引"
        self.bm25 = BM25Okapi(self.tokens)
        jieba.load_userdict(str(USERDICT))
        jieba.initialize()
        self._model = None

    @staticmethod
    def _read_json(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8"))

    @property
    def model(self):
        """延迟加载：只用 BM25 模式时不必付这个代价。"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.info["model"], device="cpu",
                                              model_kwargs={"dtype": __import__("torch").float32})
        return self._model

    def _vector_scores(self, question: str) -> np.ndarray:
        q_text = self.info["query_template"].format(task=self.info["query_task"], query=question)
        qv = self.model.encode([q_text], normalize_embeddings=True,
                               convert_to_numpy=True)[0].astype(np.float32)
        return self.emb @ qv                       # 都已 L2 归一化，点积即余弦

    def _bm25_scores(self, question: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores([t for t in jieba.lcut(question) if t.strip()]))

    def _rank_in_pool(self, pool: list[int], vec: np.ndarray, bm: np.ndarray,
                      mode: str, k: int) -> list[dict]:
        """在给定的候选集里排一次名，返回带两路名次/分数与 RRF 分的条目。"""
        ranked = []
        if mode in ("vector", "hybrid"):
            order = sorted(pool, key=lambda i: (-vec[i], i))[:K_EACH if mode == "hybrid" else k]
            for rank, i in enumerate(order, 1):
                ranked.append({"idx": i, "route": "vector", "rank": rank, "score": float(vec[i])})
        if mode in ("bm25", "hybrid"):
            order = sorted(pool, key=lambda i: (-bm[i], i))[:K_EACH if mode == "hybrid" else k]
            for rank, i in enumerate(order, 1):
                ranked.append({"idx": i, "route": "bm25", "rank": rank, "score": float(bm[i])})

        merged: dict[int, dict] = {}
        for r in ranked:
            hit = merged.setdefault(r["idx"], {
                "idx": r["idx"], "vec_score": float(vec[r["idx"]]),
                "bm_score": float(bm[r["idx"]]), "vec_rank": None, "bm_rank": None, "rrf": 0.0,
            })
            hit[RANK_KEY[r["route"]]] = r["rank"]
            # RRF：只用名次，不用原始分数——余弦相似度和 BM25 分数不同量纲，硬加权没依据
            hit["rrf"] += 1.0 / (RRF_K + r["rank"])
        out = sorted(merged.values(),
                     key=lambda h: (-h["rrf"], -(h["vec_score"] if mode != "bm25" else h["bm_score"])))
        return out[:k]

    def retrieve(self, question: str, top: int = TOP_N, mode: str = "hybrid",
                 company_filter: bool = True, expand: bool = True) -> dict:
        """一轮公司限定（若识别到公司）+ 一轮全局兜底，合并后取 top。

        限定轮的结果排在前，全局轮补足——既压住 BM25 的串公司，又不会因为
        简称识别错了就完全拿不到证据。
        """
        t0 = time.time()
        # 公司识别用原问题：同义词表里没有银行名，扩了也是白扩
        code, name = detect_company(question)
        q_search, added = expand_query(question, expand)
        vec, bm = self._vector_scores(q_search), self._bm25_scores(q_search)
        t_score = time.time() - t0

        all_idx = list(range(len(self.meta)))
        rounds: dict[str, list[dict]] = {"global": self._rank_in_pool(all_idx, vec, bm, mode, top)}

        if company_filter and code:
            pool = [i for i, m in enumerate(self.meta) if m["code"] == code]
            rounds["company"] = self._rank_in_pool(pool, vec, bm, mode, top)

        # 合并：公司轮优先，全局轮补足且去重
        order = ["company", "global"] if "company" in rounds else ["global"]
        picks: list[dict] = []
        seen: set[int] = set()
        for rname in order:
            for h in rounds[rname]:
                if h["idx"] in seen:
                    for p in picks:
                        if p["idx"] == h["idx"]:
                            p["rounds"].append(rname)
                    continue
                seen.add(h["idx"])
                picks.append({**h, "rounds": [rname], "from": rname})
        picks = picks[:top]

        for h in picks:
            m = self.meta[h["idx"]]
            h.update({
                "chunk_id": m["chunk_id"], "code": m["code"], "name": m["name"],
                "section": m["section"], "page_start": m["page_start"], "page_end": m["page_end"],
                "has_table": m.get("has_table", False), "text": self.texts[h["idx"]],
                "preview": m["preview"],
            })
        return {
            "question": question, "mode": mode, "top": top,
            "expansion": added,                    # 命中并追加的同义词，供提示词说明口径
            "query_used": q_search if added else question,
            "company": {"code": code, "name": name} if code else None,
            "hits": picks,
            "n_candidates": len(rounds.get("company", [])) if "company" in rounds else len(all_idx),
            "timing": {"score": t_score, "total": time.time() - t0},
        }

    @staticmethod
    def cite(hit: dict) -> str:
        pg = f"p{hit['page_start']}" if hit["page_start"] == hit["page_end"] \
            else f"p{hit['page_start']}-{hit['page_end']}"
        return f"{hit['name']} · {hit['section']} · {pg}"
