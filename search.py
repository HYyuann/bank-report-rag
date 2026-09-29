#!/usr/bin/env python3
"""检索自检：输入一个问题，分别输出【向量 top-5】和【BM25 top-5】，都带出处。

用法
    .venv/Scripts/python.exe search.py "招商银行 2026 年上半年净息差是多少？"
    .venv/Scripts/python.exe search.py --top 3 "问题一" "问题二"
    .venv/Scripts/python.exe search.py          # 不带问题则进交互模式

两路共用同一批块（chunks.jsonl 的行序），出处 = 公司 / 章节 / 页码。
向量查的是「章节抬头 + 正文」，抬头也一并打印出来，方便核对入库文本。
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import numpy as np  # noqa: E402
import jieba  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

# Windows 控制台默认 GBK，打印 ▸ 之类字符会 UnicodeEncodeError，统一按 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

INDEX_DIR = ROOT / "index"
EMB_PATH = INDEX_DIR / "embeddings.npy"
META_PATH = INDEX_DIR / "meta.jsonl"
TOKENS_PATH = INDEX_DIR / "bm25_tokens.jsonl"
INFO_PATH = INDEX_DIR / "index_info.json"
USERDICT = ROOT / "jieba_userdict.txt"

RULE = "═" * 78


def load_index() -> tuple:
    for p in (EMB_PATH, META_PATH, TOKENS_PATH, INFO_PATH):
        if not p.exists():
            sys.exit(f"缺少 {p.name}，先跑 build_index.py")
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    meta = [json.loads(l) for l in META_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    toks = [json.loads(l)["tokens"] for l in
            TOKENS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    emb = np.load(EMB_PATH)
    assert emb.shape[0] == len(meta) == len(toks), "索引三个文件行数不一致，重建索引"
    return info, meta, toks, emb


def load_model(info: dict):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(info["model"], device="cpu")


def vector_topk(model, info: dict, emb: np.ndarray, q: str, k: int) -> list[tuple[int, float]]:
    # query 侧加 Instruct 前缀；document 侧入库时没加，两侧不对称是 Qwen3-Embedding 的设计
    q_text = info["query_template"].format(task=info["query_task"], query=q)
    qv = model.encode([q_text], normalize_embeddings=True, convert_to_numpy=True)[0]
    sims = emb @ qv.astype(np.float32)          # 都已 L2 归一化，点积即余弦
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]


def bm25_topk(bm25: BM25Okapi, q: str, k: int) -> list[tuple[int, float]]:
    scores = bm25.get_scores([t for t in jieba.lcut(q) if t.strip()])
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]


def render(hit_list: list[tuple[int, float]], meta: list[dict], label: str) -> None:
    print(f"【{label}】")
    for rank, (i, score) in enumerate(hit_list, 1):
        m = meta[i]
        pg = f"p{m['page_start']}" if m["page_start"] == m["page_end"] \
            else f"p{m['page_start']}-{m['page_end']}"
        print(f" {rank}. {score:6.3f}  ▸ {m['name']}({m['code']}) · {m['section']} · {pg}"
              f" · {m['chunk_id']}")
        print(f"       {m['preview']}")
    print()


def answer(question: str, model, info, meta, toks, emb, bm25, k: int) -> None:
    print(f"问题：{question}")
    print(RULE)
    render(vector_topk(model, info, emb, question, k), meta, f"向量 top-{k}　{info['model']}")
    render(bm25_topk(bm25, question, k), meta, f"BM25 top-{k}　jieba + BM25Okapi")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="*", help="一个或多个问题；留空进交互模式")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    info, meta, toks, emb = load_index()
    jieba.load_userdict(str(USERDICT))
    jieba.initialize()
    print(f"索引：{info['count']} 块 · {info['model']} · {info['dim']} 维 · "
          f"建于 {info['built_at']}  (HF_ENDPOINT={os.environ['HF_ENDPOINT']})\n")

    bm25 = BM25Okapi(toks)
    model = load_model(info)

    if args.questions:
        for q in args.questions:
            answer(q, model, info, meta, toks, emb, bm25, args.top)
        return 0

    while True:
        try:
            q = input("问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q in {"exit", "quit", "q"}:
            return 0
        answer(q, model, info, meta, toks, emb, bm25, args.top)


if __name__ == "__main__":
    sys.exit(main())
