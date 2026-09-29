#!/usr/bin/env python3
"""第三步：建检索索引（向量 + BM25）。

向量通道
    BAAI/bge-small-zh-v1.5，512 维，纯 CPU。
    权重走国内镜像（HF_ENDPOINT=https://hf-mirror.com），缓存落在项目内 .hf_cache/。
    入库文本 = 《{简称}2026年半年度报告 · {章节}》 + 块正文——公司名和章节必须一起
    进向量，否则「净息差」这种全行业都有的词分不清是谁的数。document 侧不加前缀，
    query 侧才加检索指令前缀（bge 中文系列官方用法，两侧不对称是有意的）。
    2553 块规模很小，直接 numpy 归一化后点积算余弦，不装 faiss。
    编码分块落断点到 index/embeddings.partial.*（含「模型名 + 文本」指纹），
    中途被打断可续跑，换模型/换语料会自动作废断点。

    为什么不是课件推荐的 Qwen/Qwen3-Embedding-0.6B：本机无 GPU，该模型 1024 维、
    权重是 bfloat16，而本机 CPU（Alder Lake）没有 AVX512-BF16，torch 只能走慢速
    兜底，实测 261 token/s、2,553 块要 2~4 小时，全流程时间不够。bge-small-zh-v1.5
    只有 24M 参数、512 维，CPU 上快两个数量级。代价见 README「已知问题」：
    它上下文只有 512 token，本批块平均约 1,250 token，长块会被截断。

BM25 通道
    jieba 分词 + rank_bm25，同一批块，**覆盖全文不截断**——正好补上向量截断丢的尾部。
    自定义词典见 jieba_userdict.txt。

落盘 index/（已 gitignore，可重新生成）
    embeddings.npy    float32，L2 归一化，形状 (N, 512)，行序 = chunks.jsonl 顺序
    meta.jsonl        每行一块：id / 公司 / 代码 / 章节 / 页码 / 正文前 100 字
    bm25_tokens.jsonl 每行一块的分词结果（BM25 的索引载荷，meta 里只有预览字不够用）
    index_info.json   模型名、维度、块数、模板等，供 search.py 校验
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 镜像与缓存目录必须在 huggingface_hub 被 import 之前设好——它在 import 时读环境变量
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import numpy as np  # noqa: E402
import jieba  # noqa: E402
import torch  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

# Windows 控制台默认 GBK，中文/符号直接打印会 UnicodeEncodeError，统一按 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

CHUNKS_PATH = ROOT / "chunks.jsonl"
INDEX_DIR = ROOT / "index"
USERDICT = ROOT / "jieba_userdict.txt"

EMB_PATH = INDEX_DIR / "embeddings.npy"
CKPT_PATH = INDEX_DIR / "embeddings.partial.npy"   # 编码断点，跑完删掉
CKPT_META = INDEX_DIR / "embeddings.partial.json"  # 断点对应的文本指纹，防串味
META_PATH = INDEX_DIR / "meta.jsonl"
TOKENS_PATH = INDEX_DIR / "bm25_tokens.jsonl"
INFO_PATH = INDEX_DIR / "index_info.json"

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DIM = 512
MAX_SEQ = 512           # 该模型的位置编码上限，超过的部分会被截断（见 README 已知问题）
LABEL = "2026年半年度报告"
# 入库文本模板：公司 + 章节 + 正文
DOC_FMT = "《{name}{label} · {section}》"
# query 模板：bge 中文系列官方建议的检索指令前缀，只加在 query 侧
QUERY_TASK = "为这个句子生成表示以用于检索相关文章："
QUERY_FMT = "{task}{query}"

BJT = timezone(timedelta(hours=8))


def log(msg: str = "") -> None:
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """临时文件 + 原子改名，避免写一半被打断留下残缺索引。"""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def init_jieba() -> None:
    jieba.load_userdict(str(USERDICT))
    jieba.initialize()


def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


def doc_text(c: dict) -> str:
    """向量化的入库文本：章节抬头 + 块正文。"""
    return DOC_FMT.format(name=c["name"], label=LABEL, section=c["section"]) + "\n" + c["text"]


def preview(text: str, n: int = 100) -> str:
    return " ".join(text.split())[:n]


def load_chunks(limit: int | None = None) -> list[dict]:
    chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    if limit:
        chunks = chunks[:limit]
    return chunks


def texts_fingerprint(texts: list[str], model_name: str) -> str:
    """入库文本 + 模型名的指纹：语料或向量模型变了都得作废断点，否则续跑的段对不上。"""
    h = hashlib.sha1()
    h.update(model_name.encode("utf-8"))
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:16]


def encode_with_checkpoint(model, texts: list[str], batch_size: int,
                           block: int = 200, model_name: str = "") -> tuple[np.ndarray, float]:
    """分块编码 + 每块落一次断点。全程几个小时，中途被打断不该从头再来。

    返回 (embeddings, 本次编码耗时)。断点里的块数是完整的块，续跑从那里接着编。
    """
    dim = get_dimension(model)
    fp = texts_fingerprint(texts, model_name)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    done, emb = 0, None
    if CKPT_PATH.exists() and CKPT_META.exists():
        try:
            meta = json.loads(CKPT_META.read_text(encoding="utf-8"))
            prev = np.load(CKPT_PATH)
            if meta.get("fingerprint") != fp:
                log("! 断点对应的是另一版 chunks.jsonl，作废重编")
            elif not (prev.ndim == 2 and prev.shape[1] == dim and 0 < prev.shape[0] <= len(texts)):
                log("! 断点形状不对，作废重编")
            else:
                emb = np.zeros((len(texts), dim), dtype=np.float32)
                emb[:prev.shape[0]] = prev
                done = prev.shape[0]
                log(f"      发现断点：已完成 {done}/{len(texts)}，从第 {done + 1} 块续跑")
        except (OSError, ValueError) as exc:
            log(f"! 断点读取失败（{exc}），从头编码")

    if emb is None:
        emb = np.zeros((len(texts), dim), dtype=np.float32)

    t0 = time.time()
    for start in range(done, len(texts), block):
        end = min(start + block, len(texts))
        emb[start:end] = model.encode(
            texts[start:end],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # 先 NPY 再写指纹：万一只有 NPY 落地，下次会因为读不到指纹而重编，不会错用。
        # 传文件对象而不是路径：np.save("x.tmp") 会自作主张写成 "x.tmp.npy"，
        # 后面的 replace 就找不到源文件（WinError 2）。
        tmp = CKPT_PATH.with_name(CKPT_PATH.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, emb[:end])
        tmp.replace(CKPT_PATH)
        atomic_write_text(CKPT_META, json.dumps({"fingerprint": fp, "count": end}))
        dt = time.time() - t0
        n = end - done
        eta = dt / n * (len(texts) - end) / 60
        log(f"      {end:>5}/{len(texts)} 块  {dt / 60:5.1f} 分钟已用  "
            f"{n / dt:4.2f} 块/秒  剩余约 {eta:5.1f} 分钟")
    return emb, time.time() - t0


def get_dimension(model) -> int:
    # ST 6.x 把 get_sentence_embedding_dimension 改名了，两个都兼容一下
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return int(get_dim())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="只索引前 N 块（冒烟测试用）")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="编码批大小；bge-small 小模型，16 足够喂饱 CPU")
    ap.add_argument("--block", type=int, default=200,
                    help="每编多少块落一次断点；断掉重启会从这里接着编")
    ap.add_argument("--model", default=MODEL_NAME)
    args = ap.parse_args()

    chunks = load_chunks(args.limit)
    if not chunks:
        log("chunks.jsonl 是空的，先跑 extract_chunks.py")
        return 1
    log(f"待索引块数 {len(chunks)}")

    INDEX_DIR.mkdir(exist_ok=True)

    # ── BM25：分词 ───────────────────────────────────────────────────
    log("\n[1/3] jieba 分词 …")
    t0 = time.time()
    init_jieba()
    token_lists = [tokenize(doc_text(c)) for c in chunks]
    n_tok = sum(len(t) for t in token_lists)
    log(f"      词典加载 + 分词完毕，共 {n_tok:,} 词，平均 {n_tok / len(chunks):.0f} 词/块"
        f"（{time.time() - t0:.1f}s）")

    # ── 向量：编码 ───────────────────────────────────────────────────
    log(f"\n[2/3] 编码 {len(chunks)} 块 · {args.model}（CPU）…")
    log(f"      HF_ENDPOINT={os.environ['HF_ENDPOINT']}  HF_HOME={os.environ['HF_HOME']}")
    from sentence_transformers import SentenceTransformer  # 延迟 import：慢，且要镜像变量先生效

    t0 = time.time()
    # dtype 显式指定 float32：换模型时踩过坑——Qwen3-Embedding 仓库里存的是 bfloat16，
    # 而这台 CPU 没有 AVX512-BF16，torch 走慢速兜底直接慢 4.4 倍。fp32 对 bge 也是原生精度。
    model = SentenceTransformer(args.model, device="cpu",
                                model_kwargs={"dtype": torch.float32})
    model.max_seq_length = MAX_SEQ   # 512：模型位置编码上限，不改这里就不会截断
    load_s = time.time() - t0
    dim = get_dimension(model)
    dtype = next(model[0].auto_model.parameters()).dtype
    log(f"      模型加载完毕 {load_s:.1f}s，维度 {dim}，权重 dtype={dtype}，"
        f"max_seq_length={model.max_seq_length}，torch 线程 {torch.get_num_threads()}")
    if dim != DIM:
        log(f"! 维度不是预期的 {DIM}，index_info.json 会记实际值")

    texts = [doc_text(c) for c in chunks]
    # 截断统计：bge-small-zh 只吃 512 token，长块尾部进不了向量，这个数要如实报
    n_tok = [len(model.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    n_cut = sum(1 for n in n_tok if n > MAX_SEQ)
    log(f"      token 长度 中位数={sorted(n_tok)[len(n_tok)//2]} 最大={max(n_tok)}；"
        f"超过 {MAX_SEQ} 会被截断的有 {n_cut} 块（{n_cut / len(texts):.1%}）")

    emb, enc_s = encode_with_checkpoint(model, texts, args.batch_size, args.block, args.model)
    log(f"      编码完毕 {enc_s:.1f}s（{len(chunks) / max(enc_s, 1e-9):.2f} 块/秒）")

    emb = np.ascontiguousarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1)
    log(f"      L2 范数 min={norms.min():.4f} max={norms.max():.4f}（应为 1.0）")

    # ── 落盘 ─────────────────────────────────────────────────────────
    log("\n[3/3] 落盘 index/ …")
    buf = INDEX_DIR / "embeddings.tmp.npy"
    with open(buf, "wb") as fh:
        np.save(fh, emb)
    buf.replace(EMB_PATH)
    log(f"      embeddings.npy  {emb.shape} {emb.dtype} "
        f"{EMB_PATH.stat().st_size / 1e6:.1f} MB")

    meta_lines = []
    for c in chunks:
        meta_lines.append(json.dumps({
            "chunk_id": c["chunk_id"],
            "name": c["name"],
            "code": c["code"],
            "period": c["period"],
            "section": c["section"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "has_table": c.get("has_table", False),
            "preview": preview(c["text"], 100),
        }, ensure_ascii=False))
    atomic_write_text(META_PATH, "\n".join(meta_lines) + "\n")
    log(f"      meta.jsonl      {len(meta_lines)} 行  {META_PATH.stat().st_size / 1e6:.1f} MB")

    atomic_write_text(TOKENS_PATH, "\n".join(
        json.dumps({"chunk_id": c["chunk_id"], "tokens": t}, ensure_ascii=False)
        for c, t in zip(chunks, token_lists)) + "\n")
    log(f"      bm25_tokens.jsonl {len(chunks)} 行  {TOKENS_PATH.stat().st_size / 1e6:.1f} MB")

    BM25Okapi(token_lists)  # 建一次确认分词结果可用（search.py 会重建）
    atomic_write_text(INFO_PATH, json.dumps({
        "model": args.model,
        "dim": int(dim),
        "count": len(chunks),
        "built_at": datetime.now(BJT).isoformat(timespec="seconds"),
        "doc_template": DOC_FMT,
        "doc_label": LABEL,
        "query_template": QUERY_FMT,
        "query_task": QUERY_TASK,
        "chunks_path": str(CHUNKS_PATH.relative_to(ROOT)),
        "max_seq_length": int(model.max_seq_length),
        "truncated_chunks": n_cut,
        "token_len_median": sorted(n_tok)[len(n_tok) // 2],
        "texts_fingerprint": texts_fingerprint(texts, args.model),
        "model_load_seconds": round(load_s, 1),
        "encode_seconds": round(enc_s, 1),
        "encode_batch_size": args.batch_size,
        "model_dtype": str(dtype).replace("torch.", ""),
        "torch_threads": torch.get_num_threads(),
    }, ensure_ascii=False, indent=1) + "\n")

    # 索引完整落盘后再删断点，中途失败还能接着编
    for p in (CKPT_PATH, CKPT_META):
        p.unlink(missing_ok=True)
    log("      断点已清理")

    log(f"\n索引目录 {INDEX_DIR}")
    log(f"块数 {len(chunks)} | 向量维度 {dim} | 编码 {enc_s:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
