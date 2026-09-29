#!/usr/bin/env python3
"""第四步：混合检索 + DeepSeek 生成答案（命令行）。

    .venv/Scripts/python.exe ask.py "招商银行 2026 年上半年净息差是多少？"
    .venv/Scripts/python.exe ask.py --mode bm25 "工商银行上半年手续费及佣金净收入是多少？"
    .venv/Scripts/python.exe ask.py --retrieval-only "..."      # 不调模型，只看召回
    .venv/Scripts/python.exe ask.py --top 6 --show-text "..."

key 从项目根目录的 .env 读（DEEPSEEK_API_KEY），也可直接用同名环境变量覆盖。
检索与生成的核心逻辑都在这里，app.py 直接复用 answer()。
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag import TOP_N, Retriever, load_env  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

DEFAULT_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

SYSTEM_PROMPT = """你是银行财报问答助手。你只能依据用户给出的【检索材料】回答，\
不得使用材料以外的知识，不得编造数字。

要求：
1. 先用一两句话直接给出结论；所有数字必须与材料完全一致，不得换算或约等于。
2. 每条结论后面用括号标注出处，格式为（公司 · 章节 · 页码），出处照抄材料里给出的那一行。
3. 材料不足以支撑结论时，明确回答「检索到的材料无法回答」，并说明还缺什么。不要猜测、不要用常识补全。
4. 材料若来自不同公司，要分别说明是谁的数据，不要把不同公司的数字混在一起比较。
5. 用简体中文回答，不要输出与问题无关的材料内容。"""


class NoApiKey(RuntimeError):
    """没配 key。app.py 靠这个类型给出友好提示，而不是抛栈。"""


def build_context(hits: list[dict]) -> str:
    lines = []
    for n, h in enumerate(hits, 1):
        pg = f"p{h['page_start']}" if h["page_start"] == h["page_end"] \
            else f"p{h['page_start']}-{h['page_end']}"
        lines.append(f"[{n}] 出处：{h['name']} · {h['section']} · {pg}"
                     f"（块 {h['chunk_id']}，{'表格块' if h['has_table'] else '正文块'}）\n{h['text']}")
    return "\n\n".join(lines)


def build_messages(question: str, hits: list[dict],
                   expansion: list[str] | None = None) -> list[dict]:
    notes = ""
    if expansion:
        # 这一步很关键：同义词是我们检索时按自己的口径表加进去的，
        # 不告诉模型，它只会看到「问题问净息差、材料只提净利息收益率」而拒答。
        # 把对应关系作为材料的一部分给它，而不是让它自己拿常识去等同——
        # 否则就违反了 system prompt 第 3 条（不得使用材料以外的知识）。
        notes = (f"【口径说明】检索时已按本项目的指标口径对照表，把问题中的指标"
                 f"一并按「{'、'.join(expansion)}」检索。若材料中出现这些表述，"
                 f"即视为同一指标的披露口径，**可以直接用其数值回答**，"
                 f"出处照抄材料里的用词。\n\n")
    user = (f"【问题】{question}\n\n"
            f"{notes}"
            f"【检索材料】（共 {len(hits)} 段，来自 12 家 A 股银行 2026 年半年度报告）\n\n"
            f"{build_context(hits)}")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def generate(question: str, hits: list[dict], timeout: int = 120,
             expansion: list[str] | None = None) -> dict:
    """调 DeepSeek 生成答案。没有 key 直接抛 NoApiKey。"""
    load_env()
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise NoApiKey("未配置 DEEPSEEK_API_KEY：请把 key 写进项目根目录的 .env"
                       "（可参考 .env.example），或设为同名环境变量")
    base = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE).rstrip("/")
    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
    payload = json.dumps({
        "model": model,
        "messages": build_messages(question, hits, expansion),
        "temperature": 0.0,          # 财报问答要的是复述与标注，不是创作
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"DeepSeek 返回 HTTP {exc.code}：{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"连不上 DeepSeek（{exc.reason}）") from exc
    return {
        "answer": data["choices"][0]["message"]["content"].strip(),
        "model": data.get("model", model),
        "usage": data.get("usage", {}),
        "seconds": time.time() - t0,
    }


def answer(question: str, retriever: Retriever, mode: str = "hybrid",
           top: int = TOP_N, company_filter: bool = True,
           retrieval_only: bool = False, expand: bool = True) -> dict:
    """检索 + 生成。检索结果一定返回，生成失败（如没 key）不吞掉检索结果。"""
    r = retriever.retrieve(question, top=top, mode=mode,
                           company_filter=company_filter, expand=expand)
    if retrieval_only:
        return r
    try:
        r["generation"] = generate(question, r["hits"], expansion=r.get("expansion"))
    except (NoApiKey, RuntimeError) as exc:
        r["generation_error"] = str(exc)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="*", help="一个或多个问题")
    ap.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--no-company-filter", action="store_true", help="关掉公司限定轮")
    ap.add_argument("--no-expand", action="store_true",
                    help="关掉口径/同义词扩展，用于对比实验")
    ap.add_argument("--retrieval-only", action="store_true", help="只召回，不调生成模型")
    ap.add_argument("--show-text", action="store_true", help="打印召回块的正文")
    args = ap.parse_args()

    questions = args.questions or ["招商银行 2026 年上半年净息差是多少？"]
    r = Retriever()
    print(f"索引：{r.info['count']} 块 · {r.info['model']} · {r.info['dim']} 维 · "
          f"模式 {args.mode}\n")

    for q in questions:
        out = answer(q, r, mode=args.mode, top=args.top,
                     company_filter=not args.no_company_filter,
                     retrieval_only=args.retrieval_only,
                     expand=not args.no_expand)
        print("═" * 78)
        print(f"问题：{q}")
        if out["expansion"]:
            print(f"口径扩展：＋{('、'.join(out['expansion']))}")
            print(f"         实际检索词：{out['query_used']}")
        else:
            print("口径扩展：未命中同义词表"
                  f"{'（已关闭）' if args.no_expand else ''}")
        if out["company"]:
            print(f"公司过滤：识别到 {out['company']['name']}（{out['company']['code']}），"
                  f"限定轮候选 {out['n_candidates']} 块 + 全局轮兜底")
        else:
            print("公司过滤：未识别到具体银行，只跑全局轮")
        print(f"检索耗时 {out['timing']['score'] * 1000:.0f} ms\n")

        print(f"召回 {len(out['hits'])} 块：")
        for n, h in enumerate(out["hits"], 1):
            vr = f"{h['vec_rank']}" if h["vec_rank"] else "—"
            br = f"{h['bm_rank']}" if h["bm_rank"] else "—"
            print(f" {n}. {h['name']} · {h['section']} · "
                  f"p{h['page_start']}-{h['page_end']} · {h['chunk_id']}"
                  f"   [向量#{vr} {h['vec_score']:.3f} | BM25#{br} {h['bm_score']:.2f}"
                  f" | RRF {h['rrf']:.5f} | {'+'.join(h['rounds'])}]")
            if args.show_text:
                body = h["text"] if len(h["text"]) < 700 else h["text"][:700] + " …"
                print("    " + body.replace("\n", "\n    "))

        if args.retrieval_only:
            print()
            continue
        if "generation_error" in out:
            print(f"\n生成失败：{out['generation_error']}\n")
            continue
        g = out["generation"]
        u = g["usage"]
        print(f"\n【答案】{g['model']} · {g['seconds']:.1f}s · "
              f"in {u.get('prompt_tokens', '?')} / out {u.get('completion_tokens', '?')} tokens")
        print(g["answer"] + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
