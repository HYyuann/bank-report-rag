#!/usr/bin/env python3
"""第四步：Streamlit 问答页面。

    .venv/Scripts/python.exe -m streamlit run app.py

页面三块：生成的答案（带出处）、召回的块（可展开看原文与两路分数）、
左上角一个检索模式开关（混合 / 只用向量 / 只用 BM25）用来做对比实验。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ask import answer  # noqa: E402
from rag import TOP_N, Retriever  # noqa: E402

st.set_page_config(page_title="银行财报问答", page_icon="🏦", layout="wide")

MODES = {
    "混合（向量 + BM25 + RRF）": "hybrid",
    "只用向量": "vector",
    "只用 BM25": "bm25",
}


@st.cache_resource(show_spinner="正在加载索引与向量模型（首次约 40 秒）…")
def get_retriever() -> Retriever:
    r = Retriever(verbose=False)
    # 提前触发权重加载，否则这笔开销会算在用户第一次提问的等待里。
    # 注意必须是赋值：裸写 `r.model` 会被 Streamlit 的 magic 改写成 st.write(r.model)，
    # 页面顶部就会多出一坨 SentenceTransformer(...) 的 repr。
    _ = r.model
    return r


def score_caption(h: dict) -> str:
    vr = f"#{h['vec_rank']} {h['vec_score']:.3f}" if h["vec_rank"] else "未进 top-10"
    br = f"#{h['bm_rank']} {h['bm_score']:.2f}" if h["bm_rank"] else "未进 top-10"
    origin = "＋".join("公司限定轮" if r == "company" else "全局兜底轮" for r in h["rounds"])
    return f"向量 {vr}　|　BM25 {br}　|　RRF {h['rrf']:.5f}　|　来自 {origin}"


def render_hits(hits: list[dict]) -> None:
    for n, h in enumerate(hits, 1):
        pg = f"p{h['page_start']}" if h["page_start"] == h["page_end"] \
            else f"p{h['page_start']}-{h['page_end']}"
        title = (f"{n}. {h['name']}（{h['code']}） · {h['section']} · {pg}"
                 f"{' · 表格块' if h['has_table'] else ''}")
        with st.expander(title, expanded=(n == 1)):
            st.caption(score_caption(h) + f"　|　块 {h['chunk_id']}")
            st.code(h["text"], language=None)   # 等宽保留表格的竖线对齐


def render_result(out: dict, elapsed: float) -> None:
    if out["company"]:
        st.caption(f"识别到公司 **{out['company']['name']}**（{out['company']['code']}），"
                   f"已做一轮限定检索（候选 {out['n_candidates']} 块）＋ 一轮全局兜底")
    else:
        st.caption("未识别到具体银行，只跑全局检索")
    if out.get("expansion"):
        st.caption(f"口径扩展：＋{'、'.join(out['expansion'])}"
                   f"　→　实际检索词「{out['query_used']}」")

    st.markdown("### 答案")
    if "generation_error" in out:
        st.warning(f"**没有生成答案**：{out['generation_error']}\n\n"
                   f"下面的召回结果仍然有效，可用于人工核对。")
    else:
        g = out["generation"]
        u = g["usage"]
        st.markdown(out["generation"]["answer"])
        st.caption(f"模型 {g['model']} · 生成 {g['seconds']:.1f}s · "
                   f"输入 {u.get('prompt_tokens', '?')} / 输出 "
                   f"{u.get('completion_tokens', '?')} tokens · "
                   f"检索 {out['timing']['score'] * 1000:.0f} ms · 页面总耗时 {elapsed:.1f}s")

    st.markdown(f"### 召回的 {len(out['hits'])} 个块")
    render_hits(out["hits"])


def main() -> None:
    r = get_retriever()

    with st.sidebar:
        st.header("检索设置")
        mode_label = st.radio("检索模式", list(MODES), index=0,
                              help="切换成单路，用来对比混合检索是否真的更好")
        mode = MODES[mode_label]
        top = st.slider("交给生成模型的块数", 3, 10, TOP_N)
        use_company = st.checkbox("启用公司限定检索", value=True,
                                  help="问题里出现银行简称/代码时，先在该公司子集里检索一轮")
        use_expand = st.checkbox("启用口径/同义词扩展", value=True,
                                 help="把「净息差」这类问法扩成报表用词（净利息收益率/净利差），"
                                      "并把对应关系一并告诉生成模型")
        st.divider()
        st.caption(f"索引 {r.info['count']} 块 · {r.info['model']} · {r.info['dim']} 维")
        st.caption(f"每路先各取 top-10，RRF（k=60）融合后取前 {top}")
        if r.info.get("truncated_chunks"):
            st.caption(f"⚠️ 向量路截断 {r.info['truncated_chunks']}/{r.info['count']} 块"
                       f"（模型上限 {r.info['max_seq_length']} token），块尾部靠 BM25 兜")
        st.divider()
        if not (ROOT / ".env").exists():
            st.caption("未发现 .env，请在项目根目录建 .env 并写入 `DEEPSEEK_API_KEY=...`"
                       "（参考 .env.example）")

    st.title("🏦 A 股银行财报问答")
    st.caption("语料：12 家 A 股上市银行 2026 年半年度报告，2,553 块。"
               "答案由 DeepSeek 依据召回材料生成，出处标注到「公司 · 章节 · 页码」。")

    question = st.text_input("请输入问题", value="招商银行 2026 年上半年净息差是多少？",
                             placeholder="例如：工商银行上半年手续费及佣金净收入是多少、同比变化如何？")
    col_ask, col_cmp = st.columns([1, 3])
    ask_clicked = col_ask.button("提问", type="primary", use_container_width=True)

    if ask_clicked and question.strip():
        t0 = time.time()
        with st.spinner("检索 + 生成中 …"):
            st.session_state["result"] = answer(
                question.strip(), r, mode=mode, top=top,
                company_filter=use_company, expand=use_expand)
            st.session_state["elapsed"] = time.time() - t0

    if "result" in st.session_state:
        out = st.session_state["result"]
        if out["question"] != question.strip():
            st.info(f"下面是上一次提问「{out['question']}」的结果，改了问题请重新点「提问」。")
        st.divider()
        render_result(out, st.session_state.get("elapsed", 0.0))

        with st.expander("三种检索模式的召回对比（同一问题）"):
            st.caption("固定公司过滤与块数，只换检索模式。看前 6 条里哪些块是某一模式独有的。")
            if st.button("跑对比"):
                rows = {}
                for label, m in MODES.items():
                    res = r.retrieve(out["question"], top=top, mode=m,
                                     company_filter=use_company, expand=use_expand)
                    rows[label] = [f"{h['name']} · {h['section']} · p{h['page_start']}"
                                   for h in res["hits"]]
                st.table(rows)
    elif not ask_clicked:
        st.info("输入问题后点「提问」。没有配置 .env 也能用——页面会照常展示召回的块，"
                "只是答案那一栏会提示缺少 DEEPSEEK_API_KEY。")


if __name__ == "__main__":
    main()
