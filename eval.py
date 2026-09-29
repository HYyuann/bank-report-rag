#!/usr/bin/env python3
"""第五步：10 道题 × 3 种配置的逐题评测。

    .venv/Scripts/python.exe eval.py            # 全量，结果写 eval_report.md + eval_raw.json

三种配置（口径扩展一律开着，这是当前默认）：
    只用向量        mode=vector, 无公司过滤
    只用 BM25       mode=bm25,   无公司过滤
    混合+公司过滤   mode=hybrid, 有公司过滤

注意前两种**不带公司过滤**：这样对比才看得出「公司过滤」到底贡献了多少，
否则三种配置在单公司题上全都是 6/6，表里看不出差别。

ground truth 由 truth.py / truth2.py / truth3.py 直接从 chunks.jsonl 里核出来，
写在下面的 GT 字段里，用于自动判定答案对错。
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

from ask import answer          # noqa: E402
from rag import Retriever       # noqa: E402

CONFIGS = [
    ("只用向量", "vector", False),
    ("只用 BM25", "bm25", False),
    ("混合+公司过滤", "hybrid", True),
]

# (编号, 题型, 问题, 目标公司代码 or None, ground truth 关键串, 备注)
QUESTIONS = [
    (1, "单指标", "工商银行 2026 年上半年手续费及佣金净收入是多少、同比变化如何？",
     "601398", ["692.35", "3.3"], "干扰项：手续费及佣金**收入** 765.91 亿"),
    (2, "单指标", "建设银行 2026 年上半年不良贷款率是多少？",
     "601939", ["1.29"], "p34 有「民营企业贷款不良率 1.46%」干扰"),
    (3, "单指标", "招商银行 2026 年上半年净利息收益率是多少？",
     "600036", ["1.83"], "另有本公司口径 1.87%（p15）"),
    (4, "藏在附注", "招商银行 2026 年上半年信用减值损失是多少？",
     "600036", ["291.77", "29,177"], "主要指标表只写「信用减值损失/贷款均值」，数值在利润表"),
    (5, "藏在附注", "工商银行 2026 年上半年经营活动产生的现金流量净额与净利润相比差多少、"
     "主要差异来自哪些项目？",
     "601398", ["13,690.13", "1,369,013", "1,764.75"], "需跨「主要指标」与现金流量表附注"),
    (6, "长口语", "我最近在挑银行股，先翻了平安和兴业的报告，感觉息差都在往下走，又看了几家大行的资本充足率，"
     "反正就想弄清楚一件事：交通银行 2026 年上半年末的拨备覆盖率是多少？",
     "601328", ["203.80"], "铺垫里出现平安/兴业，看会不会被带跑"),
    (7, "一句话多子问题", "我想一次问清三件事：农业银行 2026 年上半年的净利息收入、不良贷款率、"
     "拨备覆盖率分别是多少？",
     "601288", ["3,122.44", "1.25", "290.10"], "三个指标、三个不同页"),
    (8, "跨公司全景", "2026 年上半年哪几家银行的营业收入同比增速最高？",
     None, [], "单块检索结构性无法回答"),
    (9, "跨公司全景", "12 家银行里，哪些银行的拨备覆盖率比 2025 年末上升了？",
     None, [], "需 12 家逐一比对两个时点"),
    (10, "边界/抗幻觉", "招商银行 2025 年全年净利润是多少？",
     "600036", [], "语料只有 2026H1，2025 全年不在语料里，应拒答"),
]


def run_one(r: Retriever, q: str, mode: str, company_filter: bool) -> dict:
    t0 = time.time()
    out = answer(q, r, mode=mode, top=6, company_filter=company_filter, expand=True)
    out["_wall"] = time.time() - t0
    return out


def main() -> int:
    r = Retriever(verbose=False)
    _ = r.model                     # 预热，别把模型加载算进第一条
    print(f"索引 {r.info['count']} 块 · {r.info['model']}\n")

    results = []
    for qid, qtype, q, target, gt, note in QUESTIONS:
        print(f"[{qid:>2}] {q[:44]}…")
        row = {"id": qid, "type": qtype, "question": q, "target": target,
               "gt": gt, "note": note, "configs": {}}
        for label, mode, cf in CONFIGS:
            out = run_one(r, q, mode, cf)
            hits = out["hits"]
            n_hit = sum(1 for h in hits if h["code"] == target) if target else None
            top1 = (hits[0]["code"] == target) if (target and hits) else None
            ans = out.get("generation", {}).get("answer", "")
            err = out.get("generation_error")
            row["configs"][label] = {
                "hits": [{"code": h["code"], "name": h["name"], "section": h["section"],
                          "ps": h["page_start"], "pe": h["page_end"],
                          "vec_rank": h["vec_rank"], "bm_rank": h["bm_rank"],
                          "rrf": round(h["rrf"], 5), "rounds": h["rounds"]} for h in hits],
                "n_hit": n_hit,           # top-6 里命中正确公司的条数
                "top1_ok": top1,          # 首条是否为正确公司
                "expansion": out.get("expansion", []),
                "answer": ans,
                "generation_error": err,
                "refused": "无法回答" in ans,
                "gt_in_answer": [g for g in gt if g in ans],   # 命中的 ground truth 串
                "seconds": round(out["_wall"], 1),
            }
            tag = f"{n_hit}/6 首条{'✓' if top1 else '✗'}" if target else "跨公司"
            print(f"      {label:<12} {tag:<14} "
                  f"{'拒答' if '无法回答' in ans else '作答'} "
                  f"GT命中 {len(row['configs'][label]['gt_in_answer'])}/{len(gt)}")
        results.append(row)

    (ROOT / "eval_raw.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n原始结果 -> eval_raw.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
