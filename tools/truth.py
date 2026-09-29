import json, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CH = [json.loads(l) for l in open("chunks.jsonl", encoding="utf-8") if l.strip()]

TARGETS = [
    ("Q1 工商银行 手续费及佣金净收入", "601398", ["手续费及佣金净收入", "手续费及佣金收入"]),
    ("Q2 建设银行 不良贷款率",         "601939", ["不良贷款率"]),
    ("Q3 招商银行 净利息收益率",       "600036", ["净利息收益率"]),
    ("Q4 招商银行 信用减值损失",       "600036", ["信用减值损失"]),
    ("Q5 工商银行 经营现金流/净利润",  "601398", ["经营活动产生的现金流量净额"]),
    ("Q6 交通银行 拨备覆盖率",         "601328", ["拨备覆盖率"]),
    ("Q7 农业银行 净利息收入/不良/拨备","601288", ["净利息收入", "不良贷款率", "拨备覆盖率"]),
]
for label, code, kws in TARGETS:
    print("=" * 100); print(label, f"（{code}）")
    for kw in kws:
        shown = 0
        for c in CH:
            if c["code"] != code or kw not in c["text"]:
                continue
            for m in re.finditer(re.escape(kw), c["text"]):
                seg = c["text"][m.start(): m.start() + 190].replace("\n", " ⏎ ")
                print(f"  [{kw}] p{c['page_start']}-{c['page_end']} {c['section'][:12]} | {seg}")
                shown += 1
                break                     # 每块只取首个匹配，够定位了
            if shown >= 4:
                break
