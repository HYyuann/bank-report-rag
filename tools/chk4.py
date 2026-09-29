import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CH = [json.loads(l) for l in open("chunks.jsonl", encoding="utf-8") if l.strip()]
for code, nm in (("601939", "建设银行"), ("601166", "兴业银行")):
    for c in CH:
        if c["code"] != code: continue
        if c["section"] != "公司简介与主要财务指标": continue
        t = c["text"]
        if "拨备覆盖率" not in t and "不良贷款率" not in t: continue
        i = t.find("资产质量")
        if i < 0: continue
        print("=" * 92)
        print(f"{nm} p{c['page_start']}-{c['page_end']}  {len(t)}字")
        print("  …" + t[i:i + 700].replace("\n", "⏎") + "…")
