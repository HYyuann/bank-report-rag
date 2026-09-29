import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CH = [json.loads(l) for l in open("chunks.jsonl", encoding="utf-8") if l.strip()]
print("### 建设银行 p7-8（纯数字块）全文")
for c in CH:
    if c["code"] == "601939" and c["page_start"] == 7:
        print(c["text"].replace("\n", "⏎")[:1100])
print("\n### 兴业银行『公司简介与主要财务指标』各块（找上年末拨备）")
for c in CH:
    if c["code"] == "601166" and c["section"] == "公司简介与主要财务指标":
        t = c["text"]
        print(f"  p{c['page_start']}-{c['page_end']} {len(t)}字 | 含'拨备'{'拨备' in t} "
              f"含'覆盖率'{'覆盖率' in t} | {t[:150]!r}".replace("\n", "⏎"))
