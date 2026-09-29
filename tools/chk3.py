import json, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CH = [json.loads(l) for l in open("chunks.jsonl", encoding="utf-8") if l.strip()]

print("### 1) 建设银行 拨备覆盖率 上下文（p8-10 那块被截成 '拨备覆盖率2'）")
for c in CH:
    if c["code"] == "601939" and "拨备覆盖率" in c["text"]:
        i = c["text"].find("拨备覆盖率")
        print(f"  p{c['page_start']}-{c['page_end']} {len(c['text'])}字 has_table={c['has_table']}")
        print("   …" + c["text"][max(0,i-260):i+200].replace("\n", "⏎") + "…")

print("\n### 2) 谁含 '238.69'")
for c in CH:
    if "238.69" in c["text"]:
        i = c["text"].find("238.69")
        print(f"  {c['name']} {c['section'][:12]} p{c['page_start']}-{c['page_end']} | "
              + c["text"][max(0,i-120):i+80].replace("\n", "⏎"))

print("\n### 3) 兴业银行 两处拨备覆盖率 的完整上下文")
for c in CH:
    if c["code"] == "601166" and "拨备覆盖率" in c["text"]:
        i = c["text"].find("拨备覆盖率")
        seg = c["text"][max(0,i-400):i+300].replace("\n", "⏎")
        print(f"  --- p{c['page_start']}-{c['page_end']} ---\n  …{seg}…")

print("\n### 4) Q5 混合检索到底召回了哪 6 块")
R = json.load(open("eval_raw.json", encoding="utf-8"))
for row in R:
    if row["id"] != 5: continue
    for lab, c in row["configs"].items():
        print(f"  {lab}:")
        for h in c["hits"]:
            print(f"     {h['name']}·{h['section'][:14]}·p{h['ps']}-{h['pe']} "
                  f"v{h['vec_rank']} b{h['bm_rank']}")
