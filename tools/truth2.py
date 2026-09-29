import json, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CH = [json.loads(l) for l in open("chunks.jsonl", encoding="utf-8") if l.strip()]
BANKS = [("601398","工商银行"),("601939","建设银行"),("601288","农业银行"),("601988","中国银行"),
         ("601328","交通银行"),("600036","招商银行"),("601166","兴业银行"),("600000","浦发银行"),
         ("601998","中信银行"),("600016","民生银行"),("601818","光大银行"),("000001","平安银行")]

print("### 各家 拨备覆盖率（取「主要财务指标」节里的第一处）")
for code, name in BANKS:
    for c in CH:
        if c["code"] != code or "拨备覆盖率" not in c["text"]:
            continue
        if "主要财务指标" not in c["section"]:
            continue
        i = c["text"].find("拨备覆盖率")
        print(f"{name:6} p{c['page_start']:<4} ...{c['text'][i:i+120].replace(chr(10),' ⏎ ')}")
        break
    else:
        print(f"{name:6} (主要财务指标节里没找到，需另找)")

print("\n### 补充：建行集团不良率 / 农行净利息收入 / 工行净利润")
for code, name, kw in [("601939","建设银行","不良贷款率"),("601288","农业银行","净利息收入"),
                       ("601398","工商银行","净利润")]:
    print(f"-- {name} / {kw}")
    shown = 0
    for c in CH:
        if c["code"] != code or kw not in c["text"]:
            continue
        for m in re.finditer(re.escape(kw), c["text"]):
            print(f"   p{c['page_start']}-{c['page_end']} {c['section'][:14]} | "
                  f"{c['text'][m.start():m.start()+150].replace(chr(10),' ⏎ ')}")
            shown += 1
            break
        if shown >= 3:
            break
