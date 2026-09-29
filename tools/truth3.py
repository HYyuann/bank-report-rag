import json, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CH = [json.loads(l) for l in open("chunks.jsonl", encoding="utf-8") if l.strip()]

def show(code, name, kw, width=230, limit=5):
    print(f"-- {name} / {kw}")
    n = 0
    for c in CH:
        if c["code"] != code or kw not in c["text"]:
            continue
        for m in re.finditer(re.escape(kw), c["text"]):
            seg = c["text"][m.start(): m.start()+width].replace("\n", " ⏎ ")
            print(f"   p{c['page_start']}-{c['page_end']} {c['section'][:14]} | {seg}")
            n += 1
            break
        if n >= limit:
            break

for code, name in [("601939","建设银行"),("601166","兴业银行"),("600000","浦发银行"),("601998","中信银行")]:
    show(code, name, "拨备覆盖率", 200, 3)
show("601288", "农业银行", "利息净收入", 170, 4)
show("601939", "建设银行", "1.29", 170, 3)
