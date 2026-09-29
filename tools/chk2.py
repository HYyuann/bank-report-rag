import json, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CH = [json.loads(l) for l in open("chunks.jsonl", encoding="utf-8") if l.strip()]

def show(title, pred, n=3, win=150):
    print("=" * 90); print(title)
    k = 0
    for c in CH:
        if not pred(c): continue
        k += 1
        if k > n: break
        t = c["text"]
        i = t.find("净利润") if "净利润" in t else 0
        print(f"  [{c['name']} {c['section']} p{c['page_start']}-{c['page_end']} "
              f"{len(t)}字] ...{t[max(0,i-win):i+win]}...")
    print(f"  (共 {k} 条匹配展示)")

# 1) 工行「净利润」在哪些块（Q5 的关键：模型说材料里没有净利润）
show("工行 块内含『净利润』", lambda c: c["code"] == "601398" and "净利润" in c["text"], 6, 90)

# 2) 招行 p72 那块里的 1,361.84（Q10 的「真数字但口径不对」陷阱）
for c in CH:
    if c["code"] == "600036" and "1,361.84" in c["text"]:
        i = c["text"].find("1,361.84")
        print("=" * 90); print(f"招行 {c['section']} p{c['page_start']}-{c['page_end']}")
        print(f"  ...{c['text'][max(0,i-200):i+200]}...")

# 3) Q9 缺的两家：建设银行 / 兴业银行 拨备覆盖率
for code, nm in (("601939", "建设银行"), ("601166", "兴业银行")):
    print("=" * 90); print(f"{nm} 块内出现『拨备覆盖率』")
    for c in CH:
        if c["code"] != code or "拨备覆盖率" not in c["text"]: continue
        t = c["text"]
        for m in re.finditer("拨备覆盖率", t):
            seg = t[m.start():m.start() + 110].replace("\n", "⏎")
            print(f"  p{c['page_start']}-{c['page_end']} {c['section'][:12]} | {seg}")
