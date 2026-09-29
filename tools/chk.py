import json, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CH = [json.loads(l) for l in open("chunks.jsonl", encoding="utf-8") if l.strip()]
def find(code, ps, pe):
    for c in CH:
        if c["code"] == code and c["page_start"] == ps and c["page_end"] == pe:
            return c
    return None

print("### 交通银行 p10-12 块里有没有 203.80？")
c = find("601328", 10, 12)
if c:
    print("  含 '203.80'：", "203.80" in c["text"])
    print("  含 '拨备覆盖率'：", "拨备覆盖率" in c["text"])
    print("  字符数：", len(c["text"]))
    i = c["text"].find("拨备覆盖率")
    print("  片段：", c["text"][max(0,i-60):i+200].replace("\n", " ⏎ ") if i >= 0 else "(无)")
print()
print("### 交通银行 p10-10 块（truth 里 203.80 所在）")
c = find("601328", 10, 10)
if c:
    print("  含 '203.80'：", "203.80" in c["text"], " 字符数：", len(c["text"]))
    i = c["text"].find("拨备覆盖率")
    print("  片段：", c["text"][max(0,i-80):i+230].replace("\n", " ⏎ "))
