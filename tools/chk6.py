import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = json.load(open("eval_raw.json", encoding="utf-8"))
for row in R:
    if row["id"] not in (1, 3): continue
    for lab, c in row["configs"].items():
        if row["id"] == 1 and lab != "只用 BM25": continue
        if row["id"] == 3 and lab != "只用 BM25": continue
        print("=" * 88); print(f"[{row['id']}] {lab}  命中 {c['n_hit']}/6")
        for h in c["hits"]:
            print(f"   {h['name']}·{h['section'][:14]}·p{h['ps']}")
        print("   >>", c["answer"][:300].replace("\n", "\n   >> "))
