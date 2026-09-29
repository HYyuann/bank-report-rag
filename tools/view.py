import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = json.load(open("eval_raw.json", encoding="utf-8"))
for row in R:
    if row["id"] not in (6,):
        continue
    print("=" * 95); print(f"[{row['id']}] {row['question']}")
    for label, c in row["configs"].items():
        print(f"\n--- {label} | top-6 命中 {c['n_hit']}/6 | 首条{'✓' if c['top1_ok'] else '✗'} | "
              f"{'拒答' if c['refused'] else '作答'}")
        for i, h in enumerate(c["hits"], 1):
            print(f"   {i}. {h['name']} · {h['section'][:16]} · p{h['ps']}-{h['pe']} "
                  f"[v{h['vec_rank']} b{h['bm_rank']} rrf{h['rrf']}] {'+'.join(h['rounds'])}")
        print("  答案：" + c["answer"][:600].replace("\n", "\n        "))
