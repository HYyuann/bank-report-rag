import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = json.load(open("eval_raw.json", encoding="utf-8"))
for row in R:
    if row["id"] not in (2, 4, 5, 7, 10):
        continue
    print("=" * 95); print(f"[{row['id']}] {row['question'][:70]}")
    for label, c in row["configs"].items():
        head = " / ".join(f"{h['name']}·{h['section'][:10]}·p{h['ps']}" for h in c["hits"][:3])
        print(f"\n--- {label} | {c['n_hit']}/6 | 首条{'✓' if c['top1_ok'] else '✗'} | "
              f"{'拒答' if c['refused'] else '作答'} | GT{c['gt_in_answer']}")
        print(f"    top3: {head}")
        print("    " + c["answer"][:430].replace("\n", "\n    "))
