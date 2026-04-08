import json

files = {
    "last_run_answers": ".cache/gaia/last_run_answers.json",
    "compare_run_answers": ".cache/gaia/compare_run_answers.json",
}

for name, path in files.items():
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"{name}: not found")
        continue

    print(f"\n{'='*60}")
    print(f"File: {name}  ({len(data)} questions)")
    print(f"{'='*60}")
    for d in data:
        idx = d.get("index", "?")
        tid = d.get("task_id", "")[:10]
        ans = str(d.get("submitted_answer", "")).strip()[:60]
        err = d.get("error")
        err_str = f"  ERROR: {str(err)[:50]}" if err else ""
        print(f"[{idx:2}] {tid}.. | {ans!r}{err_str}")

    empty = sum(1 for d in data if not str(d.get("submitted_answer", "")).strip())
    errors = sum(1 for d in data if d.get("error"))
    print(f"\nSummary: {len(data)} total | {errors} with errors | {empty} empty answers")
