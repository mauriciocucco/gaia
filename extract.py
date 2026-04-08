import json
with open('debug_full.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
for item in data:
    ans = str(item.get('submitted_answer', 'None')).replace('\n', ' ')[:100]
    err = item.get('error', '')
    taskId = item.get('task_id', '')
    idx = item.get('index', '?')
    print(f'Task {idx} ({taskId[:8]}): Ans={ans} | Err={err}')

