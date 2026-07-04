"""第二阶段汇总：读所有 *_seed*/metrics.json -> results_summary.csv + results_mean_std.csv。
results_summary.csv: 每行 seed,backbone,method,noise_rate,accuracy,balanced_accuracy,macro_f1
results_mean_std.csv: 按 (method, noise_rate) 跨 3 seed 算 mean/std。
"""
import os, json, csv, glob
from collections import defaultdict
import numpy as np

OUT = os.path.expanduser('~/aa_workspace/experiments/pre_exp_stage2')
rows = []
groups = defaultdict(lambda: defaultdict(list))   # (method, noise_rate) -> metric -> [vals over seeds]

for f in sorted(glob.glob(f'{OUT}/*_seed*/metrics.json')):
    m = json.load(open(f))
    t = m['test']
    rows.append({'seed': m['seed'], 'backbone': m['backbone'], 'method': m['method'],
                 'noise_rate': m['noise_rate'],
                 'accuracy': round(t['accuracy'], 4),
                 'balanced_accuracy': round(t['balanced_accuracy'], 4),
                 'macro_f1': round(t['macro_f1'], 4)})
    key = (m['method'], m['noise_rate'])
    for k in ['accuracy', 'balanced_accuracy', 'macro_f1']:
        groups[key][k].append(t[k])

with open(f'{OUT}/results_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['seed', 'backbone', 'method', 'noise_rate',
                                      'accuracy', 'balanced_accuracy', 'macro_f1'])
    w.writeheader()
    for r in sorted(rows, key=lambda x: (x['noise_rate'], x['method'], x['seed'])):
        w.writerow(r)

with open(f'{OUT}/results_mean_std.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['method', 'noise_rate', 'n_seeds',
                'accuracy_mean', 'accuracy_std',
                'balanced_accuracy_mean', 'balanced_accuracy_std',
                'macro_f1_mean', 'macro_f1_std'])
    for (method, noise), d in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        def ms(k):
            a = np.array(d[k]); return round(a.mean(), 4), round(a.std(), 4)
        am, asd = ms('accuracy'); bm, bsd = ms('balanced_accuracy'); fm, fsd = ms('macro_f1')
        w.writerow([method, noise, len(d['accuracy']), am, asd, bm, bsd, fm, fsd])

print(f"汇总完成: {len(rows)} 行 -> results_summary.csv；{len(groups)} 组 -> results_mean_std.csv")
print("\n=== results_mean_std.csv ===")
print(open(f'{OUT}/results_mean_std.csv').read())
