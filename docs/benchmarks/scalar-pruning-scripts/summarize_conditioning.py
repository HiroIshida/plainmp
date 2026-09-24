import os
"""Summarize held-out comparisons against both the prior interval and scalar versions."""
import gzip,hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(os.environ["PLAINMP_SCALAR_STUDY"])
seeds=[104729,130363,155921]
scenes=['panda','panda_hard','fetch_table','fetch_spheres4','fetch_spheres9','panda_boxes','fetch_table_tilted']
variants=['scalar','interval','conditioning_final']
records=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('conditioning-heldout_*.json'))]
assert len(records)==63
index={(r['scene'],r['variant'],r['seed']):r for r in records}
for r in records:
    assert r['path_mismatches']==r['call_mismatches']==r['independent_mismatches']==0
    assert sum(len(b['calls']) for b in r['blocks'])==600
summary={'baseline_commit':'caac3582018fd4e885232d39537d0c4316e3bf9b',
         'original_scalar_commit':'fe44f5ffccf38ab3a1f121723aec42893f05667c',
         'development_seed':32452843,'evaluation_seeds':seeds,'total_plans':37800,
         'paired_plans_against_each_baseline':12600,'unique_independent_queries_per_binary':220500,
         'path_mismatches':0,'call_mismatches':0,'label_mismatches':0,'variants':{},'scenes':{}}
for v in variants:
    p=next((ROOT/'build'/v).glob('_plainmp*.so'))
    summary['variants'][v]={'binary_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
for scene in scenes:
    rows=[]
    for seed in seeds:
        data={v:index[scene,v,seed] for v in variants}
        row={'seed':seed}
        for v,d in data.items():
            for metric,unit in [('ns_internal','ms'),('ns_wall','wall_ms')]:
                row[v+'_'+unit]=statistics.median(t for b in d['blocks'] for t in b[metric])/1e6
        row['additional_speedup']=row['interval_ms']/row['conditioning_final_ms']
        row['total_scalar_speedup']=row['scalar_ms']/row['conditioning_final_ms']
        row['additional_wall_speedup']=row['interval_wall_ms']/row['conditioning_final_wall_ms']
        row['old_stats']=data['interval']['blocks'][-1]['certificate_stats']
        row['new_stats']=data['conditioning_final']['blocks'][-1]['certificate_stats']
        rows.append(row)
    result={'seeds':rows}
    for key in rows[0]:
        if key not in ['seed','old_stats','new_stats']:
            result[key]=statistics.median(r[key] for r in rows)
    result['additional_speedup_range']=[min(r['additional_speedup'] for r in rows),max(r['additional_speedup'] for r in rows)]
    summary['scenes'][scene]=result
    print(scene, ' '.join(f'{k}={result[k]:.4f}' for k in ['interval_ms','conditioning_final_ms','additional_speedup','total_scalar_speedup','additional_wall_speedup']),result['additional_speedup_range'])
summary['geomean_additional_speedup']=math.exp(statistics.mean(math.log(r['additional_speedup']) for r in summary['scenes'].values()))
summary['geomean_total_scalar_speedup']=math.exp(statistics.mean(math.log(r['total_scalar_speedup']) for r in summary['scenes'].values()))
print('geomean',summary['geomean_additional_speedup'],summary['geomean_total_scalar_speedup'])
(ROOT/'results/conditioning-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with gzip.open(ROOT/'results/conditioning-raw.json.gz','wt') as f:json.dump(records,f,separators=(',',':'))
