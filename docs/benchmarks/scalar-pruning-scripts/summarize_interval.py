import os
import gzip, hashlib, json, math, statistics
from pathlib import Path
root=Path(os.environ["PLAINMP_SCALAR_STUDY"])
records=[json.loads(p.read_text()) for p in sorted((root/'results').glob('heldout_*.json'))]
assert len(records)==63
index={(r['scene'],r['variant'],r['seed']):r for r in records}
variants=['scalar','disabled','interval']; seeds=[67867967,67867979,67867987]
scenes=['panda','panda_hard','fetch_table','fetch_spheres4','fetch_spheres9','panda_boxes','fetch_table_tilted']
for r in records:
    assert r['path_mismatches']==r['call_mismatches']==r['independent_mismatches']==0
    assert sum(len(b['calls']) for b in r['blocks'])==600

def median(r,metric):return statistics.median(t for b in r['blocks'] for t in b[metric])
summary={'baseline_commit':'fe44f5ffccf38ab3a1f121723aec42893f05667c','branch':'research/scalar-pruning',
         'development_seed':32452843,'evaluation_seeds':seeds,'plans_per_scene_variant_seed':600,
         'variants':{},'scenes':{},'total_plans':37800,'matched_interval_plans':12600,
         'path_mismatches':0,'call_mismatches':0,'independent_label_mismatches':0,
         'independent_unique_queries_per_variant':sum(index[s,'scalar',seeds[0]]['independent_queries'] for s in scenes),
         'settings':{'cpu':2,'threads':1,'compiler':'GCC 9.4','build':'Release/O3/LTO/EIGEN_DONT_VECTORIZE',
                     'range':2,'resolution':1/32,'self_collision':True,'simplification':False,
                     'common_certificate_cap_steps':6,'warmup_per_scene':30,'interleaved_block_size':100,
                     'blocks_per_seed':6,'cpu_frequency_fixed':False}}
for v in variants:
    binary=next((root/'build'/v).glob('_plainmp*.so'))
    summary['variants'][v]={'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
                          'interval_option':v=='interval'}
for s in scenes:
    rows=[]
    for seed in seeds:
        ds={v:index[s,v,seed] for v in variants}
        r={'seed':seed}
        for v,d in ds.items():
            r[v+'_ms']=median(d,'ns_internal')/1e6
            r[v+'_wall_ms']=median(d,'ns_wall')/1e6
        r['speedup']=r['scalar_ms']/r['interval_ms']
        r['wall_speedup']=r['scalar_wall_ms']/r['interval_wall_ms']
        r['disabled_over_interval']=r['disabled_ms']/r['interval_ms']
        total_calls=sum(c for b in ds['interval']['blocks'] for c in b['calls'])
        r['logical_calls']=total_calls
        r['certificate_stats']=ds['interval']['blocks'][-1]['certificate_stats']
        r['point_queries_skipped_fraction']=r['certificate_stats'][3]/total_calls
        rows.append(r)
    entry={'seeds':rows}
    for key in rows[0]:
        if key not in ['seed','certificate_stats','logical_calls']:
            entry[key]=statistics.median(r[key] for r in rows)
    entry['speedup_min']=min(r['speedup'] for r in rows)
    entry['speedup_max']=max(r['speedup'] for r in rows)
    summary['scenes'][s]=entry
    print(s, ' '.join(f'{key}={entry[key]:.4f}' for key in ['scalar_ms','interval_ms','speedup','wall_speedup','disabled_over_interval','point_queries_skipped_fraction']))
summary['geometric_mean_scene_speedup']=math.exp(statistics.mean(math.log(e['speedup']) for e in summary['scenes'].values()))
print('geometric mean',summary['geometric_mean_scene_speedup'])
(root/'results/interval-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with gzip.open(root/'results/interval-raw.json.gz','wt') as f:json.dump(records,f,separators=(',',':'))
