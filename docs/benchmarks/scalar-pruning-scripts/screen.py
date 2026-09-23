import os
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(os.environ["PLAINMP_SCALAR_STUDY"])
DEFAULT_SCENES = ['panda','panda_hard','fetch_table','fetch_spheres4','fetch_spheres9','panda_boxes','fetch_table_tilted']


def exchange(process, command=None):
    if command is not None:
        process.stdin.write(json.dumps(command)+'\n')
        process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError('worker exited: '+str(process.poll()))
    return json.loads(line)


def worker(args):
    from benchmark import load_variant, create_scene
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    native = load_variant(args.worker)
    from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig, set_log_level_none, set_random_seed
    import numpy as np
    set_log_level_none()
    set_random_seed(args.seed)
    print(json.dumps({'ready': args.worker}),flush=True)
    for line in sys.stdin:
        command=json.loads(line)
        if command['action']=='scene':
            robot,cst,problem=create_scene(command['scene'])
            radius_steps = None
            if hasattr(cst, 'set_motion_certificate_steps'):
                radius_steps = float(args.worker[6:]) if args.worker.startswith('radius') else float(os.environ.get('PLAINMP_CERTIFICATE_STEPS', '6'))
                cst.set_motion_certificate_steps(radius_steps)
            reset=getattr(cst,'reset_clearance_cache',lambda:None)
            if not cst.is_valid(problem.start) or not cst.is_valid(problem.goal_const):
                raise RuntimeError('invalid endpoint')
            solver=OMPLSolver(OMPLSolverConfig(n_max_call=1000000))
            for _ in range(30):
                reset()
                if not solver.solve(problem).success:
                    raise RuntimeError('warmup failure')
            data=np.load(ROOT/'results'/('validation_'+command['scene']+'.npz'))
            actual=np.array([cst.is_valid(q) for q in data['q']])
            result={'scene':command['scene'],'independent_queries':len(actual),'independent_mismatches':int(np.sum(actual!=data['expected']))}
            result['certificate_radius_steps'] = radius_steps
            if hasattr(cst, 'reset_motion_certificate_stats'): cst.reset_motion_certificate_stats()
        else:
            result={'ns_internal':[],'calls':[],'paths':[],'ns_wall':[]}
            for _ in range(command['n']):
                reset()
                ts=time.perf_counter_ns()
                ret=solver.solve(problem)
                result['ns_wall'].append(time.perf_counter_ns()-ts)
                if not ret.success:
                    raise RuntimeError('planning failed')
                result['ns_internal'].append(ret.ns_internal)
                result['calls'].append(ret.n_call)
                result['paths'].append(hashlib.sha256(ret.traj.numpy().tobytes()).hexdigest())
            if hasattr(cst, 'motion_certificate_stats'):
                result['certificate_stats']=cst.motion_certificate_stats()
        print(json.dumps(result),flush=True)


def driver(args):
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
    with (ROOT/'results'/(args.tag+'-worker.log')).open('w') as log:
        for seed in args.seeds:
            processes={}
            try:
                for v in args.variants:
                    processes[v]=subprocess.Popen([sys.executable,__file__,'--worker',v,'--seed',str(seed)],env=env,text=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,bufsize=1)
                    exchange(processes[v])
                for scene in args.scenes:
                    data={v:dict(exchange(p,{'action':'scene','scene':scene}),blocks=[],variant=v,seed=seed) for v,p in processes.items()}
                    for b in range(args.blocks):
                        shift=b%len(args.variants)
                        for v in args.variants[shift:]+args.variants[:shift]:
                            data[v]['blocks'].append(exchange(processes[v],{'action':'block','n':args.n}))
                    baseline=data[args.variants[0]]
                    brief={}
                    for v,d in data.items():
                        paths=[h for b in d['blocks'] for h in b['paths']]
                        calls=[h for b in d['blocks'] for h in b['calls']]
                        refs=[h for b in baseline['blocks'] for h in b['paths']]
                        refc=[h for b in baseline['blocks'] for h in b['calls']]
                        d['path_mismatches']=sum(a!=b for a,b in zip(paths,refs))
                        d['call_mismatches']=sum(a!=b for a,b in zip(calls,refc))
                        (ROOT/'results'/f'{args.tag}_{scene}_{v}_s{seed}.json').write_text(json.dumps(d,indent=2))
                        ns=statistics.median(t for b in d['blocks'] for t in b['ns_internal'])
                        refns=statistics.median(t for b in baseline['blocks'] for t in b['ns_internal'])
                        brief[v]={'us':round(ns/1000,3),'speedup':round(refns/ns,3),'paths_differ':d['path_mismatches'],'labels_differ':d['independent_mismatches']}
                    print(seed,scene,json.dumps(brief),flush=True)
            finally:
                for p in processes.values():
                    p.stdin.close()
                for p in processes.values():
                    p.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--worker')
    p.add_argument('--seed',type=int)
    p.add_argument('--variants',nargs='+',default=['scalar','sse','avx'])
    p.add_argument('--scenes',nargs='+',default=DEFAULT_SCENES)
    p.add_argument('--seeds',nargs='+',type=int,default=[32452843])
    p.add_argument('--tag',default='simd-screen')
    p.add_argument('--blocks',type=int,default=3)
    p.add_argument('--n',type=int,default=100)
    args=p.parse_args()
    worker(args) if args.worker else driver(args)
