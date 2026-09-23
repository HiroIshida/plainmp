import os
"""Collect perf samples only during warmed-up planning/query replay."""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time

ROOT=Path(os.environ["PLAINMP_SCALAR_STUDY"])


def worker(args):
    from benchmark import load_variant, create_scene
    native=load_variant(args.variant)
    from plainmp.ompl_solver import OMPLSolver,OMPLSolverConfig,set_log_level_none,set_random_seed
    import numpy as np
    set_log_level_none();set_random_seed(67867967)
    robot,cst,problem=create_scene(args.scene)
    solver=OMPLSolver(OMPLSolverConfig(n_max_call=1000000))
    reset=getattr(cst,'reset_clearance_cache',lambda:None)
    for _ in range(30):
        reset()
        if not solver.solve(problem).success:raise RuntimeError('warmup failed')
    if args.mode=='query':
        d=np.load(ROOT/'results'/(args.scene+'_trace.npz'))
        q,expected=d['q'],d['expected']
        check=native.experiment.replay_batch(cst,q,expected,1,True)
        if check[1]:raise RuntimeError('query labels differ')
    ctl=os.open(args.ctl,os.O_RDWR)
    ack=os.open(args.ack,os.O_RDWR)
    def control(command):
        os.write(ctl,(command+'\n').encode())
        if not select.select([ack],[],[],10)[0]:raise RuntimeError('perf control timeout')
        reply=os.read(ack,64)
        if reply.rstrip(b'\x00\r\n')!=b'ack':raise RuntimeError('perf did not acknowledge: '+repr(reply))
    control('enable')
    start=time.perf_counter();iterations=0;logical_calls=0;checksum=0;internal_ns=0
    while (iterations<args.plans if args.plans else time.perf_counter()-start<args.seconds):
        if args.mode=='plan':
            for _ in range(min(100,args.plans-iterations) if args.plans else 100):
                reset();ret=solver.solve(problem)
                if not ret.success:raise RuntimeError('planning failed')
                logical_calls+=ret.n_call;internal_ns+=ret.ns_internal;iterations+=1
        else:
            ns,bad,total=native.experiment.replay_batch(cst,q,expected,10,True)
            if bad:raise RuntimeError('query mismatch')
            iterations+=11;logical_calls+=len(q)*11;checksum+=total
    elapsed=time.perf_counter()-start
    control('disable')
    os.close(ctl);os.close(ack)
    print(json.dumps(dict(scene=args.scene,variant=args.variant,mode=args.mode,iterations=iterations,logical_calls=logical_calls,
        internal_seconds=internal_ns/1e9,wall_seconds=elapsed,checksum=checksum)),flush=True)


def driver(args):
    tag=args.tag or f'{args.variant}-{args.scene}-{args.mode}-{args.kind}'
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0',PERF_BUILDID_DIR=str(ROOT/'results/buildid-cache'))
    with tempfile.TemporaryDirectory(prefix='perf-',dir=ROOT/'results') as tmp:
        ctl=str(Path(tmp)/'ctl');ack=str(Path(tmp)/'ack');os.mkfifo(ctl);os.mkfifo(ack)
        command=['perf',args.kind,'-D','-1','--control',f'fifo:{ctl},{ack}']
        if args.kind=='record':
            command += ['-e','cycles:u','-F','999','--call-graph','dwarf,8192','--no-buildid-cache','-o',str(ROOT/'results'/(tag+'.data'))]
        else:
            command += ['-x',';','-e',args.events,'-o',str(ROOT/'results'/(tag+'.stat'))]
        command += ['--','taskset','-c','2',sys.executable,__file__,'--worker','--variant',args.variant,'--scene',args.scene,'--mode',args.mode,'--seconds',str(args.seconds),'--plans',str(args.plans),'--ctl',ctl,'--ack',ack]
        with (ROOT/'results'/(tag+'.log')).open('w') as log,(ROOT/'results'/(tag+'.json')).open('w') as output:
            subprocess.run(command,env=env,stdout=output,stderr=log,check=True,timeout=args.seconds+90)
    if args.kind=='record':
        for mode,extra in [('self',['--no-children','-g','none']),('calls',['--children','-g','graph,0.5,caller'])]:
            with (ROOT/'results'/(tag+'-'+mode+'.txt')).open('w') as out,(ROOT/'results'/(tag+'-report.log')).open('a') as log:
                subprocess.run(['perf','report','-i',str(ROOT/'results'/(tag+'.data')),'--stdio','--no-inline','--percent-limit','0.5','--sort','dso,symbol',*extra],env=env,stdout=out,stderr=log,check=True)
    print(tag,(ROOT/'results'/(tag+'.json')).read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true');p.add_argument('--ctl');p.add_argument('--ack')
    p.add_argument('--variant',default='profile');p.add_argument('--scene',default='fetch_table')
    p.add_argument('--mode',choices=['plan','query'],default='plan');p.add_argument('--seconds',type=float,default=10)
    p.add_argument('--plans',type=int,default=0)
    p.add_argument('--kind',choices=['record','stat'],default='record');p.add_argument('--tag')
    p.add_argument('--events',default='cycles:u,instructions:u,branches:u,branch-misses:u')
    args=p.parse_args();worker(args) if args.worker else driver(args)
