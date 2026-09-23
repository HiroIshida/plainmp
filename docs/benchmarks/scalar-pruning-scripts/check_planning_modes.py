import os
"""Verify real production modules across validator modes and generic constraints."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from benchmark import load_variant,create_scene
ROOT=Path(os.environ["PLAINMP_SCALAR_STUDY"])
variant=sys.argv[1];load_variant(variant)
from plainmp.ompl_solver import OMPLSolver,OMPLSolverConfig,set_random_seed,set_log_level_none,RefineType
from plainmp.constraint import LinkPositionBoundCst
set_log_level_none();set_random_seed(86028121)
rows=[]
for mode in ['euclidean_long','euclidean_short','box','generic','shortcut','budget']:
    robot,cst,p=create_scene('panda')
    joint_ids=robot.get_kin().get_joint_ids(robot.control_joint_names)
    config=OMPLSolverConfig(n_max_call=1000000)
    if mode=='euclidean_long':p.resolution=.005
    if mode=='euclidean_short':p.resolution=.7
    if mode=='box':p.validator_type='box';p.resolution=np.full(len(p.lb),.02)
    if mode=='generic':p.global_ineq_const=LinkPositionBoundCst(robot.get_kin(),robot.control_joint_names,robot.base_type,'panda_link8',2,-1000.,1000.)
    if mode=='shortcut':config.refine_seq=[RefineType.SHORTCUT]
    if mode=='budget':config.n_max_call=80
    solver=OMPLSolver(config)
    for i in range(100):
        ret=solver.solve(p)
        rows.append(dict(final_joints=robot.get_kin().get_joint_positions(joint_ids),mode=mode,calls=ret.n_call,success=ret.success,path=hashlib.sha256(ret.traj.numpy().tobytes()).hexdigest() if ret.success else None))
(ROOT/'results'/f'planning-modes_{variant}.json').write_text(json.dumps(rows,indent=2))
if variant!='scalar':
    base=json.loads((ROOT/'results/planning-modes_scalar.json').read_text())
    if rows!=base:
        bad=[i for i,(a,b) in enumerate(zip(base,rows)) if a!=b]
        raise RuntimeError(('mismatch',bad[:10],len(bad)))
print(variant,len(rows),'plans match' if variant!='scalar' else 'reference recorded')
