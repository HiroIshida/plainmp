import os
"""Differential tests near box faces, edges, and corners with a Cartesian robot."""
import json, sys
from pathlib import Path
import numpy as np
from benchmark import load_variant
load_variant(sys.argv[1])
from plainmp.kinematics import KinematicModel, BaseType
from plainmp.constraint import SphereAttachmentSpec, SphereCollisionCst
from plainmp.psdf import BoxSDF, Pose
urdf='<robot name="cartesian"><link name="root"/>'
for i,axis in enumerate(['1 0 0','0 1 0','0 0 1']):
    parent='root' if i==0 else 'link'+str(i-1)
    urdf+=f'<link name="link{i}"/><joint name="joint{i}" type="prismatic"><parent link="{parent}"/><child link="link{i}"/><axis xyz="{axis}"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>'
urdf+='</robot>'
kin=KinematicModel(urdf)
radius=.03
spec=SphereAttachmentSpec('link2',np.zeros((3,1)),np.array([radius]),False)
cst=SphereCollisionCst(kin,['joint0','joint1','joint2'],BaseType.FIXED,[spec],[],None,True)
center=np.array([.2,-.1,.3]); half=np.array([.2,.3,.4])
box=BoxSDF(half*2,Pose(center,np.eye(3)))
cst.set_sdf(box)
rng=np.random.RandomState(49979687)
queries=certificates=0
for i in range(6000):
    # Include points infinitesimally inside/outside rounded faces, edges, corners.
    axes=rng.choice(3,1+i%3,replace=False)
    outward=np.zeros(3); outward[axes]=rng.uniform(.1,1,len(axes))
    outward/=np.linalg.norm(outward)
    signs=rng.choice([-1,1],3)
    p=center+signs*half*rng.uniform(0,1,3)
    p[axes]=center[axes]+signs[axes]*half[axes]
    p+=signs*outward*(radius+rng.choice([-1e-7,-1e-12,0,1e-12,1e-7,.005,.05]))
    direction=rng.normal(size=3); direction/=np.linalg.norm(direction)
    a,b=p-direction*.2,p+direction*.2
    assert cst.prepare_motion_certificate(a,b,.25)
    valid,r=cst.check_motion_certificate(p)
    assert valid==cst.is_valid(p),('point',p,valid,r)
    queries+=1
    if r:
        certificates+=1
        for t in np.linspace(.5-r,.5+r,31):
            q=a+(b-a)*t
            assert cst.is_valid(q),('interval',p,r,q)
            queries+=1
result=dict(anchors=6000,certificates=certificates,queries=queries,mismatches=0)
print(json.dumps(result))
root=Path(os.environ["PLAINMP_SCALAR_STUDY"])
(root/'results'/('box-certificate-validation-'+sys.argv[1]+'.json')).write_text(json.dumps(result,indent=2)+'\n')
