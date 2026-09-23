import os
"""Contact-boundary, scene mutation, attachment, and fallback checks."""
import json
import sys
from pathlib import Path
import numpy as np
from benchmark import load_variant, create_scene

variant = sys.argv[1]
load_variant(variant)
from plainmp.constraint import SphereAttachmentSpec, SphereCollisionCst
from plainmp.kinematics import BaseType
from plainmp.psdf import BoxSDF, CloudSDF, Pose

root = Path(os.environ["PLAINMP_SCALAR_STUDY"])
rng = np.random.RandomState(982451653)
summary = {}
for scene in ['panda','panda_hard','fetch_table','fetch_spheres4','panda_boxes','fetch_table_tilted']:
    robot,cst,p = create_scene(scene)
    data = np.load(root/'results'/('validation_'+scene+'.npz'))
    free = data['q'][data['expected'].astype(bool)]
    colliding = data['q'][~data['expected'].astype(bool)]
    queries = 0
    for i in range(150):
        a = free[rng.randint(len(free))]
        b = colliding[rng.randint(len(colliding))]
        lo,hi = 0.,1.
        for _ in range(40):
            mid = (lo+hi)*.5
            if cst.is_valid(a+(b-a)*mid): lo = mid
            else: hi = mid
            queries += 1
        cst.is_valid(a)
        assert cst.prepare_motion_certificate(a,b,.25)
        for t in [lo,hi,max(0.,lo-1e-7),min(1.,hi+1e-7)]:
            q = a+(b-a)*t
            valid,radius = cst.check_motion_certificate(q)
            assert valid == cst.is_valid(q), (scene,'contact label')
            queries += 1
            if radius:
                for s in np.linspace(max(0.,t-radius),min(1.,t+radius),21):
                    assert cst.is_valid(a+(b-a)*s), (scene,'contact interval')
                    queries += 1
    summary[scene] = {'boundary_segments':150,'queries':queries,'mismatches':0}

robot,cst,p = create_scene('panda')
kin = robot.get_kin()
pose_before = kin.get_base_pose()
box = BoxSDF(np.array([.2,.3,.4]),Pose(np.array([.4,.1,.5]),np.eye(3)))
cst.set_sdf(box)

def probe(label, cst, a, b, expected=True):
    cst.is_valid(a)
    ready = cst.prepare_motion_certificate(a,b,.2)
    assert ready == expected, (label,'prepare',ready)
    count=0
    for t in np.linspace(0.,1.,21):
        q=a+(b-a)*t
        valid,radius = cst.check_motion_certificate(q)
        assert valid == cst.is_valid(q), label
        count+=1
        if not expected: assert radius==0
        if radius:
            for s in np.linspace(max(0.,t-radius),min(1.,t+radius),21):
                assert cst.is_valid(a+(b-a)*s), (label,'interval')
                count+=1
    summary[label]={'queries':count,'supported':expected,'mismatches':0}

probe('box',cst,p.start,p.goal_const)
box.translate(np.array([.03,-.02,.06]))
probe('translated_box',cst,p.start,p.goal_const)
box.rotate_z(.2)
probe('rotated_box',cst,p.start,p.goal_const)
kin.set_base_pose(np.array([.1,-.1,.05,0,0,np.sin(.1),np.cos(.1)]))
probe('base_changed_between_edges',cst,p.start,p.goal_const)
kin.set_base_pose(pose_before)
kin.add_new_link('interval_test_attachment','panda_link8',np.array([.1,.1,.1]),np.zeros(3),True)
probe('link_added_after_preparation',cst,p.start,p.goal_const)
attachment=SphereAttachmentSpec('interval_test_attachment',np.array([[.05],[0],[0]]),np.array([.03]),False)
attached=robot.create_collision_const(attachments=[attachment],use_cache=False)
attached.set_sdf(box)
probe('attachment',attached,p.start,p.goal_const)
small=SphereAttachmentSpec('panda_link8',np.zeros((3,1)),np.array([1e-7]),False)
tiny=robot.create_collision_const(attachments=[small],use_cache=False)
tiny.set_sdf(box)
probe('tiny_radius_fallback',tiny,p.start,p.goal_const,False)
cst.set_sdf(CloudSDF(rng.uniform(-1,1,(100,3)),.01))
probe('cloud_fallback',cst,p.start,p.goal_const,False)
planar=SphereCollisionCst(kin,robot.control_joint_names,BaseType.PLANAR,robot.parse_sphere_specs(),robot.get_self_collision_pairs(),None,True)
planar.set_sdf(box)
probe('planar_fallback',planar,np.r_[p.start,0,0,0],np.r_[p.goal_const,.1,.1,.1],False)
robot,cst,p=create_scene('fetch_table')
out=p.goal_const.copy();out[0]=100
assert not cst.prepare_motion_certificate(p.start,out,.2)
summary['prismatic_out_of_bounds_fallback']={'mismatches':0}
(root/'results'/('interval-edge-checks-'+variant+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
