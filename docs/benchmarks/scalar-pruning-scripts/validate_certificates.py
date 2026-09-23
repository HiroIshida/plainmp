import os
"""Check scalar interval certificates against ordinary point validation."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from benchmark import load_variant, create_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='endpoint')
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()
    os.sched_setaffinity(0, {2})
    load_variant(args.variant)
    output = {}
    rng = np.random.RandomState(8675309)
    root = Path(os.environ["PLAINMP_SCALAR_STUDY"])
    for scene in ['panda','panda_hard','fetch_table','fetch_spheres4', 'panda_boxes','fetch_table_tilted']:
        robot, cst, problem = create_scene(scene)
        cst.set_motion_certificate_steps(6)
        data = np.load(root/'results'/('validation_'+scene+'.npz'))
        lower, upper = robot.angle_bounds()
        anchors = data['q'][rng.choice(len(data['q']), args.n, replace=False)]
        certified = queries = unproven = 0
        for anchor in anchors:
            direction = rng.normal(size=anchor.size)
            direction /= np.linalg.norm(direction)
            length = rng.choice([.03125,.125,.25,.5,1.,2.])
            a, b = np.clip(anchor-length*direction,lower,upper), np.clip(anchor+length*direction,lower,upper)
            for rate in [0., .5, 1.]:
                cst.is_valid(a)
                assert cst.prepare_motion_certificate(a,b,.25)
                q = a+(b-a)*rate
                valid, radius = cst.check_motion_certificate(q)
                assert valid == cst.is_valid(q), (scene,'anchor mismatch')
                queries += 1
                if radius == 0:
                    unproven += 1
                    continue
                certified += 1
                lo, hi = max(0.,rate-radius), min(1.,rate+radius)
                rates = np.r_[lo,hi,np.linspace(lo,hi,9),rng.uniform(lo,hi,8)]
                for t in rates:
                    assert cst.is_valid(a+(b-a)*t), (scene,'false free',q,radius,t)
                    queries += 1
        # SDF replacement invalidates the prepared workspace.
        cst.set_sdf(cst.get_sdf())
        valid, radius = cst.check_motion_certificate(problem.start)
        assert radius == 0 and valid == cst.is_valid(problem.start)
        huge = problem.start.copy(); huge[-1] = 1001
        assert not cst.prepare_motion_certificate(problem.start,huge,.25)
        result = dict(anchors=len(anchors)*3, certified=certified, unproven=unproven,
                      point_queries=queries, mismatches=0)
        output[scene] = result
        print(scene,json.dumps(result),flush=True)
    (root/'results'/('certificate-validation-'+args.variant+'.json')).write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__': main()
