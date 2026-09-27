Search semantics in `plainmp_rrtc` follow OMPL 1.6.0 RRTConnect, without intermediate states.

Source: https://github.com/ompl/ompl/blob/1.6.0/src/ompl/geometric/planners/rrt/src/RRTConnect.cpp

The implementation uses contiguous pools, integer parent/root indices, and an exact KD-tree.
See LICENSE for the upstream BSD license.
