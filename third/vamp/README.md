# VAMP algorithm attribution

The RRTC search procedure and Halton recurrence in
`cpp/plainmp/ompl/vamp_rrtc.{hpp,cpp}` are adapted from
[KavrakiLab/vamp](https://github.com/KavrakiLab/vamp), revision
`f6de6a72e0725ba08e0829f0a34261a3f465aed7`, distributed under Apache-2.0.
The upstream license is reproduced in [LICENSE](LICENSE).

Changes: generic double precision configurations, OMPL integration, pooled
storage, an independently implemented batch KD-tree, integer Halton arithmetic,
bounded termination, and exact connection endpoint handling. VAMP's generated
collision checker, SIMD implementation, and Nigh dependency are not vendored.
