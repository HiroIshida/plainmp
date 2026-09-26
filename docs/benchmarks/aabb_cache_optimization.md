球グループ AABB の生成処理最適化（2026-09-26）

生成処理をさらに最適化した。座標変換と AABB 生成を合わせた単体測定では
約26〜35%短縮。`example/bench` の計画全体でも前回の AABB 版に対し
Panda 0.9%、Panda difficult 1.8%、Fetch 4.7%短縮した。
ただし AABB なしと比べると Panda 系はまだ遅く、Fetch は同程度だった。

比較対象は同じ `7574426caf41b9964f3acb47cc89f05728f05288` を基にした3ビルド。
`baseline` は AABB なし、`fused` は [前回の実装](aabb_cache.md)、
`optimized` は現在の作業ツリー。certificate はこの checkout に含まれない。
前回の測定値を流用せず、3ビルドを今回あらためて比較した。

| ケース | AABB なし ms | 前回 AABB ms | 改善 AABB ms | 前回 AABB 比 | AABB なし比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Panda | 0.21042 | 0.21594 | 0.21410 | −0.86% | +1.75% |
| Panda difficult | 0.77554 | 0.80939 | 0.79480 | −1.80% | +2.48% |
| Fetch | 0.78264 | 0.81337 | 0.77522 | −4.69% | −0.95% |

値は `OMPLSolverResult.time_elapsed` の平均。各ケース・各ビルド **50,400 solve**。
seed 301〜312 の12個を使い、各プロセスで300回ウォームアップ、4,200回計測した。
ビルドの順序は3!通りを2周して均等化し、CPU 2 に固定して順次実行した。
全453,600 solve が成功。同じ seed に対応する151,200組すべてで、
**3ビルドの経路 SHA-256 と衝突判定回数が完全一致**した。

12 seed のうち改善版が前回版より速かったのは、Panda 11、difficult 12、Fetch 11。
seed ごとの対応を保ってブロック単位で20,000回 bootstrap した平均時間比の
95%区間は、それぞれ −1.29〜−0.38%、−2.90〜−0.97%、−6.46〜−2.23%。
個々の solve を独立した実行環境の反復とは扱っていない。
Fetch の AABB なし比は −2.56〜+0.22%で、速度向上を断定できない。
外れ値は除いておらず、実行中の周波数や負荷の揺れも残る。

ハードウェア・設定は前回と同じ AMD Ryzen 7 7840HS、turbo 有効、GCC 9.4、
Release / `-O3 -DNDEBUG -flto`、`EIGEN_DONT_VECTORIZE`、Python 3.8.10、
nanobind 2.9.2、OMPL 1.6。`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。
RRTConnect、range 2.0、refinement なし、Box validator の幅0.05。
計測用カウンタ、コンパイラフラグ、探索条件は追加・変更していない。

実装では、球位置を作った直後のローカル値を同じループ内で境界計算へ渡す。
位置キャッシュへの保存は自己衝突・勾配計算等でも使うため維持した。
前回版もループは融合済みで、今回はその中の仕事を減らしている。

- 半径の有限性・非負性・全半径が同じかを、姿勢によらない情報として初期化時に計算。
- 境界の6成分をスカラーで集約し、Eigen の境界ベクトルを最後に1回だけ構成。
- 中心座標の総和に NaN / Inf を伝播させ、有限性をループ後に1回確認。
  min/max が途中の NaN を捨てても検出できる。有限値の総和が overflow した場合も
  安全側に倒して AABB を使わない。`-ffast-math` は使っていない。
- 共通半径なら中心座標の min/max を取った後で半径を加減算。
  加減算の単調性により元の球ごとの集約と同じ境界になる。
  半径が異なる場合は各球の半径を使い、最大半径で一律に広げる近似はしていない。

混在半径の生成ループを同じビルド条件で逆アセンブルすると、条件ジャンプが
大きく減った。スタックへの一時退避は残っており、全成分がレジスタに留まると
いう主張ではない。計画全体の改善率は上の実測による。

明示的な SSE2、回転・並進のローカルコピー、検査用総和の3軸分割も試したが、
安定した改善は得られなかった。最大半径による一律拡張も試したが、混在半径で
境界が緩くなるため採用していない。丸め余裕の最小値を通常の最小正数に替える
案にも利点がなく、従来の `denorm_min()` を維持した。

単体ベンチは [bench_sphere_group_cache.cpp](../../bench/bench_sphere_group_cache.cpp)。
混在半径0.03〜0.09、固定姿勢で位置と AABB のキャッシュを毎回無効化して再生成する。
1 pass あたり `20,000,000 / 球数` 回、1 pass ウォームアップ後の5 pass の中央値。
既に計算済みのリンク回転行列を使い、FK 本体と障害物判定は含まない。

| 球数 | 前回 ns/生成 | 改善 ns/生成 | 時間短縮 |
| --- | ---: | ---: | ---: |
| 8 | 40.34 | 29.99 | 25.7% |
| 16 | 78.60 | 54.75 | 30.3% |
| 128 | 594.09 | 385.51 | 35.1% |

この数値は保存したベンチソースをビルドした最終測定。作業途中の簡易版では
8球40.3→27.9 nsだったが、実行状態や呼び出し側のコード配置によって絶対時間は
変わる。単体の短縮率をそのまま計画全体へ当てはめることはできない。
Panda 系では追加する集約や分岐の費用がなお残る。Fetch では今回の改善によって
AABB なしとほぼ同じ速度になり、生成費用を下げる方向には効果があった。

全 Python テスト **144 passed**（NumPy / OMPL seed 42）。AABB 回帰テストは26件で、
姿勢変更、丸め境界、混在・共通半径、非有限中心が球列の先頭・途中・末尾にある場合、
有限中心の総和だけが overflow する場合等を確認した。これは検査した入力での一致であり、
全入力に対する数値的同値の証明ではない。

計画ベンチの再実行例:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  example/bench/compare_aabb.py \
  --module .cache/aabb/optimized/_plainmp.cpython-38-x86_64-linux-gnu.so \
  --case panda --samples 4200 --warmup 300 --seed 301 --cpu 2 \
  --output .cache/aabb/recheck_panda_301_optimized.json
```

`--case` を3ケース、`--module` を `baseline` / `fused` / `optimized` に替え、
seed 301〜312 を実行する。順序は `baseline,fused,optimized` の辞書式の6順列を2周。
この環境の一括実行スクリプトは `.cache/aabb/optimization/run_final.py`。
インストール済みの Python 拡張モジュールは変更していない。

単体ベンチは Release ビルドの core object にリンクできる:

```bash
aabb_build=.cache/aabb/build
c++ -std=c++17 -O3 -DNDEBUG -flto -DEIGEN_DONT_VECTORIZE -DEIGEN_NO_DEBUG \
  -Icpp -Ithird/urdf_parser/include -I/usr/include/eigen3 \
  bench/bench_sphere_group_cache.cpp \
  "$aabb_build/CMakeFiles/_plainmp.dir/cpp/plainmp/constraints/primitive_sphere_collision.cpp.o" \
  "$aabb_build/CMakeFiles/_plainmp.dir/cpp/plainmp/kinematics/kinematics.cpp.o" \
  "$aabb_build/CMakeFiles/_plainmp.dir/cpp/plainmp/kinematics/algorithm.cpp.o" \
  "$aabb_build/CMakeFiles/_plainmp.dir/cpp/plainmp/kinematics/kinematic_model_wrapper.cpp.o" \
  "$aabb_build/CMakeFiles/_plainmp.dir/cpp/plainmp/collision/kdtree.cpp.o" \
  "$aabb_build/third/urdf_parser/liburdfdom_model.a" \
  "$aabb_build/third/urdf_parser/tinyxml/libtinyxml.a" \
  -o .cache/aabb/bench_cache
taskset -c 2 .cache/aabb/bench_cache 8 aabb
```

前回版をビルドする場合は対応する旧ヘッダと旧 object を使い、
`PLAINMP_BENCH_PREVIOUS_AABB` を定義する。保存済みの比較実行ファイルは
`.cache/aabb/optimization/bench_previous` と `bench_optimized`。

集計、seed 別平均、bootstrap 区間、単体測定の全 pass、バイナリ SHA-256 は
[aabb_cache_optimization.json](aabb_cache_optimization.json) に保存した。
全 solve の生データは `.cache/aabb/optimization/final/`。
旧ソース、試した各実装、逆アセンブルは `.cache/aabb/optimization/` に保存した。
`.cache` 内の成果物は Git 管理対象外。
