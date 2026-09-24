# 区間pruningの条件判定・管理コストを減らす実験

## 結果

前回のスカラー区間pruning `caac3582018fd4e885232d39537d0c4316e3bf9b` を基準に、
さらに **1.026–1.096倍、7環境の幾何平均1.056倍** 改善した。
同時に測ったSIMD導入前 `fe44f5f` に対しては **1.363–1.771倍、幾何平均1.575倍**。
この追加改善はSIMDを使わず、既存の幾何学的条件判定を安く実行する変更で得た。
分岐予測だけで大きな追加改善を得た結果ではない。

branchは引き続き `research/scalar-pruning`。
`PLAINMP_ENABLE_INTERVAL_PRUNING` の既定値OFF、作業treeのbuildはON。
球配置、self collision pair、解像度、URDFの誤差上界、区間半径上限6 stepsは変更していない。

## 同じ条件の計画時間

Ryzen 7 7840HS、CPU 2、GCC 9.4、Release/O3/LTO、`EIGEN_DONT_VECTORIZE`。
RRTConnect range 2、Euclidean resolution 1/32、simplificationなし。
3 seed × 各600計画、各環境30 warmup、100計画ずつ3バイナリの順番を交代。
調整seedは32452843。評価には新しいseed104729、130363、155921を使った。

| 環境 | 前回 ms | 今回 ms | 前回比 | seed別範囲 | SIMD前比 | Pythonを含む前回比 |
|---|---:|---:|---:|---:|---:|---:|
| Panda | 0.1866 | 0.1733 | **1.077×** | 1.065–1.085× | 1.771× | 1.071× |
| Panda + ceiling | 0.7724 | 0.7034 | **1.096×** | 1.081–1.109× | 1.587× | 1.086× |
| Fetch table | 0.6707 | 0.6486 | **1.026×** | 1.016–1.049× | 1.659× | 1.015× |
| Fetch spheres × 4 | 0.2949 | 0.2822 | **1.042×** | 1.039–1.045× | 1.590× | 1.038× |
| Fetch spheres × 9 | 0.3602 | 0.3459 | **1.049×** | 1.041–1.054× | 1.363× | 1.046× |
| Panda tilted boxes | 0.2101 | 0.1949 | **1.069×** | 1.060–1.096× | 1.627× | 1.065× |
| Fetch tilted table | 0.5901 | 0.5773 | **1.033×** | 1.004–1.033× | 1.464× | 1.027× |

時間はseed別中央値の中央値。倍率は対応するseedの中央値の比を取り、その中央値。
全37,800計画を保存した。倍率は以前の測定結果同士を掛けた値ではなく、今回の同時測定による。
CPU周波数・背景負荷・ASLRは固定していない。数%の差には実行間変動があり、
Fetch tilted tableでは1 seedの改善は0.4%程度だった。

## perfで調べたこと

前回版をO3/LTOに `-g1 -fno-omit-frame-pointer` を足した診断用buildで調べた。
warm区間のみ `perf record`、cyclesとbranch-missesを別々に採取した。
Fetch tableのexclusive cyclesは区間証明30.1%、FK組立11.5%、親球変換7.8%、
motion validator5.4%だった。branch-missサンプルの22.4%が区間証明、5.5%が
motion validatorに帰属した。区間証明のAABB比較・sphereループ周辺が目立った。
通常のbranch-misses samplingにはskidがあるため、特定命令のサンプル数を
その分岐固有の正確なmiss数とは扱っていない。

AMDの `ex_ret_brn_ind_misp` も数えたところ、間接分岐のmissは全体の約4.7%。
この診断では6 counterを同時測定したため約83%のmultiplexがあった。
最終比較は通常の4 counterに絞り、全測定で100% runningを確認した。

最終Releaseバイナリでは同じseed、3,000計画、9,982,975論理queryを3回ずつ測定。
順序は前→後、後→前、前→後。下表はcounter別の中央値で、Pythonのwarm計画ループを含む。

| counter | 前回 | 今回 | 変化 |
|---|---:|---:|---:|
| cycles:u | 11.526 G | 11.072 G | -3.95% |
| instructions:u | 36.051 G | 35.459 G | -1.64% |
| branches:u | 4.167 G | 4.075 G | -2.20% |
| branch-misses:u | 50.795 M | 49.667 M | -2.22% |

branch missの減少は小さい。3回目の計測では速度改善がほぼ出ず、missにも実行間変動があった。
したがって「予測精度向上で5.6%速くなった」と全改善を帰属させることはできない。
命令削減・条件判定の重複削減・callback削減を含めた効果として報告する。
3回分の生counterと計画時間を [perf記録](conditioning-perf-summary.json) に保存した。

## 採用した変更

1. **証明済み区間の周辺だけを走査する。**
   以前は区間が証明されるたびに、辺の全補間点について所属判定していた。
   今回は区間の両端から候補indexを計算し、その範囲だけを調べる。
   浮動小数点の丸めに備えて左右1 indexずつ余裕を取り、最後は元と同じ比較を行う。
   補間点や検査順は変えない。

2. **Boxの面比較をAABB判定と距離判定で共有する。**
   軸に沿ったBoxでは、面までの距離から、そのまま区間の棄却と余裕計算を行う。
   遠い面が見つかれば即終了する。角の距離だけが必要な段階では、小さな比較を
   branchlessなmaxと二乗距離にまとめる。
   近接して証明が必要な場合だけ平方根を計算する。
   全部を一律にbranchless化する方法は採用しなかった。
   回転Boxなどは従来の幾何判定を使う。

3. **区間半径が縮んだ時だけ膨張量を再計算する。**
   `speed * radius + error` をleaf/pairごとに再計算せず、区間半径変更時に更新する。
   現在の証明中だけで使うローカル値であり、過去の問い合わせの履歴は使わない。

4. **連続して省略したqueryの回数更新をまとめる。**
   以前は省略した補間点ごとに間接callbackを呼んでいた。
   次の実検査の直前または辺の終端にまとめて加算する。
   最初の衝突点で失敗する場合も、元と同じ論理query数になる。
   終了後のkinematic stateを復元する処理も維持した。

今回の21組（7環境×3 seed）では、前回版と今回版の
準備辺数・実certificate query数・証明数・省略query数もすべて一致した。
したがって、今回の主な効果は省略対象を増やしたことではなく、
同じ枝刈り結果に達するまでの処理を安くしたことにある。

## 採用しなかった案

- AABBの6比較を全てbranchless化：missは減るが、Fetchで約5–6%悪化。
  固定3,000計画の命令数は約36.0 Gから39.6 Gに増えた。早期終了の利点を失った。
- 薄い軸を先に判定し、両側の面を1つの分岐にまとめる：改善と悪化が混在。
- 辺全域で非衝突のgroup/障害物pairをmarkして除外：試行と管理費用を回収できなかった。
  証明を試す順番やfallbackでの再利用を変えても採用案より遅かった。
- sphereを空間的に近い順で検査：比較結果の並びをそろえる狙いだったが、
  index参照と証明失敗時の戻り方の費用が増え、改善が安定しなかった。
- 近似sin/cosの誤差を回転関節数だけで見積もる案：証明可能なqueryは少し増えたが、
  安定した計画時間の追加改善が小さく、今回は元の保守的な上界を維持した。

各試作のbinary、patch、開発用測定はローカル実験ディレクトリに残した。
最終ソースには上の棄却案を含めていない。

## 検証

- 全37,800計画成功。前回版・SIMD導入前版との各12,600組で経路SHA256と論理query数が一致。
- 7環境の独立220,500姿勢ラベルが一致。同じ集合のseedごとの再検査はunique数に含めない。
- 6環境、18,000 anchorと区間内の計143,704点検査で不一致0。
- 900本の衝突境界探索、39,600点検査で不一致0。
- SDF移動・回転、base変更、link追加、attachment、cloud/planar等のfallbackを確認。
- 新しいBox処理について、3直動関節のロボットを使い、面・辺・角の接触近傍
  6,000 anchorと区間内を計33,032点検査。不一致0。
- 長い辺、短い辺、box validator、generic constraint、shortcut、検査回数予算の
  各100計画で、success・経路・query数・最終関節値が一致。
- index範囲の短縮は、ULP境界を含む500,000組について元の全走査と集合が完全一致。
- ONのRelease build、再現scriptの構文確認、`git diff --check`。
  整形後のbinaryも検証・perf測定済みbinaryとSHA256が一致した。
  全Python test suiteは実行していない。

この差分検証を形式検証と同一視はしない。適用範囲と数値誤差の扱いは
[元の区間pruning実験](scalar-pruning.md) を参照。

## 記録と再現

[集計とbinary hash](conditioning-summary.json)、[全計画raw](conditioning-raw.json.gz)、
[検証記録](conditioning-validation.json)、[perf記録](conditioning-perf-summary.json)。
作業treeは `/home/h-ishida/tmp/plainmp-scalar-pruning`、実験データは
`/home/h-ishida/tmp/plainmp_scalar_pruning_study`。
`build/interval` が前回版、`build/conditioning_final` が今回版、`build/scalar` がSIMD導入前。
これらは保存済みbinaryなので、作業treeの再ビルドでは変わらない。
`conditioning_stream` と `conditioning_final` は同じSHA256のbinary。
validationデータとモデル資産は従来の実験ディレクトリを参照する。

```sh
export PLAINMP_SCALAR_STUDY=/home/h-ishida/tmp/plainmp_scalar_pruning_study
cd /home/h-ishida/tmp/plainmp-scalar-pruning/docs/benchmarks/scalar-pruning-scripts
taskset -c 2 python3 screen.py --variants scalar interval conditioning_final \
  --tag conditioning-heldout --seeds 104729 130363 155921 --blocks 6 --n 100
python3 summarize_conditioning.py
python3 profile.py --variant interval --scene fetch_table --plans 3000 --kind stat
python3 profile.py --variant conditioning_final --scene fetch_table --plans 3000 --kind stat
python3 profile.py --variant profile_interval --scene fetch_table --seconds 8 \
  --kind record --record-event branch-misses:u

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 validate_certificates.py --variant conditioning_final --n 1000
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 check_interval_edges.py conditioning_final
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 check_box_certificates.py conditioning_final
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 check_planning_modes.py conditioning_final
g++-9 -O2 -std=c++17 check_interval_grid.cpp -o /tmp/plainmp-check-interval-grid
/tmp/plainmp-check-interval-grid
```

`check_planning_modes.py` は先に記録したscalar参照JSONを使う。
別ディレクトリで再現する場合は先に `python3 check_planning_modes.py scalar` を実行する。
計時とビルド・検証を同時実行しない。
