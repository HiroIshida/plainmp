# SIMD導入前を基準にしたスカラー区間pruning実験

## 結果

SIMD導入直前の `fe44f5ffccf38ab3a1f121723aec42893f05667c` を基準に、
複数姿勢をSIMDレーンに詰めず、関節運動の上界と障害物までの余裕から
非衝突区間を求める実装を追加した。別seedの7環境で **1.31–1.67倍、
環境別倍率の幾何平均1.48倍**。7環境中3環境で1.5倍を超えた。
全環境で1.5倍は達成していない。

目標は「SIMD導入と同程度の効果をスカラーの工夫で得ること」。
現在のSIMD版をさらに1.5倍上回ることではない。
ここでSIMDなしとは明示的な複数姿勢バッチ処理を追加しない意味であり、
既存のコンパイラが発行するすべてのSSE命令を排除する意味ではない。
ビルドは従来どおり `EIGEN_DONT_VECTORIZE`。

作業場所 `/home/h-ishida/tmp/plainmp-scalar-pruning`、branch `research/scalar-pruning`。
初回SIMD commit `5fb9380` の親から作成した。既存の作業treeは変更していない。
実験機能は `PLAINMP_ENABLE_INTERVAL_PRUNING=ON` で有効化する。CMake既定値はOFF。
この作業treeの `build` はONでビルド済み。

追加の条件判定・管理処理の最適化は
[次の実験記録](scalar-pruning-conditioning.md)にまとめた。以下の数値は初回実装時の記録。

## 計画時間

2026-09-24、Ryzen 7 7840HS、CPU 2、GCC 9.4、Release/O3/LTO。
同じPythonコード、URDF、球配置、self collision pair、関節範囲を使用。
RRTConnect range 2、Euclidean解像度1/32、簡略化なし。
各環境3 seed × 600計画、30回warmup後、100計画ごとに実行順を交代した。

| 環境 | 元のscalar ms | 区間pruning ms | 計画時間倍率 | seed別倍率範囲 | Pythonを含む倍率 | 姿勢検査省略率 |
|---|---:|---:|---:|---:|---:|---:|
| Panda | 0.3261 | 0.1973 | **1.67×** | 1.61–1.69× | 1.61× | 73.8% |
| Panda + ceiling | 1.1891 | 0.8181 | **1.45×** | 1.43–1.46× | 1.43× | 67.7% |
| Fetch table | 1.0447 | 0.6641 | **1.57×** | 1.57–1.60× | 1.55× | 67.1% |
| Fetch spheres × 4 | 0.4381 | 0.3077 | **1.47×** | 1.42–1.49× | 1.42× | 66.2% |
| Fetch spheres × 9 | 0.4969 | 0.3740 | **1.31×** | 1.30–1.35× | 1.32× | 62.0% |
| Panda tilted boxes | 0.3204 | 0.2070 | **1.51×** | 1.50–1.55× | 1.46× | 73.9% |
| Fetch tilted table | 0.8991 | 0.6347 | **1.42×** | 1.41–1.47× | 1.41× | 64.7% |

時間はseed別中央値の中央値。倍率は各seedで対応する中央値の比を取り、その中央値。
そのため表示時間同士を割った値と倍率列は厳密には一致しない。
省略率は論理上の姿勢検査のうち、非衝突証明で実行しなかった割合。
最後の公開kinematic stateを合わせるため、必要時には関節更新だけを行う。

調整にはseed `32452843` の4環境を使用し、区間半径上限を全環境共通で
**6 resolution steps** に固定した。
最終評価は `67867967 / 67867979 / 67867987` の7環境。
これらのseedは最適化前の基準測定にも使用したが、今回の実装・上限値の調整には使用していない。
新実装をOFFにした同一ソースの対照バイナリも交代で測定し、
OFF→ONの環境別倍率は1.32–1.65倍。バイナリ配置の差だけでは説明できない。

CPU周波数・ASLR・背景負荷は固定していない。特にOFF版のPanda + ceiling、
seed 67867987は他の2 seedに対して相対的に遅い結果が出た。
この値を除外せずrawに保存し、3 seedの中央値と範囲を報告した。
厳密な数%の差より、問題依存で約1.3–1.7倍という規模を読むべき測定である。

## 何を省いたか

### URDFから移動量の上界を作る

各sphere group gについて、制御関節jから球中心までの最大レバー長 `L[g,j]` を求める。
回転関節では下流の固定オフセット・prismaticの可動範囲・球のlink内位置を足して
上から抑え、直動関節では軸の長さを使う。これはロボット構造から作る値であり、
過去の衝突問い合わせを学習した値ではない。

辺 `q(t)=q0+t(q1-q0)` に対して、groupの移動速度上界は

```
v[g] = Σ_j L[g,j] * abs(q1[j] - q0[j])
```

自己衝突pairでは、両方のリンクを共通に動かす祖先関節は相対距離を変えないので、
片側だけを動かす関節を足す。
辺を検査する直前にこれらを計算し、その辺の検査中に使う。

### 余裕から区間を証明する

姿勢q(t)で障害物までの余裕dがあれば、概略 `|Δt| < d/v` の近傍も非衝突。
実装は球半径を `v*r + 誤差上界` だけ膨張させて、親球、AABB、leafの順に判定する。
上限半径rで証明できないleafでは余裕を求めてrを縮める。
Boxの面やCylinderの端面は平方根なしで処理し、球間距離も必要になるまで二乗のまま扱う。

区間証明が成立すれば、その区間に入る**元の補間点**の検査を省く。
FK・親球の変換・leaf球の変換・collision判定をまとめて省ける。
証明できない場合は、未解決のsphere/pairの位置から元の点検査に戻る。
先に非衝突を証明したpairを最初から調べ直さないことも速度に効いた。

元の補間点集合と検査順序を維持し、skipも論理的な `n_call` に加算する。
これによりplannerの予算判定や同じseedの探索経路を維持する。
履歴に依存するcacheや近似classifierは追加していない。

### 数値誤差と適用範囲

既存FKは近似sin/cosを使うので、理想的な回転運動の上界だけでは不足する。
半角のTaylor多項式誤差から関節quaternionの誤差を4e-5以下と見積もり、
深さDに対して `e=(1.00004)^D-1`、
回転行列誤差上界 `4e+2e²` と根からの到達長を使ってgroupごとの位置誤差Eを求める。
環境衝突ではanchorと対象点の両方の誤差として2E、自己衝突では2(Ea+Eb)を引く。
浮動小数点の余裕も追加する。この上界の導出と差分検証は行ったが、
浮動小数点演算全体を形式検証した実装ではない。

今回の高速経路は固定base、組み込みBox/Sphere/Cylinder/Ground、
元の検査点数4–128、通常のURDF構造を対象とする。
点群SDF、移動base、その他の制約、長すぎる辺などは元の処理に戻る。
旧Box predicateが別の扱いをする半径1e-6未満の球もfallbackする。
URDF可動範囲外のprismatic、非有限値、非正規化回転等は証明を行わない。
SDF交換・link追加を扱い、baseや未制御関節の変更は辺の準備時に反映する。
辺の検査中に別スレッドからrobot/SDFを変更する用途は対象外。

Pythonの `prepare_motion_certificate` / `check_motion_certificate` は研究用診断API。
qは直前にprepareした線分上に置き、その線分の検査中はbase・未制御関節・環境・構造を固定する。
返す半径は関節距離ではなく正規化されたtの半径。
通常の点問い合わせと勾配計算のAPIは変更していない。

## perfと棄却した案

元のFetch tableのwarm区間を `perf record` で採取したところ、exclusive cyclesは
`is_valid_dirty` 27.8%、FK組立18.0%、親球の変換13.7%、関節更新8.9%、
近似sin/cos 5.5%、leaf球変換3.6%だった。
leafの距離計算だけを減らしても総時間の1/3を削ることは難しい。

最終版を同じseed・3,000計画で `perf stat` 比較した。両者9,982,975論理query。
Pythonを含むwarm区間の測定で、結果は次のとおり。

| user-space counter | 元のscalar | 区間pruning | 変化 |
|---|---:|---:|---:|
| cycles | 19.16 G | 11.93 G | −37.7% |
| instructions | 68.85 G | 36.05 G | −47.6% |
| branches | 7.62 G | 4.17 G | −45.3% |
| branch misses | 34.66 M | 54.23 M | +56.5% |

分岐missは増えている。それでも不要な計算を大きく省き、総cyclesは減った。
この実験では「分岐を減らす」より「分岐に費用を払ってFKと検査を省く」効果が勝った。
これは同一CPUの今回の問題集合における観察である。

採用しなかった試作も保存した。

- leafごとの遅延xyz変換：Fetchで約5–8%悪化。別obstacle/self pairでの再計算が増えた。
- 分離しそうな座標成分だけ先に求める方法：追加分岐の費用が上回った。
- quaternionのゼロ成分を使う疎なFK演算：環境により改善と悪化が混在。
- URDFの下流subtreeを包む球：境界が緩く、約7–21%悪化。
- 固定半径4 stepsの区間証明：1.06–1.30倍。距離に応じて半径を縮める方が良かった。

SIMDでもマスクやlaneの再配置でpruningは実装できるので、
「SIMDには不可能」とは主張しない。
論文としては、robot構造・幾何学的余裕・証明の失敗時の戻り方が
どれだけ計算量を減らすかを切り分ける必要がある。
既存の距離に基づく区間検査に対する新規性は、この実験だけでは判定していない。

## 正しさの確認

- 3バイナリ合計37,800計画すべて成功。元版と区間版の12,600組で
  経路SHA256と論理検査回数が一致。OFF対照版も一致。
- 7環境、各31,500、合計220,500の独立姿勢ラベルが各バイナリで一致。
  同じ集合をseedごとにも再検査したが、unique数には重複を含めていない。
- 6環境・18,000 anchorから区間内を検証。計143,704点検査、不一致0。
- 衝突境界を二分探索した900線分について39,600点検査、不一致0。
- SDF移動・回転、辺間のbase変更、prepare後のlink追加と再prepare、
  attachment、点群・移動base・微小球のfallbackを確認。
- 長い辺、短い辺、box validator、generic constraint、shortcut、
  n_max_call=80の各100計画、計600組でsuccess・経路・回数・最終関節値が一致。
- ON/OFFの両方をビルド。最終ONの再ビルドは評価済みbinaryとSHA256が一致。
  `git diff --check` を実施。全Python test suiteは実行していない。

## 再現

[集計・binary hash](scalar-pruning-summary.json)、
[全37,800計画のraw](scalar-pruning-raw.json.gz)、
[区間・境界・fallback検証とperf](scalar-pruning-validation.json)。
[最適化前の参考測定](scalar-pruning-baseline-summary.json)も保持した。
現在のSIMDとの参考比較は目的の判定に使っていない。

実験ディレクトリは `/home/h-ishida/tmp/plainmp_scalar_pruning_study`。
`source` は作業tree、`build/scalar` は変更前の保存済みbinary、
`build/disabled` は同一ソースOFF、`build/interval` はON。
開発中の各binaryとpatchは同ディレクトリの `build` / `prototypes`、
query集合とログは `results` に保存。
再現scriptは [scalar-pruning-scripts](scalar-pruning-scripts/) にも収録した。
scriptの `PLAINMP_SCALAR_STUDY` はこのディレクトリ構成とvalidation_*.npzを持つ場所を指す。
モデル資産・既存のvalidationデータはこのreportに同梱していない。

```sh
cmake -S /home/h-ishida/tmp/plainmp-scalar-pruning \
  -B /home/h-ishida/tmp/plainmp-scalar-pruning/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-9 -DCMAKE_CXX_COMPILER=/usr/bin/g++-9 \
  -Dompl_DIR=/opt/ros/noetic/share/ompl/cmake \
  -DPLAINMP_ENABLE_INTERVAL_PRUNING=ON
cmake --build /home/h-ishida/tmp/plainmp-scalar-pruning/build -j4

export PLAINMP_SCALAR_STUDY=/home/h-ishida/tmp/plainmp_scalar_pruning_study
cd /home/h-ishida/tmp/plainmp-scalar-pruning/docs/benchmarks/scalar-pruning-scripts
taskset -c 2 python3 screen.py --variants scalar disabled interval \
  --tag heldout --seeds 67867967 67867979 67867987 --blocks 6 --n 100
python3 summarize_interval.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 validate_certificates.py --variant interval --n 1000
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 check_interval_edges.py interval
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 check_planning_modes.py scalar
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 check_planning_modes.py interval
python3 profile.py --variant scalar --scene fetch_table --plans 3000 --kind stat
python3 profile.py --variant interval --scene fetch_table --plans 3000 --kind stat
```

`screen.py` は差分をrawに記録し、`summarize_interval.py` が全63結果を読み、
path/call/label不一致がすべて0であることをassertする。
計時中はビルドや他のベンチマークを同時実行しない。
