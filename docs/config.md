# 設定ファイルの詳細

Sampler の `Advanced` には、内部で自動生成される設定ファイルをユーザー指定のファイルで上書きするための項目が 2 つある。どちらも未入力なら自動生成され、通常は触らなくてよい。挙動を細かく制御したい場合だけ使う。

- `Camera Config`: stella_vslam に渡すカメラ設定（`.yaml`）。
- `Sampler Config`: フレーム選定の設定（`.json`）。

この文書では、この 2 ファイルの中身を説明する。基本的な使い方は [usage.md](usage.md) を参照。

## Camera Config（stella_vslam カメラ設定 / YAML）

`Camera Config` を未入力で実行すると、全天球（equirectangular）向けの設定が動画の fps に合わせて自動生成され、stella_vslam に渡される。自前の設定を使いたい場合は、ここで `.yaml` を指定すると自動生成の代わりにそのファイルが使われる。

自動生成される内容は以下と同等である（`fps` は入力動画から決まる）。

```yaml
Camera:
  name: "360 Camera 2K"
  setup: "monocular"
  model: "equirectangular"
  fps: <動画の fps>
  cols: 1920
  rows: 960
  color_order: "BGR"

Preprocessing:
  min_size: 800

Feature:
  name: "default ORB feature extraction setting"
  scale_factor: 1.2
  num_levels: 8
  ini_fast_threshold: 20
  min_fast_threshold: 7

Mapping:
  backend: "g2o"
  baseline_dist_thr_ratio: 0.02
  redundant_obs_ratio_thr: 0.95
  num_covisibilities_for_landmark_generation: 20
  num_covisibilities_for_landmark_fusion: 20
  residual_deg_thr: 0.4

Tracking:
  backend: "g2o"

LoopDetector:
  backend: "g2o"
  enabled: true
  reject_by_graph_distance: true
  min_distance_on_graph: 50

GraphOptimizer:
  min_num_shared_lms: 200

GlobalOptimizer:
  thr_neighbor_keyframes: 100

System:
  map_format: "msgpack"
  num_grid_cols: 96
  num_grid_rows: 48
```

これは速度重視の初期設定である。書式と各キーの意味は stella_vslam の設定仕様に従う。トラッキングが安定しない、ループ検出を調整したい、解像度を変えたい、といった場合に上書きを使う。

## Sampler Config（フレーム選定設定 / JSON）

`Sampler Config` を未入力で実行すると、後述の初期値が使われる。`.json` を指定すると、その内容で初期値を上書きする。指定したファイルに書かれていないキーは初期値のままになる。

実際に使われた設定は、実行ごとに `sampled/run_config.resolved.json` として書き出される。これをコピーして編集すると、確実に同じ書式で上書き設定を作れる。

トップレベルの構造:

```json
{
  "schema_version": 1,
  "sampling": { ... },
  "continuity": {
    ...,
    "orb": { ... }
  },
  "trajectory": { ... }
}
```

`schema_version` は現在 `1`。これ以外の値を指定するとエラーになる。

### sampling — フレームの選び方

軌跡上からどのフレームを採用するかを決める。

| キー | 初期値 | 意味 |
| --- | --- | --- |
| `scale_mode` | `"robust_extent"` | 採用間隔の決め方。`metric`（実距離）、`robust_extent`（軌跡の広がりに対する比率）、`target_count`（目標枚数）のいずれか。 |
| `radius` | `null` | `metric` モードでの採用間隔（実距離）。`metric` では正の値が必須。 |
| `radius_ratio` | `0.1` | `robust_extent` モードでの採用間隔（軌跡の広がりに対する比率）。小さくするほど密に採用する。 |
| `target_count` | `null` | `target_count` モードでの目標採用枚数。`target_count` では正の値が必須。 |
| `force_first_frame` | `true` | 最初のフレームを必ず採用する。 |
| `force_segment_endpoints` | `false` | 各セグメントの端点を必ず採用する。 |
| `allow_weak_tracking` | `false` | トラッキングが弱いフレームも候補に含める。 |
| `distance_tolerance` | `1.0e-6` | 距離比較の許容誤差。`[0, 1)`。 |
| `tie_distance_ratio` | `1.0e-3` | 同距離とみなす比率のしきい値。`[0, 1)`。 |

採用枚数を直接決めたいときは `scale_mode` を `target_count` にして `target_count` を指定するのが分かりやすい。

### continuity — フレーム間の接続性

採用フレーム同士がつながっているか（特徴量を共有できるか）を評価し、つながりが弱い箇所に補助フレームを追加する。

| キー | 初期値 | 意味 |
| --- | --- | --- |
| `enabled` | `true` | 接続性評価を行うか。 |
| `max_path_gap_ratio` | `2.0` | これを超える間隔を「つながりが弱い」とみなす比率。 |
| `bridge_min_separation_ratios` | `[0.25, 0.1]` | 補助フレームを挿入する際の最小間隔の比率（段階的に適用）。 |
| `bridge_max_recursion_depth` | `12` | 補助フレーム挿入の再帰の深さ上限。 |

#### continuity.orb — ORB による接続性判定

接続性の判定に ORB 特徴量のマッチングを使う部分の設定。

| キー | 初期値 | 意味 |
| --- | --- | --- |
| `enabled` | `true` | ORB によるマッチングを使うか。 |
| `max_width` | `720` | マッチング時に縮小する最大幅（px）。 |
| `feature_count` | `1000` | 抽出する特徴点数。8 以上。 |
| `ratio_test` | `0.75` | Lowe の ratio test のしきい値。`(0, 1)`。 |
| `min_inliers` | `40` | マッチ成立に必要な最小インライア数。4 以上。 |
| `min_inlier_ratio` | `0.2` | 必要な最小インライア比率。`[0, 1]`。 |
| `ransac_threshold` | `2.0` | 平面マッチング時の RANSAC しきい値。正の値。 |
| `spherical` | `true` | 全天球の歪みを避けるため、複数の透視投影パッチへ展開してから ORB を行う。`false` にすると速いが極付近の精度は落ちる。 |
| `tangent_layout` | `"cubemap"` | `spherical` 時のパッチ配置。`cubemap`（6 面・重なりなし・最速）または `equatorial`（複数 pitch のヨー環・重なりあり）。 |
| `tangent_fov_deg` | `90.0` | パッチの画角（度）。`(0, 180)`。 |
| `tangent_size` | `512` | パッチの一辺の画素数。正の値。 |
| `tangent_yaw_count` | `6` | ヨー方向のパッチ数。正の値。 |
| `tangent_pitch_deg` | `[-30.0, 0.0, 30.0]` | パッチの pitch（度）。各要素は `(-90, 90)`。 |
| `spherical_ransac_threshold` | `0.01` | `spherical` 時の RANSAC しきい値。正の値。 |

初期値では `spherical` が有効で、極付近の歪みに強い。処理を軽くしたい場合は `false` にする。

### trajectory — 軌跡の前処理

| キー | 初期値 | 意味 |
| --- | --- | --- |
| `jitter_threshold` | `0.0` | これ未満の微小な移動をノイズとして無視するしきい値。`0` で無効。非負。 |

