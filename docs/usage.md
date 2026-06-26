# 基本的な使用方法

この文書は、セットアップ完了後に Panodeon を使って全天球動画から COLMAP 用データを作り、復元を実行する手順をまとめたものである。セットアップは [setup.md](setup.md) を参照。

## 全体の流れ

1. `Sampler` で全天球動画から復元に使うフレームを抽出する。
2. `Mask` で人物検出・セグメンテーションを実行し、全天球画像のマスクを作る。必要なら手動で修正する。
3. `Mask` から COLMAP 用にエクスポートする。ここで全天球画像とマスクが 12 枚の透視投影画像・マスクへ変換される。
4. `COLMAP` で feature extraction、matching、mapping を実行する。

主な出力先は以下である。

- Sampler 出力: `<output>/frames/`
- マスク: `<project>/masks/`
- COLMAP 入力: `<project>/exports/images/`, `<project>/exports/masks/`, `<project>/exports/rig_config.json`
- COLMAP 結果: `<project>/exports/database.db`, `<project>/exports/sparse/`

## 起動

通常起動:

```powershell
.\scripts\run_app.ps1
```

手動起動:

```powershell
.\.venv\Scripts\panodeon
```

## 1. フレーム抽出

`Sampler` タブを使う。長い全天球動画から、復元に必要なフレームだけを抽出する。内部では縮小版の動画を作成し、stella_vslam でカメラ軌跡を推定し、farthest point sampling と ORB 特徴量による接続を基準として採用フレームを選定する。

手順:

1. `Video` に全天球動画を指定する。
2. `Output` を指定する。未入力なら動画名から `<video>_frames` が自動入力される。
3. 必要なら `Frame Skip`、`Image Format`、`JPEG Quality` を調整する。
4. `SLAM Exe` と `ORB Vocab` がセットアップ済みのパスを指していることを確認する。
5. 挙動を細かく制御したい場合は `Advanced` の `Camera Config` / `Sampler Config` で設定ファイルを上書きする（後述）。
6. `Run Sampler` を押す。

設定ファイルの上書き:

`Advanced` には、内部で自動生成される設定ファイルをユーザー指定のファイルで差し替えるための項目が 2 つある。どちらも未入力なら自動生成され、通常は触らなくてよい。

- `Camera Config`: stella_vslam に渡すカメラ設定（`.yaml`）。未入力なら全天球向けの設定が動画の fps に合わせて自動生成される。
- `Sampler Config`: フレーム選定の設定（`.json`）。未入力なら初期値が使われる。指定したファイルに書かれたキーだけが初期値を上書きする。

実際に使われたフレーム選定設定は、実行ごとに `sampled/run_config.resolved.json` に書き出される。これをコピーして編集すると上書き用ファイルを作りやすい。各キーの意味や書式は [config.md](config.md) を参照。

出力:

```text
<output>/
  proxy/
  stella/
  sampled/
  frames/
```

`frames/` が次の工程で使うプロジェクトフォルダである。Sampler 完了後は `frames/` が自動で読み込まれる。`Show Trajectory` は `sampled/trajectory_visualization.html` を開く。

再実行:

- 既に完了したステージはスキップされる。
- `stella/trajectory.csv` が存在する場合、stella_vslam を再実行せずに選択・抽出だけ実行できる。
- 最初からやり直す場合は、対象の `<output>/` を別名にするか、不要な出力を削除してから実行する。

## 2. マスク作成

`Mask` タブを使う。ここでは全天球画像そのものに対して人物検出・セグメンテーションを行い、マスク画像を作る。

手順:

1. `Open Folder` で `<output>/frames/` を開く。Sampler 完了直後は自動で開かれる。
2. `ONNX Model` で人物検出モデルを選ぶ。モデルを追加した場合は `Refresh` を押す。
3. `EP Provider` で推論実行環境を選ぶ。CUDA / DirectML がなければ CPU を使う。
4. `Mode` を選ぶ。通常は `both`（次の2手法の結果を合成したもの）。高速に試す場合は `direct` （画像から直接人物検出）、歪みが気になる場合は `cubemap` （一度キューブマップに投影してから人物検出）を使う。必要なら `Conf Score %` と `Mask Thresh %` を調整する。検出漏れが多い場合は下げ、誤検出が多い場合は上げる。
5. `Generate Select` （選択画像のみ出力）または `Generate All` （全画像を出力）を押す。

出力:

- 確定マスク: `<project>/masks/.../*.mask.png`
- direct 結果: `<project>/masks/.../*.mask.direct.png`
- cubemap 結果: `<project>/masks/.../*.mask.cubemap.png`

マスク表示:

- 赤いオーバーレイがマスク領域である。
- `Red Overlay` で表示濃度を変更する。

手動編集:

マスク画像は、手動で編集も可能である。

- `Brush Size` でブラシサイズを変える。
- 通常ブラシはマスクを追加する。
- `Eraser Mode` を有効にするとマスクを削る。
- `Undo` / `Redo` が使える。
- 編集後は `Save Mask` で保存する。

## 3. COLMAP 用データ出力

`Mask` タブの `Export Select` / `Export All` を使う。この段階で、全天球画像と全天球マスクを 12 枚の透視投影画像・マスクへ変換する。

手順:

1. `Tile Size` を確認する。初期値は `3072` である。
2. `FOV deg` を確認する。初期値は `90` である。
3. 一部だけ出す場合は画像リストで対象を選び `Export Select` を押す。
4. 全画像を出す場合は `Export All` を押す。

出力:

```text
<project>/exports/
  images/
  masks/
  rig_config.json
  README_colmap.txt
```

仕様:

- パノラマ 1 枚につき 12 枚の透視投影画像を作る。
- yaw は 4 方向、pitch は `-35`, `0`, `35` である。
- `masks/` は COLMAP feature mask である。黒ピクセルは特徴抽出から除外される。
- マスクには人物領域と、各仮想カメラの担当外領域が含まれる。

## 4. COLMAP 実行

`COLMAP` タブを使う。対象は現在のプロジェクトの `exports/` である。

手順:

1. `Executable` に `colmap` または `colmap.exe` のフルパスを指定する。
2. `Matcher` を選ぶ。通常は `sequential`。隣接関係を明示したい場合は `pairs`。
3. `Sparse Mapper` を選ぶ。通常は `mapper`。
4. 途中結果を再利用する場合は `Skip completed steps` を有効にする。
5. 既存の `database.db` や `sparse/` を消して作り直す場合は `Overwrite outputs` を有効にする。
6. GPU を使う場合は `Use GPU if available` を有効にする。GPU 番号は `GPU Index` で指定する。`-1` は COLMAP に任せる。
7. 必要なら `Rig bundle adjustment`、`Dense reconstruction`、`Mapper snapshots` を有効にする。
8. `Run COLMAP` を押す。

主な出力:

- `database.db`: feature / matching database
- `sparse/`: sparse reconstruction
- `sparse_rig_ba/`: rig bundle adjustment 後の model
- `dense/fused.ply`: dense reconstruction の点群
- `snapshots/`: mapper snapshots

状態確認:

- `COLMAP Status` は export、feature、rig、match、sparse、dense の完了状態を表示する。
- `Registered` は登録済み画像数である。
- `Resume From` は次に実行されるステージである。
- 実行ログは `Log` に表示される。
