# Panodeon

Panodeon は、360 度カメラで撮影した動画から、COLMAP で 3D 復元を行うためのツールである。

Panodeon を使うと、全天球動画から以下を得られる。

- COLMAP が扱いやすい透視投影画像セット
- 人物や不要領域を除外する feature mask
- 分割した視点を同一のリグとして扱う `rig_config.json`
- COLMAP の `database.db` と sparse model

## 処理の流れ

1. 全天球動画を用意する。
2. 動画から復元に使うフレームだけを自動抽出する。
3. 選んだ全天球画像に対して人物検出とセグメンテーションを行い、マスク画像を作る。必要なら手動で修正する。
4. 各全天球画像とマスクをそれぞれ 12 枚の透視投影画像・マスクに変換する。
5. 透視投影画像、透視投影マスク、仮想カメラ設定を COLMAP 用フォルダへ出力する。
6. COLMAP で feature extraction、matching、mapping を実行する。

## 出力

主な出力は以下である。

- 抽出フレーム: `<output>/frames/`
- マスク: `masks/`
- COLMAP 入力: `exports/images/`, `exports/masks/`, `exports/rig_config.json`
- COLMAP 結果: `exports/database.db`, `exports/sparse/`

## セットアップ

[docs/setup.md](docs/setup.md) を参照。

## 起動

```powershell
.\scripts\run_app.ps1
```

手動起動:

```powershell
.\.venv\Scripts\panodeon
```

インストールせずに起動:

```powershell
$env:PYTHONPATH="src"
python -m panodeon.app
```

## 使用方法

[docs/usage.md](docs/usage.md) を参照。

## ライセンス

本プロジェクトは MIT ライセンスで提供される。詳細は [LICENSE](LICENSE) を参照。

`patches/` 配下の第三者プロジェクト（stella_vslam など）には、それぞれ独自のライセンスが適用される。詳細は [patches/README.md](patches/README.md) を参照。
