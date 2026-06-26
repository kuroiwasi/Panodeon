# セットアップ

この文書は、Panodeon をローカルで実行するためのセットアップ手順である。

## Python 環境

通常セットアップは以下である。

```powershell
.\scripts\setup_env.ps1
```

起動:

```powershell
.\scripts\run_app.ps1
```

Windows がダウンロード済みスクリプトをブロックする場合は、内容を確認したうえで以下を実行する。

```powershell
Get-ChildItem scripts -Filter *.ps1 | Unblock-File
```

手動セットアップ:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e .[dev,inference]
.\.venv\Scripts\panodeon
```

## GPU 推論

GPU 推論を使う場合は、通常の `scripts\setup_env.ps1` の代わりに以下のどちらか 1 つを実行する。

- NVIDIA CUDA: `scripts\setup_env_cuda.ps1`
- NVIDIA / AMD DirectML: `scripts\setup_env_directml.ps1`

ONNX Runtime は 1 つの仮想環境に 1 種類だけ入れる。CUDA 用スクリプトは `onnxruntime-gpu` と CUDA / cuDNN runtime package を入れる。DirectML 用スクリプトは、Windows 上の対応 NVIDIA / AMD GPU 向けに `onnxruntime-directml` を入れる。

## モデルファイル

人物検出・セグメンテーション用の DEIMv2 Wholebody49 resources を `third_party\models` に取得する。

```powershell
.\scripts\setup_model.ps1 -OutputDir third_party\models
```

取得後、アプリの `ONNX Model` で対象の `.onnx` ファイルを選択する。

## COLMAP

COLMAP 4.0.4 CUDA 版を `third_party\colmap` に取得する。

```powershell
.\scripts\setup_colmap.ps1
```

CUDA なし版を使う場合:

```powershell
.\scripts\setup_colmap.ps1 -NoCuda
```

主なオプション:

- `-NoCuda`: CUDA なし版を取得する。
- `-Force`: 既存ファイルがあっても再取得する。
- `-ColmapVersion 4.0.4`: 取得する COLMAP release を指定する。

## stella_vslam

Sampler は stella_vslam を使って全天球動画のカメラ軌跡を推定する。Windows では事前に Git、CMake、Visual Studio 2022 C++ Build Tools が必要である。

`run_video_slam.exe` と `orb_vocab.fbow` は以下で用意する。

```powershell
.\scripts\setup_stella.ps1
```

このスクリプトは pinned revision の stella_vslam / stella_vslam_examples / vcpkg を取得し、`patches/` の Windows patch を適用してビルドする。

出力:

- `third_party/runtime/run_video_slam.exe`
- `third_party/FBoW_orb_vocab/orb_vocab.fbow`
- `third_party/runtime/build_manifest.json`

## 管理外ファイル

`third_party/`、`.venv/`、モデルファイル、COLMAP 実行結果は git 管理外である。
