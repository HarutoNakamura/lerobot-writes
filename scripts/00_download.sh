#!/bin/bash
# === ① ログインノードで実行（ネットあり）===
# データセットとSmolVLAベースモデルを $HF_HOME に事前ダウンロードする。
# 計算ノードはネット不通なので、必ずここで落としておく。
set -euo pipefail

cd /work/gw13/$USER/lerobot-handson
source config.env
module load apptainer/1.3.5

# 共有 $HF_HOME は読み取り専用なので、自分の書き込み可能領域（共有下のUSER_DIR）に上書き
export HF_HOME="${USER_DIR}/hf_home"
mkdir -p "$HF_HOME"
export APPTAINERENV_HF_HOME="$HF_HOME"   # コンテナ内にも確実に渡す
echo "HF_HOME=$HF_HOME"

# コンテナ内では /work はデフォルト読み取り専用。共有領域を書き込み可能でバインドする
# （USER_DIR は SHARED_DIR の下なので、これ1つで自分の書き込み先もカバーされる）
BIND="$SHARED_DIR"

# あなたの数字データ（write0〜write9 のタスク付き・変換済み）
apptainer exec --bind "$BIND" "$APPTAINER_IMAGE" hf download --repo-type dataset k1000dai/so101-write

# SmolVLA の事前学習済みベース
apptainer exec --bind "$BIND" "$APPTAINER_IMAGE" hf download lerobot/smolvla_base

# コンテナに無い追加パッケージを、書き込み可能領域に入れて PYTHONPATH で読ませる
# （SmolVLM プロセッサは num2words を要求する）
export PYLIBS="${USER_DIR}/pylibs"
mkdir -p "$PYLIBS"
apptainer exec --bind "$BIND" "$APPTAINER_IMAGE" \
  pip install --no-cache-dir --target="$PYLIBS" num2words

echo "done: dataset + base model + pylibs cached under $USER_DIR"
