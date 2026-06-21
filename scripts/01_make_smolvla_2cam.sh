#!/bin/bash
# === ②  smolvla_base を「2カメラ版(top/wrist)」に作り替える（ログインノードでOK）===
# smolvla_base は camera1/2/3 の3カメラ固定。あなたのデータは top/wrist の2カメラ。
# 事前学習の重みは残したまま、config.json の入力カメラだけを top/wrist に書き換える。
set -euo pipefail

cd /work/gw13/$USER/lerobot-handson
source config.env
module load apptainer/1.3.5

export HF_HOME="${USER_DIR}/hf_home"
export APPTAINERENV_HF_HOME="$HF_HOME"
DST="${USER_DIR}/smolvla_base_2cam"          # 書き換え後モデルの置き場
export DST

apptainer exec --bind "$SHARED_DIR" "$APPTAINER_IMAGE" python - <<'PY'
import os, glob, json, shutil

hf_home = os.environ["HF_HOME"]
dst     = os.environ["DST"]

# HFキャッシュ内の smolvla_base スナップショットを探す
cand = glob.glob(os.path.join(hf_home, "hub", "models--lerobot--smolvla_base", "snapshots", "*"))
assert cand, "smolvla_base が見つからない。先に 00_download.sh を実行してください"
src = cand[0]
print("source snapshot:", src)

# まるごとコピー（シンボリックリンクは実体化）
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst, symlinks=False)

# config.json の input_features を top/wrist の2カメラに書き換える
cfg_path = os.path.join(dst, "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)

infeat = cfg["input_features"]
# 既存のカメラ定義をテンプレートとして1つ取得（shape/type を流用）
cam_tmpl = infeat["observation.images.camera1"]

# camera1/2/3 を削除し、top/wrist を追加（state はそのまま残す）
for k in ["observation.images.camera1",
          "observation.images.camera2",
          "observation.images.camera3"]:
    infeat.pop(k, None)
infeat["observation.images.top"]   = dict(cam_tmpl)
infeat["observation.images.wrist"] = dict(cam_tmpl)

cfg["input_features"] = infeat
cfg["empty_cameras"]  = 0

with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)

print("written:", cfg_path)
print("input_features keys ->", list(infeat.keys()))
PY

echo "done: 2-camera base at $DST"
