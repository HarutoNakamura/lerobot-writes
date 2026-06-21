# SO-101 で「指定した数字を書かせる」学習 — まとめ

ロボットアーム **SO-101** に、`write3` のように**指定した数字 (0〜9) を書かせる**ための模倣学習プロジェクト。
スパコン **Miyabi (GH200)** 上で、LeRobot の **SmolVLA** を自分たちの実演データで微調整（ファインチューニング）する。

---

## 1. ゴール

> 「数字 N を書いて」と**言語で指定**すると、ロボットがその数字を書く。

数字を指定できるかどうかは**方策(モデル)の種類**で決まる:

| モデル | 言語指示 `task` | 数字指定 |
|---|---|---|
| ACT / Diffusion Policy | 無視する | ❌ できない |
| **SmolVLA / π0 (VLA)** | 読む | ✅ できる |

→ 本プロジェクトは **SmolVLA** を使用。

---

## 2. データ

- **使用データ**: [`k1000dai/so101-write`](https://huggingface.co/datasets/k1000dai/so101-write)
  - 形式: **LeRobotDataset v3.0**
  - 規模: **100 エピソード / 41,995 フレーム / 30 fps**
  - 元データ: 数字ごとの [`yen-0/so101-write-0`](https://huggingface.co/datasets/yen-0/so101-write-0) 〜 `-9` を1つに統合したもの
- **ロボット**: SO-101 (`so_follower`)、**6 自由度**
  `shoulder_pan / shoulder_lift / elbow_flex / wrist_flex / wrist_roll / gripper`
- **カメラ 2 台**:
  - `observation.images.top` (480×848) … 俯瞰
  - `observation.images.wrist` (480×640) … 手首
- **タスクラベル**（数字の指定方法）: `task_index 0→"write0"` … `9→"write9"`
  各エピソードに「どの数字か」が `write0`〜`write9` の文字列として付いている。**ここが言語条件付けの肝**。

> 補足: [`k1000dai/number_images`](https://huggingface.co/datasets/k1000dai/number_images) は数字の参考写真10枚。
> 言語指定で書かせる本手法では**使わない**（画像をゴールにする別方式を取る場合のみ利用）。

---

## 3. 前処理（モデルをデータに合わせる）

データは既に学習可能な形（変換不要）。唯一の調整は**カメラ枚数の整合**:

- `lerobot/smolvla_base` は `config.json` が **3 カメラ固定**（`camera1/2/3`）。
- 本データは **2 カメラ**（`top` / `wrist`）。
- `--policy.path` で base を読むと、この 3 カメラ設定がデータで上書きされず**不一致エラー**になる。

**対処**: base をコピーし、`config.json` の `input_features` を `top`/`wrist` の2カメラに書き換えた
**2カメラ版 (`smolvla_base_2cam`)** を作る。
カメラ枚数は画像トークン数を変えるだけで重みの形状に影響しないため、**事前学習の重みはそのまま使える**。
（スクリプト: `01_make_smolvla_2cam.sh`）

---

## 4. モデル

- **SmolVLA**（Vision-Language-Action モデル）
  - VLM バックボーン: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
  - action expert（行動生成部）: フローマッチングで学習
  - **言語 `task` を読む**ので、`write3` の指定で「3」を書ける
- 学習方法: `lerobot/smolvla_base`（ロボット事前学習済み）の **2カメラ版から微調整**
  → 少数(100エピソード)データでも有利

---

## 5. 学習設定

```
lerobot-train
  --dataset.repo_id = k1000dai/so101-write
  --policy.path     = <2カメラ版 smolvla_base>   # 種別も自動判定（--policy.type は併用不可）
  --batch_size      = 4
  --steps           = 20000     # 動作確認は 500
  --save_freq       = 2000
  --wandb.mode      = offline
```

- 損失: ノイズ除去のフローマッチング損失（行動チャンクを生成）
- 速度: GH200 で **約 5 step/s** → 20k ステップ ≈ **約1.1時間**
- 出力: `$OUTPUT_DIR/smolvla_write/checkpoints/last/pretrained_model/`

---

## 6. 実行環境（Miyabi / HPC）

- **Miyabi-G**（GH200 GPU ノード）、コンテナは Apptainer（共有 `lerobot-v0.5.1.sif`）
- スケジューラ **PBS**（`qsub` で投入、`qstat` で確認、`qdel` で削除）
  - project 指定は `-W group_list=gw13`、キューは `debug-g`(確認) / `short-g`(本番)
  - バッチ投入はログアウトしても継続
- 重要な環境設定（ハマり所）:
  - コンテナ内 `/work` は読取専用 → 書込先は `--bind "$SHARED_DIR"` で渡す
  - 自分の書込領域は `$USER_DIR = /work/gw13/share/handson/<user>`
  - HF キャッシュは `HF_HOME=$USER_DIR/hf_home` に上書きして事前DL
  - 不足パッケージ（`num2words`）は `pip install --target=$USER_DIR/pylibs` → `PYTHONPATH` で読ませる
  - W&B は `--wandb.mode=offline`（キー不要、後で `wandb sync`）

---

## 7. 推論（学習後）

学習時と**同じ文字列**を `task` に渡すと、その数字を書く:

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained(".../checkpoints/last/pretrained_model")
batch = {
    "observation.images.top":   img_top,
    "observation.images.wrist": img_wrist,
    "observation.state":        state,    # (1,6)
    "task":                     "write3", # ← write0〜write9 で書く数字を指定
}
action = policy.select_action(batch)      # (1,6) 次の関節指令
```

実機 SO-101 では、カメラ取得 → `select_action` → 関節指令送信 のループを回す。

---

## 8. ファイル一覧（このフォルダ）

| ファイル | 役割 | 実行場所 |
|---|---|---|
| `00_download.sh` | データ + SmolVLAベース + num2words を取得 | ログインノード |
| `01_make_smolvla_2cam.sh` | base を 2カメラ版に作り替え | ログインノード |
| `smolvla_write_smoke.pbs` | 動作確認（500ステップ） | `qsub` |
| `smolvla_write.pbs` | 本番学習（20000ステップ） | `qsub` |
| `README_ja.md` | このまとめ | — |

### 実行順
```bash
cd /work/gw13/$USER/lerobot-handson
bash 00_download.sh
bash 01_make_smolvla_2cam.sh
qsub smolvla_write_smoke.pbs   # 確認
qsub smolvla_write.pbs         # 本番
```
