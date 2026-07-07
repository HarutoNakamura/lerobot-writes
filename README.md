# lerobot-writes — SO-101 に数字を書かせる SmolVLA

ロボットアーム **SO-101** に、`write3` のように**指定した数字 (0〜9) を書かせる**ための
言語条件付き模倣学習プロジェクト。LeRobot の **SmolVLA** を自前の実演データで微調整した学習済みモデル一式。

> 詳細な解説（データ・前処理・モデル・学習・実行環境）は [`scripts/README_ja.md`](scripts/README_ja.md) を参照。

## 中身

| パス | 内容 |
|---|---|
| `lerobot-write/` | ベースライン学習のチェックポイント（config・前後処理。**重みは下記HFを参照**） |
| `lerobot-write-aug/` | 画像オーグメンテーション有りで学習したチェックポイント |
| `scripts/` | Miyabi(GH200) 上での学習スクリプト一式（DL / 2カメラ化 / 学習 PBS）と日本語まとめ |
| `inference/` | **進行度リアルタイム表示**の推論一式（どのPCでも動く配布用。[`inference/README_ja.md`](inference/README_ja.md) 参照） |

`model.safetensors`（各 ~865MB）は GitHub の 100MB 制限を超えるため**リポジトリには含めず**、
Hugging Face Hub で配布しています:

- ベースライン: <https://huggingface.co/HarutoNakamura/lerobot-write>
- aug 版:        <https://huggingface.co/HarutoNakamura/lerobot-write-aug>
- 進行度推定（別建て ResNet18）: <https://huggingface.co/HarutoNakamura/so101-write-progress>
- 書字+進行度（SmolVLA 統合・7次元action）: <https://huggingface.co/HarutoNakamura/lerobot-write-prog>

## 概要

- **データ**: [`k1000dai/so101-write`](https://huggingface.co/datasets/k1000dai/so101-write)
  （LeRobotDataset v3.0 / 100エピソード / 2カメラ `top`+`wrist` / 6自由度 / タスク `write0`〜`write9`）
- **モデル**: SmolVLA（VLM バックボーン `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` + action expert）
  を `lerobot/smolvla_base` から微調整。`task` の言語指示を読むので数字を指定できる。
- **前処理の要点**: smolvla_base は 3カメラ固定なので、`config.json` を書き換えて
  **2カメラ版 (`top`/`wrist`)** にしてから微調整（事前学習重みはそのまま利用）。
- **学習**: Miyabi-G (GH200) / Apptainer / PBS、`lerobot-train` で 20,000 ステップ。

## 推論（指定した数字を書かせる）

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# HF から取得（または手元の checkpoints/last/pretrained_model を指定）
policy = SmolVLAPolicy.from_pretrained("HarutoNakamura/lerobot-write")
policy.eval()

batch = {
    "observation.images.top":   img_top,    # (1,3,H,W)
    "observation.images.wrist": img_wrist,
    "observation.state":        state,      # (1,6)
    "task":                     "write3",   # ← write0〜write9 で書く数字を指定
}
action = policy.select_action(batch)        # (1,6) 次の関節指令
```
