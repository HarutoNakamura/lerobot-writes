# 進行度リアルタイム表示 — どのPCでも動かすためのパッケージ

SO-101 に数字を書かせながら「いま何%まで書き上がったか」を進捗バー表示する推論一式。
学習済みモデルは Hugging Face にあるので、**このフォルダと Python 環境だけあればどのPCでも動く**。

| モデル | HF リポジトリ | 方式 |
|---|---|---|
| 書字ポリシー（ベースライン） | [`HarutoNakamura/lerobot-write`](https://huggingface.co/HarutoNakamura/lerobot-write) | SmolVLA |
| 進行度推定（別建て） | [`HarutoNakamura/so101-write-progress`](https://huggingface.co/HarutoNakamura/so101-write-progress) | ResNet18 回帰 |
| 書字+進行度（統合・20k step） | [`HarutoNakamura/lerobot-write-prog`](https://huggingface.co/HarutoNakamura/lerobot-write-prog) | SmolVLA 7次元action |
| 書字+進行度（統合・**60k step**） | [`HarutoNakamura/lerobot-write-prog-60k`](https://huggingface.co/HarutoNakamura/lerobot-write-prog-60k) | 同上・学習量3倍（progress MAE 0.027） |

## セットアップ（新しいPCでやること）

```bash
git clone https://github.com/HarutoNakamura/lerobot-writes.git
cd lerobot-writes/inference
pip install -r requirements.txt        # Python 3.10+ / venv や conda 推奨
```

モデルは初回実行時に HF から自動DLされる（統合版 ~865MB）。
別建て版の重みだけ先に落とす場合:

```bash
hf download HarutoNakamura/so101-write-progress progress_net.pt --local-dir .
```

## デバイス設定（コード書き換え不要）

実機のポートとカメラ番号は**引数で指定**する:

- `--port` … SO-101 のシリアルポート。Linux: `/dev/ttyACM0`、Windows: `COM5` など
- `--cam_top` / `--cam_wrist` … カメラ番号（既定 0 / 1）

カメラの視点（俯瞰 top / 手首 wrist）は**学習データ収集時と同じ配置**にすること。

解像度や fps まで変えたい場合のみ `realtime_progress.py` の `make_robot()` を編集する
（統合版 `realtime_smolvla_prog.py` もこの関数を使う）:

```python
def make_robot(port="/dev/ttyACM0", cam_top=0, cam_wrist=1):
    cfg = SO101FollowerConfig(
        port=port,   # ← --port 引数がここに入る
        cameras={
            "top":   OpenCVCameraConfig(index_or_path=cam_top,   width=848, height=480, fps=30),
            "wrist": OpenCVCameraConfig(index_or_path=cam_wrist, width=640, height=480, fps=30),
        },
    )
```

## 動かし方

### 実機なしの動作確認

```bash
# Webカメラだけで進捗バーの反応を見る（手書きで数字を書いてみる。r=リセット, q=終了）
python realtime_progress.py camera --ckpt progress_net.pt --digit 3 --cam 0

# 学習データの再生で確認（初回に k1000dai/so101-write を自動DL・数GB）
python realtime_progress.py dataset --ckpt progress_net.pt --episode 9
```

### 実機で書かせながら表示

```bash
# ① 別建て版: ベースライン SmolVLA が書き、ResNet18 が紙面から進行度を推定
python realtime_progress.py robot --ckpt progress_net.pt --digit 3 \
    --policy HarutoNakamura/lerobot-write --port COM5 --cam_top 0 --cam_wrist 1

# ② 統合版: 1回の推論で関節指令と進行度が同時に出る（GPU推奨）
python realtime_smolvla_prog.py robot --ckpt HarutoNakamura/lerobot-write-prog \
    --digit 3 --port COM5 --cam_top 0 --cam_wrist 1

# ②' 学習量3倍の 60k 版に切り替える場合は --ckpt を変えるだけ
python realtime_smolvla_prog.py robot --ckpt HarutoNakamura/lerobot-write-prog-60k \
    --digit 3 --port COM5 --cam_top 0 --cam_wrist 1

# ②の実機なし動作確認（派生データセット HarutoNakamura/so101-write-prog を上げてある場合）
python realtime_smolvla_prog.py dataset --ckpt HarutoNakamura/lerobot-write-prog \
    --repo_id HarutoNakamura/so101-write-prog --episode 9
```

- 緑バー = 平滑化済み進行度（EMA+単調化、戻らない）、`raw ○%` = 生の推定値
- 1枚書かせるごとにスクリプトを起動し直す（起動時にポリシーと進行度がリセットされる）
- ProgressNet(①の進行度側) は CPU で十分。SmolVLA を回す部分は GPU 推奨

## 自作コードへの組み込み

```python
# ① 別建て版: 3行で任意のループに足せる
from progress_model import ProgressEstimator, draw_progress_bar
est = ProgressEstimator("progress_net.pt", device="cuda")
est.reset()                                    # エピソード開始時
p = est.estimate(img_top, digit)               # ループ内。(3,H,W) float か (H,W,3) uint8
draw_progress_bar(frame_bgr, est.smoothed, digit=digit, raw=p)

# ② 統合版: action の7次元目が進行度
from eval_smolvla_prog import load_policy, predict_step
policy, preproc, postproc = load_policy("HarutoNakamura/lerobot-write-prog", "cuda")
joints, p = predict_step(policy, preproc, postproc,
                         img_top, img_wrist, state, "write3", "cuda")
```

> **注意（lerobot v0.5系）**: 正規化はポリシー本体ではなく pre/post processor 側にある。
> `select_action` を生で呼ばず、必ず `load_policy` が返す preprocessor / postprocessor を
> 前後に挟むこと（`predict_step` は挟んである）。

## 学習・評価をやり直す場合

`train_progress.py`（別建て版の学習）と `eval_progress.py` / `eval_smolvla_prog.py`（評価）も
このフォルダに同梱。Miyabi での学習手順・統合版のデータセット作成は
別リポジトリ `monopro/miyabi/README_progress_ja.md`, `README_smolvla_prog_ja.md` を参照。
