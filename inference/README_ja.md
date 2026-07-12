# 進行度リアルタイム表示 — どのPCでも動かすためのパッケージ

SO-101 に数字を書かせながら「いま何%まで書き上がったか」を進捗バー表示する推論一式。
学習済みモデルは Hugging Face にあるので、**このフォルダと Python 環境だけあればどのPCでも動く**。

| モデル | HF リポジトリ | 方式 |
|---|---|---|
| 書字ポリシー（ベースライン） | [`HarutoNakamura/lerobot-write`](https://huggingface.co/HarutoNakamura/lerobot-write) | SmolVLA |
| 進行度推定（別建て） | [`HarutoNakamura/so101-write-progress`](https://huggingface.co/HarutoNakamura/so101-write-progress) | ResNet18 回帰 |
| 書字+進行度（統合・20k step） | [`HarutoNakamura/lerobot-write-prog`](https://huggingface.co/HarutoNakamura/lerobot-write-prog) | SmolVLA 7次元action |
| 書字+進行度（統合・**60k step**） | [`HarutoNakamura/lerobot-write-prog-60k`](https://huggingface.co/HarutoNakamura/lerobot-write-prog-60k) | 同上・学習量3倍（progress MAE 0.027） |
| 進行度推定（別建て・**実環境ft**） | 同 `so101-write-progress` の `progress_net_kadokawa.pt` | kadokawa 実環境データで fine-tune。実環境ホールドアウト MAE **0.022** |
| 書字+進行度（統合・**実環境ft**） | [`HarutoNakamura/lerobot-write-prog-ft-60k`](https://huggingface.co/HarutoNakamura/lerobot-write-prog-ft-60k) | 60k版に実環境データを混ぜて 60k step 追加学習 |

実環境（kadokawa セットアップ）で動かすなら **実環境ft の2つ**を使う。
元の学習環境・データセット再生には ft 無し版を使う。

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
# 実環境ft版を使う場合はこちらも
hf download HarutoNakamura/so101-write-progress progress_net_kadokawa.pt --local-dir .
```

### pixi で動かす場合（推奨・再現性あり）

リポジトリ直下の `pixi.toml` に環境とタスクを定義してある。
[pixi](https://pixi.sh) を入れて `pixi install` するだけで、ffmpeg や
opencv-python-headless の競合対策込みの環境が入る。

```bash
cd lerobot-writes
pixi install          # 環境構築（初回のみ）
pixi run get-ckpt     # progress_net.pt を先に落とす（別建て版用）

# 4つの実行方法（引数はそのまま後ろに足せる）
pixi run camera --digit 3 --cam 0          # ① Webカメラで動作確認
pixi run dataset --episode 9               # ② 学習データ再生で確認
pixi run robot --digit 3 --port /dev/ttyACM0 --cam_top 0 --cam_wrist 1        # ③ 実機・別建て版
pixi run robot-prog --digit 3 --port /dev/ttyACM0 --cam_top 0 --cam_wrist 1   # ④ 実機・統合版
pixi run robot-prog-60k --digit 3 --port /dev/ttyACM0                         # ④' 60k版（別建て版も並走・自動録画。要 get-ckpt 済み）
pixi run dataset-prog --episode 9          # ④の実機なし確認（要 HF ログイン）

# 実環境(kadokawa)では引数を後ろに足して実環境ft版に差し替える
# （タスクの引数は後勝ちなので --ckpt 等を上書きできる）
pixi run robot-prog-60k --digit 5 --port /dev/follower_arm \
    --ckpt HarutoNakamura/lerobot-write-prog-ft-60k \
    --prog_ckpt progress_net_kadokawa.pt --zero_start 0

# どのタスクも --save_dir を足すと録画され（保存先: inference/records/）、
# --hf_repo を足すと終了時に HF の dataset リポジトリへ自動アップロードされる（要 HF ログイン。詳細は後述）
pixi run camera --digit 3 --cam 0                --save_dir records --hf_repo HarutoNakamura/so101-run-logs
pixi run dataset --episode 9                     --save_dir records --hf_repo HarutoNakamura/so101-run-logs
pixi run robot --digit 3 --port /dev/ttyACM0     --save_dir records --hf_repo HarutoNakamura/so101-run-logs
pixi run robot-prog --digit 3 --port /dev/ttyACM0 --save_dir records --hf_repo HarutoNakamura/so101-run-logs
pixi run robot-prog-60k --digit 3 --port /dev/ttyACM0 --save_dir records --hf_repo HarutoNakamura/so101-run-logs
pixi run dataset-prog --episode 9                --save_dir records --hf_repo HarutoNakamura/so101-run-logs
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

# ②'' 60k 版と別建て版の同時実行・比較: --prog_ckpt で ProgressNet を並走させる。
#      緑バー=[VLA] 統合版 / 橙バー=[ResNet] 別建て版 の2本が表示され、
#      --save_dir を付けると両方の値が progress.csv に記録される
#      （pixi run robot-prog-60k はこれが既定。progress_net.pt は先に落としておく）
python realtime_smolvla_prog.py robot --ckpt HarutoNakamura/lerobot-write-prog-60k \
    --digit 3 --port COM5 --cam_top 0 --cam_wrist 1 \
    --prog_ckpt progress_net.pt --save_dir records

# ③ 実環境（kadokawa セットアップ）では実環境ft版を使う。
#    重みが実環境に校正済みなので --zero_start 0 で再基準化を切る
python realtime_progress.py robot --ckpt progress_net_kadokawa.pt --digit 5 \
    --policy HarutoNakamura/lerobot-write --port /dev/follower_arm

python realtime_smolvla_prog.py robot --ckpt HarutoNakamura/lerobot-write-prog-ft-60k \
    --digit 5 --port /dev/follower_arm \
    --prog_ckpt progress_net_kadokawa.pt --zero_start 0 --save_dir records

# ②の実機なし動作確認（派生データセット HarutoNakamura/so101-write-prog を上げてある場合。
#   --prog_ckpt / --save_dir はここでも併用できる）
python realtime_smolvla_prog.py dataset --ckpt HarutoNakamura/lerobot-write-prog \
    --repo_id HarutoNakamura/so101-write-prog --episode 9
```

- 緑バー = 平滑化済み進行度、`raw ○%` = 生の推定値。①は EMA+単調化（戻らない）。
  ②の統合版は action チャンク内で未来の progress を先読みして上振れするため、
  再推論による下方修正の直後の値を「現在地」とみなす ChunkedSmoother で表示する
  （単調ホールドなし。書けていないのに 100% に張り付くのを防ぐ）
- 別建て版は学習データと違う環境だと白紙でも 0.3 前後を出す（平均回帰）。
  ②の並走時は `--zero_start 15`（既定）で開始時の値を 0% に再基準化する。
  ①でも `--zero_start 15` を付ければ同じ補正が効く（既定は無効）。
  **実環境ft版 (`progress_net_kadokawa.pt`) は校正済みなので `--zero_start 0` にする**
- 1枚書かせるごとにスクリプトを起動し直す（起動時にポリシーと進行度がリセットされる）
- ProgressNet(①の進行度側) は CPU で十分。SmolVLA を回す部分は GPU 推奨

## 実行履歴の保存と Hugging Face アップロード

`--save_dir` を付けると、表示と同じ映像（進捗バー重畳済み）と進行度の推移が保存される。
全モード（dataset / camera / robot、①②とも）で使える。pixi タスクにもそのまま
追記できる（例: `pixi run robot-prog --digit 3 --port /dev/ttyACM0 --save_dir records`。
この場合の保存先は `inference/records/`）:

```bash
# ローカル保存のみ
python realtime_progress.py robot --ckpt progress_net.pt --digit 3 \
    --policy HarutoNakamura/lerobot-write --port COM5 --cam_top 0 --cam_wrist 1 \
    --save_dir records

# さらに終了時に HF の dataset リポジトリへ自動アップロード（要 hf auth login）
python realtime_smolvla_prog.py robot --ckpt HarutoNakamura/lerobot-write-prog \
    --digit 3 --port COM5 --cam_top 0 --cam_wrist 1 \
    --save_dir records --hf_repo HarutoNakamura/so101-run-logs
```

保存先は `<save_dir>/<実行日時>/` で、中身は3ファイル:

- `video.mp4` — 進捗バー重畳済みの top カメラ映像（並走時は2本のバー入り）
- `progress.csv` — frame, time_sec, raw, smoothed, sep_raw, sep_smoothed, event。
  raw/smoothed は統合版（②で `--prog_ckpt` 未指定なら唯一の系列）、
  sep_* は `--prog_ckpt` で並走させた別建て ProgressNet の値（未使用時は空欄）。
  camera モードの r リセットは event 列に記録される
- `meta.json` — 実行時の条件（mode / digit / ckpt / prog_ckpt / port / ema など）

`--hf_repo` のみ指定した場合はローカル `records/` に保存してからアップロードする。
リポジトリは private の dataset として自動作成され、実行日時ごとのフォルダに積み上がる。
アップロードに失敗してもローカルのデータは残る。

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
