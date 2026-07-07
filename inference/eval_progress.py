"""学習済み進行度モデルの評価。

検証エピソードを頭から流して「予測進行度 vs 真の進行度」を
- エピソードごとの折れ線 PNG (progress_ep{N}.png)
- 進捗バーを焼き込んだ MP4 (--video 指定時)
として $OUT に保存する。

実行例 (コンテナ内):
  python eval_progress.py --ckpt $OUTPUT_DIR/progress_write/progress_net.pt \
      --out $OUTPUT_DIR/progress_write/eval --video 9
"""
import argparse
from pathlib import Path

import torch

from progress_model import ProgressEstimator, task_to_digit, draw_progress_bar
from train_progress import CAM_KEY, get_episode_lengths


def run_episode(ds, estimator, digit):
    """1エピソード分の (真値, 生予測, 平滑化予測) 系列と各フレーム(tensor)を返す"""
    gt, raw, smooth, frames = [], [], [], []
    n = len(ds)
    estimator.reset()
    for i in range(n):
        item = ds[i]
        img = item[CAM_KEY]  # (3,H,W) float [0,1]
        p = estimator.estimate(img, digit)
        gt.append(i / max(n - 1, 1))
        raw.append(p)
        smooth.append(estimator.smoothed)
        frames.append(img)
    return gt, raw, smooth, frames


def save_plot(path, ep, digit, gt, raw, smooth):
    mae = sum(abs(a - b) for a, b in zip(raw, gt)) / len(gt)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # matplotlib が無ければ CSV に落とす
        csv = Path(path).with_suffix(".csv")
        csv.write_text("gt,raw,smooth\n" + "\n".join(
            f"{a},{b},{c}" for a, b, c in zip(gt, raw, smooth)))
        print(f"  (matplotlib なし: {csv} に保存)")
        return mae
    plt.figure(figsize=(7, 4))
    plt.plot(gt, gt, "k--", label="ideal")
    plt.plot(gt, raw, alpha=0.5, label="raw")
    plt.plot(gt, smooth, lw=2, label="smoothed")
    plt.xlabel("true progress")
    plt.ylabel("predicted progress")
    plt.title(f"episode {ep} (write{digit})  MAE={mae:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return mae


def save_video(path, frames, smooth, raw, digit, fps=30):
    import cv2
    import numpy as np
    h, w = frames[0].shape[1:]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for img, p, r in zip(frames, smooth, raw):
        bgr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()
        draw_progress_bar(bgr, p, digit=digit, raw=r)
        vw.write(bgr)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo_id", default="k1000dai/so101-write")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", default="9,19,29,39,49,59,69,79,89,99")
    ap.add_argument("--video", type=int, default=None,
                    help="このエピソードだけ進捗バー付き動画も書き出す")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    est = ProgressEstimator(args.ckpt, device=args.device)

    maes = []
    for ep in [int(x) for x in args.eps.split(",")]:
        ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[ep])
        digit = task_to_digit(ds[0]["task"])
        gt, raw, smooth, frames = run_episode(ds, est, digit)
        mae = save_plot(out / f"progress_ep{ep}.png", ep, digit, gt, raw, smooth)
        maes.append(mae)
        print(f"episode {ep} (write{digit}): MAE={mae:.3f}")
        if args.video == ep:
            fps = int(getattr(ds.meta, "fps", 30))
            save_video(out / f"progress_ep{ep}.mp4", frames, smooth, raw, digit, fps)
            print(f"  video -> {out / f'progress_ep{ep}.mp4'}")

    print(f"overall MAE = {sum(maes) / len(maes):.3f}  (plots in {out})")


if __name__ == "__main__":
    main()
