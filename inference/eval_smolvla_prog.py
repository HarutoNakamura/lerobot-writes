"""SmolVLA 統合版（7次元 action の7次元目 = 進行度）の評価。

検証エピソードの観測を頭から流し、select_action の7次元目を進行度として
真値 (frame_index / (length-1)) と比較する。出力は eval_progress.py と同形式:
- 折れ線 PNG (progress_ep{N}.png)
- 進捗バー焼き込み MP4 (--video 指定時)
あわせて関節6次元のオープンループ MAE も表示する（行動側が壊れていないかの目安）。

実行例 (コンテナ内・要GPU):
  python eval_smolvla_prog.py \
      --ckpt $OUTPUT_DIR/smolvla_prog/checkpoints/last/pretrained_model \
      --root $USER_DIR/so101-write-prog \
      --out  $OUTPUT_DIR/smolvla_prog/eval_progress --video 9
"""
import argparse
from pathlib import Path

import torch

from progress_model import ProgressSmoother, task_to_digit
from eval_progress import save_plot, save_video

CAM_TOP = "observation.images.top"
CAM_WRIST = "observation.images.wrist"
JOINT_DIM = 6


def load_policy(ckpt: str, device: str):
    """policy + 前処理/後処理パイプラインを返す（lerobot v0.5 系）"""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = SmolVLAPolicy.from_pretrained(ckpt).to(device).eval()
    preproc, postproc = make_pre_post_processors(policy.config, pretrained_path=ckpt)
    return policy, preproc, postproc


@torch.no_grad()
def predict_step(policy, preproc, postproc, img_top, img_wrist, state, task, device):
    """1ステップ推論。戻り値: (関節6次元 tensor, 進行度 float)"""
    obs = {
        CAM_TOP: img_top.unsqueeze(0).to(device),      # (1,3,H,W) float [0,1]
        CAM_WRIST: img_wrist.unsqueeze(0).to(device),
        "observation.state": state.unsqueeze(0).to(device),  # (1,6)
        "task": task,
        "robot_type": "",
    }
    obs = preproc(obs)
    action = policy.select_action(obs)
    action = postproc(action)                          # (1,7) 逆正規化済み
    a = action.squeeze(0).float().cpu()
    return a[:JOINT_DIM], float(a[JOINT_DIM].clamp(0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoints/.../pretrained_model")
    ap.add_argument("--repo_id", default="local/so101-write-prog")
    ap.add_argument("--root", required=True, help="進行度付きデータセットのルート")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", default="9,19,29,39,49,59,69,79,89,99")
    ap.add_argument("--video", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy, preproc, postproc = load_policy(args.ckpt, args.device)

    prog_maes, joint_maes = [], []
    for ep in [int(x) for x in args.eps.split(",")]:
        ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[ep])
        digit = task_to_digit(ds[0]["task"])
        policy.reset()
        smoother = ProgressSmoother()
        gt, raw, smooth, frames = [], [], [], []
        joint_err = 0.0
        n = len(ds)
        for i in range(n):
            item = ds[i]
            joints, p = predict_step(
                policy, preproc, postproc,
                item[CAM_TOP], item[CAM_WRIST], item["observation.state"],
                item["task"], args.device)
            gt.append(float(item["action"][JOINT_DIM]))  # データ側の進行度ラベル
            raw.append(p)
            smooth.append(smoother.update(p))
            frames.append(item[CAM_TOP])
            joint_err += (joints - item["action"][:JOINT_DIM].float()).abs().mean().item()

        mae = save_plot(out / f"progress_ep{ep}.png", ep, digit, gt, raw, smooth)
        prog_maes.append(mae)
        joint_maes.append(joint_err / n)
        print(f"episode {ep} (write{digit}): progress MAE={mae:.3f} "
              f"joint open-loop MAE={joint_err / n:.3f}", flush=True)
        if args.video == ep:
            fps = int(getattr(ds.meta, "fps", 30))
            save_video(out / f"progress_ep{ep}.mp4", frames, smooth, raw, digit, fps)
            print(f"  video -> {out / f'progress_ep{ep}.mp4'}")

    print(f"overall: progress MAE={sum(prog_maes) / len(prog_maes):.3f} "
          f"joint MAE={sum(joint_maes) / len(joint_maes):.3f}  (plots in {out})")


if __name__ == "__main__":
    main()
