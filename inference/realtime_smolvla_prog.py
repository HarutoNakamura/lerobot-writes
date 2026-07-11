"""SmolVLA 統合版のリアルタイム進行度表示。

統合版は select_action の7次元目がそのまま進行度なので、
別モデル (progress_net.pt) は不要。1回の推論で関節指令と進行度が同時に出る。

  dataset : 検証エピソードの観測をポリシーに流して進捗バー表示（実機なし・要GPU推奨）
  robot   : SO-101 実機で書かせながら進捗バー表示

--prog_ckpt で別建て ProgressNet (progress_net.pt) を渡すと、同じ top 画像に対して
別建て版も並走させ、緑バー=統合版(VLA) / 橙バー=別建て版(ResNet) の2本で比較表示する。
録画時は progress.csv の sep_raw / sep_smoothed 列に別建て版の値も記録される
（60k 版と別建て版の挙動比較用）。

実行例:
  python realtime_smolvla_prog.py dataset --ckpt .../pretrained_model \
      --root .../so101-write-prog --episode 9
  python realtime_smolvla_prog.py robot --ckpt .../pretrained_model --digit 3
  python realtime_smolvla_prog.py robot --ckpt HarutoNakamura/lerobot-write-prog-60k \
      --digit 3 --prog_ckpt progress_net.pt --save_dir records
"""
import argparse
import time

# cv2 は torch/torchvision (経由で av) より先に import しないと cv2.imshow が
# フリーズする (ffmpeg 同梱バージョン競合: pytorch/vision#5940, opencv#21952)
import cv2  # noqa: F401
import torch

from progress_model import (ChunkedSmoother, ProgressEstimator,
                            draw_progress_bar, task_to_digit)
from eval_smolvla_prog import CAM_TOP, CAM_WRIST, load_policy, predict_step
from realtime_progress import make_robot
from recorder import maybe_recorder

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper")


def draw_overlays(frame_bgr, smoother, p, est, p_sep, digit):
    """統合版のバーを描く。別建て版を並走中は2本目を1段上に重ねる"""
    single = est is None
    draw_progress_bar(frame_bgr, smoother.smoothed, digit=digit, raw=p,
                      name=None if single else "VLA")
    if not single:
        draw_progress_bar(frame_bgr, est.smoothed, raw=p_sep,
                          slot=1, color=(0, 170, 255), name="ResNet")


def mode_dataset(args, policy, preproc, postproc, est=None):
    """データセットのエピソードをポリシーに流し、7次元目の進行度を表示"""
    import cv2
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode])
    digit = task_to_digit(ds[0]["task"])
    fps = int(getattr(ds.meta, "fps", 30))
    rec = maybe_recorder(args, fps=fps, ckpt=args.ckpt, prog_ckpt=args.prog_ckpt,
                         repo_id=args.repo_id, episode=args.episode, digit=digit)
    policy.reset()
    if est:
        est.reset()
    smoother = ChunkedSmoother(ema=args.ema)  # チャンク先読みの上振れを表示しない
    try:
        for i in range(len(ds)):
            item = ds[i]
            _, p = predict_step(policy, preproc, postproc,
                                item[CAM_TOP], item[CAM_WRIST],
                                item["observation.state"], item["task"], args.device)
            smoother.update(p)
            p_sep = est.estimate(item[CAM_TOP], digit) if est else None
            bgr = (item[CAM_TOP].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()
            draw_overlays(bgr, smoother, p, est, p_sep, digit)
            cv2.imshow("progress (integrated)", bgr)
            if rec:
                rec.write(bgr, p, smoother.smoothed, sep_raw=p_sep,
                          sep_smoothed=est.smoothed if est else None)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if rec:
            rec.close()
        cv2.destroyAllWindows()


def mode_robot(args, policy, preproc, postproc, est=None):
    """SO-101 実機ループ。関節指令と進行度が1回の select_action で出る"""
    import cv2
    import numpy as np

    robot = make_robot(args.port, args.cam_top, args.cam_wrist)
    task = f"write{args.digit}"
    rec = maybe_recorder(args, ckpt=args.ckpt, prog_ckpt=args.prog_ckpt,
                         port=args.port)
    policy.reset()
    if est:
        est.reset()
    smoother = ChunkedSmoother(ema=args.ema)  # チャンク先読みの上振れを表示しない
    print(f"task={task}  q で終了")
    try:
        while True:
            t0 = time.time()
            obs = robot.get_observation()
            img_top = torch.from_numpy(obs["top"]).permute(2, 0, 1).float() / 255.0
            img_wrist = torch.from_numpy(obs["wrist"]).permute(2, 0, 1).float() / 255.0
            state = torch.tensor([obs[f"{m}.pos"] for m in MOTORS], dtype=torch.float32)

            joints, p = predict_step(policy, preproc, postproc,
                                     img_top, img_wrist, state, task, args.device)
            robot.send_action({f"{m}.pos": float(a) for m, a in zip(MOTORS, joints)})

            smoother.update(p)
            p_sep = est.estimate(img_top, args.digit) if est else None
            frame = (obs["top"][:, :, ::-1]).astype(np.uint8).copy()  # RGB->BGR
            draw_overlays(frame, smoother, p, est, p_sep, args.digit)
            cv2.imshow("progress (integrated)", frame)
            if rec:
                rec.write(frame, p, smoother.smoothed, sep_raw=p_sep,
                          sep_smoothed=est.smoothed if est else None)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            # 30fps 目安のペーシング
            time.sleep(max(0.0, 1 / 30 - (time.time() - t0)))
    finally:
        if rec:
            rec.close()
        robot.disconnect()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dataset", "robot"])
    ap.add_argument("--ckpt", required=True,
                    help="checkpoints/.../pretrained_model か HF repo id")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ema", type=float, default=0.8)
    ap.add_argument("--prog_ckpt", default=None,
                    help="別建て ProgressNet (progress_net.pt) のパス。指定すると"
                         "統合版と同時に推定して2本のバーで比較表示し、録画時は"
                         "CSV の sep_raw/sep_smoothed 列にも記録する"
                         "（先に pixi run get-ckpt などで取得しておく）")
    ap.add_argument("--zero_start", type=int, default=15,
                    help="別建て ProgressNet の開始キャリブレーション: 最初のNフレームの"
                         "中央値を 0%% に再基準化 (0で無効)。初見環境では白紙でも"
                         "0.3前後を出すため既定で有効")
    ap.add_argument("--save_dir", default=None,
                    help="指定すると映像(MP4)と進行度ログ(CSV)をこのフォルダに保存")
    ap.add_argument("--hf_repo", default=None,
                    help="指定すると終了時に録画一式を HF dataset リポジトリへ"
                         "アップロード (例: HarutoNakamura/so101-run-logs)")
    # dataset モード
    ap.add_argument("--repo_id", default="local/so101-write-prog")
    ap.add_argument("--root", default=None)
    ap.add_argument("--episode", type=int, default=9)
    # robot モード
    ap.add_argument("--digit", type=int, default=3)
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="SO-101 のシリアルポート (Windows は COM5 など)")
    ap.add_argument("--cam_top", type=int, default=0, help="topカメラの番号")
    ap.add_argument("--cam_wrist", type=int, default=1, help="wristカメラの番号")
    args = ap.parse_args()

    policy, preproc, postproc = load_policy(args.ckpt, args.device)
    # 別建て版の並走 (ProgressNet は軽いので同じ device に同居できる)
    est = ProgressEstimator(args.prog_ckpt, device=args.device, ema=args.ema,
                            zero_start=args.zero_start) if args.prog_ckpt else None
    if args.mode == "dataset":
        mode_dataset(args, policy, preproc, postproc, est)
    else:
        mode_robot(args, policy, preproc, postproc, est)


if __name__ == "__main__":
    main()
