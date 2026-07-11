"""進行度のリアルタイム表示。

3つのモードがある:
  dataset : 検証エピソードを再生しながら進捗バーを表示（実機なしで動作確認）
  camera  : Web カメラ映像に対して進捗バーを表示（実機の top カメラを流用可）
  robot   : SO-101 実機ループ (SmolVLA 推論 + 進行度表示) の組み込み例

実行例:
  python realtime_progress.py dataset --ckpt .../progress_net.pt --episode 9
  python realtime_progress.py camera  --ckpt .../progress_net.pt --digit 3 --cam 0

robot モードは手元の SO-101 セットアップ (ポート名・カメラ設定) に合わせて
`make_robot()` を書き換えてから使うこと。既存の推論ループに組み込むだけなら:

    est = ProgressEstimator("progress_net.pt", device="cuda")
    est.reset()                              # エピソード開始時
    ...ループ内...
    p = est.estimate(obs["observation.images.top"], digit)   # 生の推定値
    draw_progress_bar(frame_bgr, est.smoothed, digit, raw=p) # 表示は平滑値
"""
import argparse
import time

# cv2 は torch/torchvision (経由で av) より先に import しないと cv2.imshow が
# フリーズする (ffmpeg 同梱バージョン競合: pytorch/vision#5940, opencv#21952)
import cv2  # noqa: F401
import torch

from progress_model import ProgressEstimator, draw_progress_bar, task_to_digit
from recorder import maybe_recorder


def mode_dataset(args, est):
    """データセットのエピソードを fps 通りに再生して進捗バーを重ねる"""
    import cv2
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode])
    digit = task_to_digit(ds[0]["task"])
    fps = int(getattr(ds.meta, "fps", 30))
    rec = maybe_recorder(args, fps=fps, repo_id=args.repo_id,
                         episode=args.episode, digit=digit)
    est.reset()
    try:
        for i in range(len(ds)):
            t0 = time.time()
            img = ds[i]["observation.images.top"]
            p = est.estimate(img, digit)
            bgr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()
            draw_progress_bar(bgr, est.smoothed, digit=digit, raw=p)
            cv2.imshow("progress", bgr)
            if rec:
                rec.write(bgr, p, est.smoothed)
            if cv2.waitKey(max(1, int(1000 / fps - (time.time() - t0) * 1000))) & 0xFF == ord("q"):
                break
    finally:
        if rec:
            rec.close()
        cv2.destroyAllWindows()


def mode_camera(args, est):
    """Web カメラ映像に進捗バーを重ねる（top カメラを直接見る場合）"""
    import cv2

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"camera {args.cam} を開けません")
    rec = maybe_recorder(args, cam=args.cam)
    est.reset()
    print("q: 終了 / r: 進行度リセット(新しいエピソード開始)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            p = est.estimate(frame, args.digit, bgr=True)
            draw_progress_bar(frame, est.smoothed, digit=args.digit, raw=p)
            cv2.imshow("progress", frame)
            if rec:
                rec.write(frame, p, est.smoothed)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                est.reset()
                if rec:
                    rec.mark("reset")
    finally:
        if rec:
            rec.close()
        cap.release()
        cv2.destroyAllWindows()


def make_robot(port="/dev/ttyACM0", cam_top=0, cam_wrist=1):
    """SO-101 実機接続。ポート/カメラは引数で指定（Windows は port="COM5" など）"""
    # lerobot v0.5系でモジュール名が so101_follower → so_follower に変更された
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.cameras.opencv import OpenCVCameraConfig

    cfg = SO101FollowerConfig(
        port=port,
        cameras={
            "top": OpenCVCameraConfig(index_or_path=cam_top, width=848, height=480, fps=30),
            "wrist": OpenCVCameraConfig(index_or_path=cam_wrist, width=640, height=480, fps=30),
        },
    )
    robot = SO101Follower(cfg)
    robot.connect()
    return robot


def mode_robot(args, est):
    """SmolVLA で書かせながら進行度を表示する組み込み例"""
    import cv2
    import numpy as np
    from eval_smolvla_prog import load_policy  # 前処理/後処理パイプライン込みで読む

    motors = ("shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll", "gripper")
    device = args.device
    # lerobot v0.5系は正規化が pre/post processor 側にあるため必ず挟む
    policy, preproc, postproc = load_policy(args.policy, device)
    robot = make_robot(args.port, args.cam_top, args.cam_wrist)
    task = f"write{args.digit}"
    rec = maybe_recorder(args, policy=args.policy, port=args.port)
    est.reset()
    policy.reset()
    print(f"task={task}  q で終了")
    try:
        while True:
            obs = robot.get_observation()
            img_top = torch.from_numpy(obs["top"]).permute(2, 0, 1).float() / 255.0
            img_wrist = torch.from_numpy(obs["wrist"]).permute(2, 0, 1).float() / 255.0
            state = torch.tensor([obs[f"{m}.pos"] for m in motors], dtype=torch.float32)
            batch = {
                "observation.images.top": img_top.unsqueeze(0).to(device),
                "observation.images.wrist": img_wrist.unsqueeze(0).to(device),
                "observation.state": state.unsqueeze(0).to(device),
                "task": task,
                "robot_type": "",
            }
            batch = preproc(batch)
            action = policy.select_action(batch)
            action = postproc(action).squeeze(0).float().cpu()
            robot.send_action({f"{m}.pos": float(a) for m, a in zip(motors, action[:6])})

            # ここが進行度表示。ポリシーとは独立に top 画像だけで推定する
            p = est.estimate(img_top, args.digit)
            frame = (obs["top"][:, :, ::-1]).astype(np.uint8).copy()  # RGB->BGR
            draw_progress_bar(frame, est.smoothed, digit=args.digit, raw=p)
            cv2.imshow("progress", frame)
            if rec:
                rec.write(frame, p, est.smoothed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if rec:
            rec.close()
        robot.disconnect()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dataset", "camera", "robot"])
    ap.add_argument("--ckpt", required=True, help="progress_net.pt のパス")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ema", type=float, default=0.8, help="表示平滑化 (0で平滑化なし)")
    ap.add_argument("--zero_start", type=int, default=0,
                    help="最初のNフレームの中央値を 0%% に再基準化 (0で無効)。"
                         "学習データと違う環境では白紙でも 0.3 前後を出すため、"
                         "実機では 15 程度を推奨")
    ap.add_argument("--save_dir", default=None,
                    help="指定すると映像(MP4)と進行度ログ(CSV)をこのフォルダに保存")
    ap.add_argument("--hf_repo", default=None,
                    help="指定すると終了時に録画一式を HF dataset リポジトリへ"
                         "アップロード (例: HarutoNakamura/so101-run-logs)")
    # dataset モード
    ap.add_argument("--repo_id", default="k1000dai/so101-write")
    ap.add_argument("--root", default=None)
    ap.add_argument("--episode", type=int, default=9)
    # camera / robot モード
    ap.add_argument("--digit", type=int, default=3, help="書かせる数字 (0-9)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--policy", default="HarutoNakamura/lerobot-write",
                    help="robot モードで使う SmolVLA (HF repo か checkpoints/.../pretrained_model)")
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="SO-101 のシリアルポート (Windows は COM5 など)")
    ap.add_argument("--cam_top", type=int, default=0, help="topカメラの番号")
    ap.add_argument("--cam_wrist", type=int, default=1, help="wristカメラの番号")
    args = ap.parse_args()

    est = ProgressEstimator(args.ckpt, device=args.device, ema=args.ema,
                            zero_start=args.zero_start)
    {"dataset": mode_dataset, "camera": mode_camera, "robot": mode_robot}[args.mode](args, est)


if __name__ == "__main__":
    main()
