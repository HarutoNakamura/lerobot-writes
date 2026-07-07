"""進行度推定モデルの学習。

k1000dai/so101-write の各フレームに progress = frame_index / (length-1) を
教師として付け、top カメラ画像 + 数字 one-hot から回帰する。

検証は「各数字の最後の1エピソード」(既定: 9,19,...,99) をホールドアウト。
最良 (val MAE 最小) のモデルを $OUT/progress_net.pt に保存する。

実行例 (コンテナ内):
  python train_progress.py --out $OUTPUT_DIR/progress_write
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from progress_model import ProgressNet, preprocess, task_to_digit

CAM_KEY = "observation.images.top"


def get_episode_lengths(ds) -> dict:
    """{episode_index: フレーム数} を返す（lerobot のバージョン差異を吸収）"""
    eps = ds.meta.episodes
    if isinstance(eps, dict):  # v2.x 系: {ep: {"length": ...}}
        return {int(k): int(v["length"]) for k, v in eps.items()}
    try:  # pandas.DataFrame
        import pandas as pd
        if isinstance(eps, pd.DataFrame):
            return {int(r["episode_index"]): int(r["length"]) for _, r in eps.iterrows()}
    except ImportError:
        pass
    # datasets.Dataset / pyarrow 表など列アクセスできるもの
    return {int(e): int(l) for e, l in zip(eps["episode_index"], eps["length"])}


def make_length_lut(lengths: dict, device) -> torch.Tensor:
    lut = torch.zeros(max(lengths) + 1, dtype=torch.float32)
    for k, v in lengths.items():
        lut[k] = v
    return lut.to(device)


def batch_targets(batch, len_lut) -> torch.Tensor:
    ep = batch["episode_index"].long().to(len_lut.device)
    fi = batch["frame_index"].float().to(len_lut.device)
    denom = (len_lut[ep] - 1).clamp(min=1)
    return (fi / denom).clamp(0, 1)


def batch_digits(batch, device) -> torch.Tensor:
    return torch.tensor([task_to_digit(t) for t in batch["task"]], device=device)


@torch.no_grad()
def evaluate(model, loader, len_lut, device, limit=None) -> float:
    model.eval()
    abs_err, n = 0.0, 0
    for i, batch in enumerate(loader):
        if limit is not None and i >= limit:
            break
        img = preprocess(batch[CAM_KEY].to(device))
        pred = model(img, batch_digits(batch, device))
        target = batch_targets(batch, len_lut)
        abs_err += (pred - target).abs().sum().item()
        n += len(pred)
    return abs_err / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="k1000dai/so101-write")
    ap.add_argument("--root", default=None, help="ローカルデータセットのルート(省略時はHFキャッシュ)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_eps", default="9,19,29,39,49,59,69,79,89,99",
                    help="検証用エピソード(各数字の最後の1本)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--pretrained", type=int, default=1,
                    help="1: ImageNet 事前学習 ResNet18 (要 03_prepare_progress.sh)")
    ap.add_argument("--limit_batches", type=int, default=None, help="スモーク用: 1epochあたりの上限")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device

    val_eps = [int(x) for x in args.val_eps.split(",") if x != ""]
    probe = LeRobotDataset(args.repo_id, root=args.root)  # メタ取得用
    lengths = get_episode_lengths(probe)
    all_eps = sorted(lengths.keys())
    train_eps = [e for e in all_eps if e not in val_eps]
    print(f"episodes: train={len(train_eps)} val={len(val_eps)} / frames total={sum(lengths.values())}")

    train_ds = LeRobotDataset(args.repo_id, root=args.root, episodes=train_eps)
    val_ds = LeRobotDataset(args.repo_id, root=args.root, episodes=val_eps)
    len_lut = make_length_lut(lengths, device)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(device == "cuda"),
                          drop_last=True, persistent_workers=(args.num_workers > 0))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=max(2, args.num_workers // 2))

    model = ProgressNet(pretrained=bool(args.pretrained)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_mae, history = float("inf"), []
    for epoch in range(args.epochs):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for i, batch in enumerate(train_dl):
            if args.limit_batches is not None and i >= args.limit_batches:
                break
            img = preprocess(batch[CAM_KEY].to(device, non_blocking=True))
            pred = model(img, batch_digits(batch, device))
            target = batch_targets(batch, len_lut)
            loss = torch.nn.functional.smooth_l1_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(pred)
            seen += len(pred)
            if i % 50 == 0:
                print(f"epoch {epoch} step {i}/{len(train_dl)} loss={loss.item():.4f} "
                      f"({seen / max(time.time() - t0, 1e-9):.0f} samples/s)", flush=True)

        val_limit = None if args.limit_batches is None else max(1, args.limit_batches // 2)
        mae = evaluate(model, val_dl, len_lut, device, limit=val_limit)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_mae": mae})
        print(f"[epoch {epoch}] train_loss={running / max(seen, 1):.4f} val_MAE={mae:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if mae < best_mae:
            best_mae = mae
            model.save(str(out / "progress_net.pt"))
            print(f"  -> best 更新, 保存: {out / 'progress_net.pt'}")

    model.save(str(out / "progress_net_last.pt"))
    (out / "history.json").write_text(json.dumps(
        {"history": history, "best_val_mae": best_mae, "val_eps": val_eps,
         "args": vars(args)}, indent=2))
    print(f"done. best val MAE = {best_mae:.4f} (進行度の平均絶対誤差; 0.05なら±5%程度)")


if __name__ == "__main__":
    main()
