"""実行履歴の保存。

進捗バーを重ねた表示フレームを MP4 に、進行度の推移を CSV に、
実行条件を meta.json に書き出す。realtime_progress.py /
realtime_smolvla_prog.py の全モードから使う。

保存先: <save_dir>/<YYYYmmdd_HHMMSS>/
  video.mp4     進捗バー重畳済みのカメラ映像
  progress.csv  frame, time_sec, raw, smoothed, sep_raw, sep_smoothed, event
                (sep_* は統合版と別建て ProgressNet を並走させた場合のみ入る)
  meta.json     実行時の引数・fps・開始時刻

--hf_repo を指定すると close 時に同じフォルダ構成のまま Hugging Face の
dataset リポジトリへアップロードする (要 `hf auth login`)。
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2


class SessionRecorder:
    def __init__(self, save_dir, fps=30, meta=None, hf_repo=None):
        self.hf_repo = hf_repo
        self.dir = Path(save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._writer = None  # フレームサイズが分かってから開く
        self._t0 = time.time()
        self._n = 0
        self._csv_f = open(self.dir / "progress.csv", "w", newline="")
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow(["frame", "time_sec", "raw", "smoothed",
                            "sep_raw", "sep_smoothed", "event"])
        with open(self.dir / "meta.json", "w") as f:
            json.dump({"started": datetime.now().isoformat(), "fps": fps,
                       **(meta or {})}, f, ensure_ascii=False, indent=2)
        print(f"録画開始: {self.dir}")

    def write(self, frame_bgr, raw, smoothed, event="",
              sep_raw=None, sep_smoothed=None):
        """表示直前/直後のオーバーレイ済み BGR フレームと進行度を1件記録。

        sep_raw / sep_smoothed には別建て ProgressNet を並走させたときの値を渡す。
        """
        if self._writer is None:
            h, w = frame_bgr.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.dir / "video.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self._writer.write(frame_bgr)
        fmt = lambda v: "" if v is None else f"{float(v):.4f}"
        self._csv.writerow([self._n, f"{time.time() - self._t0:.3f}",
                            fmt(raw), fmt(smoothed),
                            fmt(sep_raw), fmt(sep_smoothed), event])
        self._n += 1

    def mark(self, event):
        """フレームを伴わないイベント (進行度リセット等) を記録"""
        self._csv.writerow([self._n, f"{time.time() - self._t0:.3f}",
                            "", "", "", "", event])

    def close(self):
        if self._writer is not None:
            self._writer.release()
        self._csv_f.close()
        print(f"録画保存: {self.dir} ({self._n} frames)")
        if self.hf_repo:
            self._upload()

    def _upload(self):
        """録画一式を HF dataset リポジトリへ。失敗してもローカルは残る"""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(self.hf_repo, repo_type="dataset",
                            private=True, exist_ok=True)
            api.upload_folder(repo_id=self.hf_repo, repo_type="dataset",
                              folder_path=str(self.dir),
                              path_in_repo=self.dir.name)
            print(f"HFアップロード完了: https://huggingface.co/datasets/"
                  f"{self.hf_repo}/tree/main/{self.dir.name}")
        except Exception as e:
            print(f"HFアップロード失敗 (ローカルには保存済み: {self.dir}): {e}")


def maybe_recorder(args, fps=30, **meta):
    """--save_dir か --hf_repo 指定時のみ SessionRecorder を返す (未指定なら None)"""
    save_dir = getattr(args, "save_dir", None)
    hf_repo = getattr(args, "hf_repo", None)
    if not save_dir and not hf_repo:
        return None
    # 呼び出し側が明示した meta (dataset モードの digit 等) を優先する
    meta = {"mode": args.mode, "ema": args.ema,
            "digit": getattr(args, "digit", None), **meta}
    return SessionRecorder(save_dir or "records", fps=fps, meta=meta,
                           hf_repo=hf_repo)
