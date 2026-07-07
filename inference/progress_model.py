"""数字書きタスクの進行度 (0〜1) 推定モデル。

top カメラ画像 + 数字ラベル (one-hot) から「どこまで書けたか」を回帰する。
教師ラベルはエピソード内の時刻正規化 progress = frame_index / (length - 1)。
追加アノテーション不要で k1000dai/so101-write からそのまま学習できる。

学習: train_progress.py / 評価: eval_progress.py / 実機表示: realtime_progress.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMG_SIZE = 224
NUM_TASKS = 10  # write0〜write9
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preprocess(img: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) または (3,H,W) の float [0,1] 画像を 224x224 + ImageNet 正規化に整える"""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    img = F.interpolate(img, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return (img - _IMAGENET_MEAN.to(img.device)) / _IMAGENET_STD.to(img.device)


def task_to_digit(task: str) -> int:
    """'write3' -> 3"""
    return int(task[-1])


class ProgressNet(nn.Module):
    """ResNet18 バックボーン + 数字 one-hot を結合した回帰ヘッド。出力は sigmoid で 0〜1"""

    def __init__(self, pretrained: bool = True, num_tasks: int = NUM_TASKS):
        super().__init__()
        weights = None
        if pretrained:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        backbone = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # (B,512,1,1)
        self.num_tasks = num_tasks
        self.head = nn.Sequential(
            nn.Linear(512 + num_tasks, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, img: torch.Tensor, digit: torch.Tensor) -> torch.Tensor:
        """img: (B,3,224,224) 正規化済み, digit: (B,) long → (B,) 進行度 0〜1"""
        feat = self.backbone(img).flatten(1)
        onehot = F.one_hot(digit, num_classes=self.num_tasks).float()
        return torch.sigmoid(self.head(torch.cat([feat, onehot], dim=1))).squeeze(-1)

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "num_tasks": self.num_tasks,
                    "img_size": IMG_SIZE}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ProgressNet":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = cls(pretrained=False, num_tasks=ckpt["num_tasks"])
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device).eval()


class ProgressSmoother:
    """進行度系列の表示用平滑化 (EMA + 単調化)。

    別建てモデル・SmolVLA統合版どちらの出力にも使える。
    """

    def __init__(self, ema: float = 0.8, monotonic: bool = True):
        self.ema = ema
        self.monotonic = monotonic
        self.reset()

    def reset(self):
        """エピソード開始時に呼ぶ"""
        self.smoothed = 0.0
        self._ema_val = None

    def update(self, p: float) -> float:
        self._ema_val = p if self._ema_val is None else \
            self.ema * self._ema_val + (1 - self.ema) * p
        self.smoothed = max(self.smoothed, self._ema_val) if self.monotonic else self._ema_val
        return self.smoothed


class ProgressEstimator:
    """実機ループ組み込み用の薄いラッパ。

    - estimate() は生の推定値、smoothed は EMA + 単調化（表示が暴れない）した値
    - 画像は torch float [0,1] (3,H,W) / numpy uint8 (H,W,3) RGB or BGR を受け付ける
    """

    def __init__(self, ckpt_path: str, device: str = "cpu", ema: float = 0.8,
                 monotonic: bool = True):
        self.model = ProgressNet.load(ckpt_path, device)
        self.device = device
        self._smoother = ProgressSmoother(ema, monotonic)

    @property
    def smoothed(self) -> float:
        return self._smoother.smoothed

    def reset(self):
        """エピソード開始時に呼ぶ"""
        self._smoother.reset()

    @torch.no_grad()
    def estimate(self, img, digit: int, bgr: bool = False) -> float:
        import numpy as np
        if isinstance(img, np.ndarray):  # (H,W,3) uint8
            if bgr:
                img = img[:, :, ::-1]
            img = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
        img = preprocess(img.to(self.device))
        d = torch.tensor([digit], device=self.device)
        p = float(self.model(img, d).item())
        self._smoother.update(p)
        return p


def draw_progress_bar(frame, progress: float, digit=None, raw=None):
    """OpenCV BGR フレーム下部に進捗バーを描画して返す（フレームは書き換えられる）"""
    import cv2
    h, w = frame.shape[:2]
    bar_h = max(18, h // 20)
    y0, y1 = h - bar_h - 8, h - 8
    x0, x1 = 8, w - 8
    cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 60), -1)
    fill = x0 + int((x1 - x0) * max(0.0, min(1.0, progress)))
    cv2.rectangle(frame, (x0, y0), (fill, y1), (80, 200, 80), -1)
    label = f"{progress * 100:5.1f}%"
    if raw is not None:
        label += f" (raw {raw * 100:4.0f}%)"
    if digit is not None:
        label = f"write{digit}: " + label
    cv2.putText(frame, label, (x0 + 6, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
