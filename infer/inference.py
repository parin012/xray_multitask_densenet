#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================
# Inference Only (Single image or folder)
# - Loads best_model(.pt) (supports space in path)
# - MPS/CUDA/CPU auto device
# - Outputs: CSV (per-image probs), seg mask PNG, overlay PNG
# ============================================================
import os, sys, glob, argparse
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
from pathlib import Path

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def preprocess_xray_1ch_array(gray_u8, out_hw=(448, 448)):
    g = cv2.resize(gray_u8, out_hw[::-1], interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    p1, p99 = np.percentile(g, [1, 99])
    g = np.clip((g.astype(np.float32) - p1) / max(p99 - p1, 1e-6), 0, 1)
    x = torch.from_numpy(g).unsqueeze(0)
    x = (x - 0.5) / 0.25
    return x

def convert_first_conv_to_1ch(densenet_model: nn.Module):
    old = densenet_model.features.conv0
    new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=False)
    with torch.no_grad():
        new.weight[:] = old.weight.mean(dim=1, keepdim=True)
    densenet_model.features.conv0 = new
    return densenet_model

class SimpleSegDecoder(nn.Module):
    def __init__(self, in_ch, mid_ch=64, out_ch=1):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_ch, mid_ch, 2, 2)
        self.up2 = nn.ConvTranspose2d(mid_ch, mid_ch, 2, 2)
        self.up3 = nn.ConvTranspose2d(mid_ch, mid_ch, 2, 2)
        self.up4 = nn.ConvTranspose2d(mid_ch, mid_ch, 2, 2)
        self.out_conv = nn.Conv2d(mid_ch, out_ch, 1)
    def forward(self, x, out_size):
        for up in [self.up1, self.up2, self.up3, self.up4]:
            x = F.relu(up(x))
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return self.out_conv(x)

class MultiTaskDenseNetXRay(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.backbone = convert_first_conv_to_1ch(base)
        ch = self.backbone.classifier.in_features
        self.cls_head = nn.Linear(ch, num_classes)
        self.seg_head = SimpleSegDecoder(ch, 64, 1)
    def forward(self, x):
        H, W = x.shape[-2:]
        f = F.relu(self.backbone.features(x))
        cls = self.cls_head(F.adaptive_avg_pool2d(f, 1).flatten(1))
        seg = self.seg_head(f, (H, W))
        return cls, seg

def smart_load_state(model, ckpt):
    def _try(sd):
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded state_dict (missing={len(missing)}, unexpected={len(unexpected)})")
    if isinstance(ckpt, dict) and ("model_state" in ckpt or "state_dict" in ckpt):
        _try(ckpt.get("model_state", ckpt.get("state_dict")))
    elif isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        _try(ckpt)
    else:
        _try(ckpt)

IMG_EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")

def list_images(path):
    p = Path(path)
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        return [str(p)]
    if p.is_dir():
        files = []
        for ext in IMG_EXTS:
            files += glob.glob(str(p / f"**/*{ext}"), recursive=True)
        return sorted(files)
    raise FileNotFoundError(f"Input not found: {path}")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def colorize_mask(mask_u8):
    return cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)

def overlay_on_gray(gray_u8, heatmap_bgr, alpha=0.35):
    gray_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(gray_bgr, 1.0, heatmap_bgr, alpha, 0)

def infer_folder(args):
    device = get_device()
    print("Device:", device)

    if args.class_names:
        class_names = [s.strip() for s in args.class_names.split(",")]
        num_classes = len(class_names)
    else:
        class_names = [f"class_{i}" for i in range(1, args.num_classes+1)]
        num_classes = args.num_classes

    weights_path = args.weights
    assert os.path.exists(weights_path), f"Checkpoint not found: {weights_path}"

    model = MultiTaskDenseNetXRay(num_classes=num_classes).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    smart_load_state(model, ckpt)
    model.eval()

    img_paths = list_images(args.input)
    print(f"Found {len(img_paths)} images")

    out_dir = ensure_dir(args.save_dir)
    seg_dir = ensure_dir(os.path.join(out_dir, "seg"))
    ovl_dir = ensure_dir(os.path.join(out_dir, "overlay"))

    import pandas as pd
    csv_rows = []
    softmax = nn.Softmax(dim=1)

    for ip in img_paths:
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skip (cannot read): {ip}")
            continue

        x = preprocess_xray_1ch_array(img, (args.img_size, args.img_size)).unsqueeze(0).to(device)
        with torch.no_grad():
            cls_out, seg_out = model(x)
            probs = softmax(cls_out).squeeze(0).cpu().numpy()
            pred_id = int(probs.argmax())
            pred_name = class_names[pred_id] if pred_id < len(class_names) else str(pred_id)
            seg_sig = torch.sigmoid(seg_out).squeeze().cpu().numpy()

        seg_u8 = (np.clip(seg_sig, 0, 1) * 255).astype(np.uint8)
        seg_color = colorize_mask(seg_u8)
        overlay = overlay_on_gray(cv2.resize(img, (args.img_size, args.img_size)), seg_color, alpha=args.alpha)

        stem = Path(ip).stem
        cv2.imwrite(os.path.join(seg_dir, f"{stem}_seg.png"), seg_u8)
        cv2.imwrite(os.path.join(ovl_dir, f"{stem}_overlay.png"), overlay)

        csv_rows.append({
            "file": ip,
            "pred_id": pred_id,
            "pred_name": pred_name,
            **{f"prob_{i}": float(p) for i,p in enumerate(probs)}
        })

    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(out_dir, "inference_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"✅ Done. Saved to: {out_dir}")
    print(f"- CSV: {csv_path}")
    print(f"- Seg masks: {seg_dir}")
    print(f"- Overlays: {ovl_dir}")

def parse_args():
    ap = argparse.ArgumentParser(description="X-ray MultiTask DenseNet Inference")
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--weights", required=True, help="path to best_model.pt (spaces OK)")
    ap.add_argument("--save-dir", default="./inference_out", help="output directory")
    ap.add_argument("--img-size", type=int, default=448, help="inference resolution")
    ap.add_argument("--num-classes", type=int, default=5, help="num classes if --class-names not given")
    ap.add_argument("--class-names", type=str, default="", help="comma-separated names (e.g. A,B,C,D,E)")
    ap.add_argument("--alpha", type=float, default=0.35, help="overlay alpha")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    infer_folder(args)

