#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from configs.config import *
from data.dataset import make_items_by_order, XraySegClsDataset
from models.densenet_multitask import MultiTaskDenseNetXRay
from train.utils import acc_fn, dice_fn, save_curves, save_metrics_csv

import os, torch, torch.nn as nn
from torch.utils.data import DataLoader

# === 데이터: train/val 각각 별도 폴더에서 로드 ===
train_items, train_class_names = make_items_by_order(RAW_TRAIN, LABEL_TRAIN, NUM_CLASSES)
val_items,   val_class_names   = make_items_by_order(RAW_VAL,   LABEL_VAL,   NUM_CLASSES)

# 클래스 이름 일치 확인(선택)
assert train_class_names == val_class_names, \
    f"train/val 클래스 폴더명이 다릅니다: {train_class_names} vs {val_class_names}"

def _count_by_class(items, num_classes):
    cnt = {c:0 for c in range(1, num_classes+1)}
    for _,_,c in items: cnt[c]+=1
    return cnt

tr_cnt = _count_by_class(train_items, NUM_CLASSES)
va_cnt = _count_by_class(val_items, NUM_CLASSES)
print("\n===== 데이터 요약 (train/val 분리 사용) =====")
for c in range(1, NUM_CLASSES+1):
    cname = train_class_names[c-1]
    print(f"  [{c}] {cname:<15}  Train: {tr_cnt[c]:4d} | Val: {va_cnt[c]:4d}")
print(f"-------------------------------------------------")
print(f"  총계                 Train: {len(train_items):4d} | Val: {len(val_items):4d}\n")

# === DataLoader ===
train_loader = DataLoader(XraySegClsDataset(train_items, IMG_SIZE), batch_size=BATCH, shuffle=True,  num_workers=2)
val_loader   = DataLoader(XraySegClsDataset(val_items,   IMG_SIZE), batch_size=BATCH, shuffle=False, num_workers=2)

# === 모델 & 옵티마이저 ===
model = MultiTaskDenseNetXRay(num_classes=NUM_CLASSES).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
crit_seg = nn.BCEWithLogitsLoss()

# === 학습 루프 ===
history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[], "train_dice":[], "val_dice":[]}
best_val_loss = float("inf")
BEST_PATH  = os.path.join(SAVE_DIR, "best_model.pt")
PLOT_PATH  = os.path.join(SAVE_DIR, "curves.png")
METRIC_CSV = os.path.join(SAVE_DIR, "metrics.csv")

for epoch in range(1, EPOCHS+1):
    model.train()
    trL=trA=trD=0; n_tr=0
    for x,m,y in train_loader:
        x,m,y = x.to(device), m.to(device), y.to(device)
        c,s = model(x)
        loss = crit_cls(c,y) + LAMBDA_SEG*crit_seg(s,m)
        opt.zero_grad(); loss.backward(); opt.step()
        bs = x.size(0)
        trL += loss.item()*bs; trA += acc_fn(c,y)*bs; trD += dice_fn(s,m)*bs; n_tr += bs
    sched.step()
    trL/=n_tr; trA/=n_tr; trD/=n_tr

    model.eval()
    vaL=vaA=vaD=0; n_va=0
    with torch.no_grad():
        for x,m,y in val_loader:
            x,m,y = x.to(device), m.to(device), y.to(device)
            c,s = model(x)
            loss = crit_cls(c,y) + LAMBDA_SEG*crit_seg(s,m)
            bs = x.size(0)
            vaL += loss.item()*bs; vaA += acc_fn(c,y)*bs; vaD += dice_fn(s,m)*bs; n_va += bs
    vaL/=n_va; vaA/=n_va; vaD/=n_va

    history["train_loss"].append(trL); history["val_loss"].append(vaL)
    history["train_acc"].append(trA);  history["val_acc"].append(vaA)
    history["train_dice"].append(trD); history["val_dice"].append(vaD)
    print(f"[{epoch:02d}/{EPOCHS}] L={trL:.4f}/{vaL:.4f} ACC={trA:.3f}/{vaA:.3f} DICE={trD:.3f}/{vaD:.3f}")

    if vaL < best_val_loss:
        best_val_loss = vaL
        torch.save(model.state_dict(), BEST_PATH)
        print(f"💾 Best updated! val_loss={vaL:.4f}")

    save_curves(history, PLOT_PATH)
    save_metrics_csv(history, METRIC_CSV)

print("✅ 학습 완료: best_model.pt / metrics.csv / curves.png 저장 완료 (train/val 분리 버전)")

