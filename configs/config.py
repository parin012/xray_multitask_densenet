#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================
# X-ray 멀티태스크 DenseNet (일반 버전: train/val 폴더 분리 사용)
# ============================================================
import os, random, torch, numpy as np

# === 데이터 경로 (train/val 분리) ===
RAW_TRAIN   = " "
LABEL_TRAIN = " "

RAW_VAL     = " "
LABEL_VAL   = " "

# === 저장 경로 ===
SAVE_DIR = " "
os.makedirs(SAVE_DIR, exist_ok=True)

# === 하이퍼파라미터 ===
IMG_SIZE     = 448
BATCH        = 8
EPOCHS       = 30
LR           = 1e-4
WD           = 0.05
LAMBDA_SEG   = 0.3
NUM_CLASSES  = 5
SEED         = 2025

# === 디바이스 / 시드 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

