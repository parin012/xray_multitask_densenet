#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch, matplotlib.pyplot as plt, pandas as pd, os

def acc_fn(out, y):
    return (out.argmax(1) == y).float().mean().item()

def dice_fn(seg, mask, eps=1e-6):
    p = (torch.sigmoid(seg) > 0.5).float()
    inter = (p*mask).sum((1,2,3))
    denom = p.sum((1,2,3)) + mask.sum((1,2,3))
    return ((2*inter+eps)/(denom+eps)).mean().item()

def save_curves(history, path):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(history["train_loss"],label="Train"); plt.plot(history["val_loss"],label="Val")
    plt.title("Loss"); plt.legend(); plt.grid()
    plt.subplot(1,3,2)
    plt.plot(history["train_acc"],label="Train"); plt.plot(history["val_acc"],label="Val")
    plt.title("Accuracy"); plt.legend(); plt.grid()
    plt.subplot(1,3,3)
    plt.plot(history["train_dice"],label="Train"); plt.plot(history["val_dice"],label="Val")
    plt.title("Dice"); plt.legend(); plt.grid()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def save_metrics_csv(history, path):
    pd.DataFrame(history).to_csv(path, index_label="epoch")

