#!/usr/bin/env python3
"""
test_stimulus_t1.py
-------------------
Tests the PredNet t+1 model on a custom video stimulus prepared by
mp4_to_prednet.py.  Mirrors the original KITTI t+1 test script but
reads from .h5 files instead of .hkl, and uses the stimulus dataset
rather than KITTI.

Usage:
    python test_stimulus_t1.py

Edit the CONFIG block to point at your files before running.
"""

import os
import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from torch.utils.data import Dataset, DataLoader

from prednet import PredNet

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG  — edit these to match your setup
# ═════════════════════════════════════════════════════════════════════════════

DATA_DIR    = './prednet_data'          # output dir from mp4_to_prednet.py
CKPT_PATH   = '/content/drive/MyDrive/prednet_checkpoints/best_model.pth'

A_CHANNELS  = (3, 48, 96, 192)
R_CHANNELS  = (3, 48, 96, 192)

NT          = 10    # sequence length — must match --nt used in mp4_to_prednet.py
BATCH_SIZE  = 4
N_VIS_SEQS  = 4     # sequences shown in the mosaic

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ═════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class StimulusDataset(Dataset):
    """
    Reads X_test.h5 / sources_test.h5 produced by mp4_to_prednet.py.
    Returns float32 tensors shaped [T, H, W, C] to match PredNet's
    expected input convention (permuted to [T, C, H, W] in the loop).
    """

    def __init__(self, data_dir: str, nt: int):
        data_dir = Path(data_dir)
        x_path   = data_dir / 'X_test.h5'
        s_path   = data_dir / 'sources_test.h5'

        if not x_path.exists():
            raise FileNotFoundError(
                f"X_test.h5 not found in {data_dir}. "
                "Run mp4_to_prednet.py first."
            )

        with h5py.File(x_path, 'r') as f:
            self.X = f['data'][:]                    # [S, NT, H, W, C] uint8

        with h5py.File(s_path, 'r') as f:
            self.sources = np.array(
                [s.decode('utf-8') for s in f['data'][:]]
            )

        self.nt = nt

        if self.X.shape[1] < nt:
            raise ValueError(
                f"Sequences have length {self.X.shape[1]} but NT={nt}. "
                "Re-run mp4_to_prednet.py with a matching --nt value."
            )

        log.info(f"Loaded {len(self.X)} sequences  shape={self.X.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        clip = self.X[idx, :self.nt]                 # [NT, H, W, C] uint8
        return torch.from_numpy(clip.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred.float() - target.float())).item()


def to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] float tensor → HWC uint8 numpy array."""
    return np.clip(t.cpu().float().permute(1, 2, 0).numpy(), 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

BG    = '#0b0c10'
PANEL = '#13151c'
GT_C  = '#4fc3f7'
PR_C  = '#ff6b35'
ERR_C = '#ffd166'
SEP   = '#1f2230'
TXT   = '#d8d5cc'
DIM   = '#555870'

matplotlib.rcParams.update({
    'figure.facecolor':  BG,   'axes.facecolor':  PANEL,
    'savefig.facecolor': BG,   'text.color':      TXT,
    'axes.labelcolor':   TXT,  'xtick.color':     DIM,
    'ytick.color':       DIM,  'axes.edgecolor':  SEP,
    'font.family':       'monospace', 'font.size': 8,
})

ERR_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'err', ['#0b0c10', '#1a1030', '#7c2d12', '#ff6b35', '#ffd166'], N=256
)


def build_mosaic(gt_seqs, pred_seqs, err_seqs, mae_per_seq,
                 step_mae, n_batches, out_path='stimulus_t1_mosaic.png'):
    """
    Multi-sequence mosaic:
      rows = sequences,  cols = all NT frames
      strips per row = GT / PRED / |err|
      bottom panel = per-frame MAE ribbon across the test set
    """
    F     = NT
    N     = len(gt_seqs)
    FIG_W = 1.8 * F + 1.2
    FIG_H = 2.2 * N + 2.4

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=130)

    outer = gridspec.GridSpec(
        N + 1, 1, figure=fig,
        hspace=0.06, height_ratios=[1.0] * N + [0.85],
        left=0.06, right=0.96, top=0.93, bottom=0.05,
    )

    fig.text(0.5, 0.965,
             'PredNet  ·  t+1 Prediction  ·  Custom Stimulus',
             ha='center', color=TXT, fontsize=11, fontweight='bold',
             fontfamily='monospace')
    fig.text(0.5, 0.950,
             f'frames t0–t{NT-2} = input context  |  t{NT-1} = predicted vs ground truth',
             ha='center', color=DIM, fontsize=7, fontfamily='monospace')

    for row_i in range(N):
        inner = gridspec.GridSpecFromSubplotSpec(
            3, F, subplot_spec=outer[row_i], hspace=0.04, wspace=0.03,
        )
        for t in range(F):
            is_pred_frame = (t == F - 1)
            border = PR_C if is_pred_frame else GT_C
            lw     = 1.8  if is_pred_frame else 0.6

            def _ax(strip, img, label=None, ylabel=None, col=border):
                ax = fig.add_subplot(inner[strip, t])
                ax.imshow(img)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor(border); sp.set_linewidth(lw)
                if row_i == 0 and label:
                    ax.set_title(label, fontsize=6, color=border, pad=2)
                if t == 0 and ylabel:
                    ax.set_ylabel(ylabel, fontsize=6, color=col,
                                  rotation=0, labelpad=30, va='center')
                return ax

            tag = f't={t}\n[pred]' if is_pred_frame else f't={t}'
            _ax(0, gt_seqs[row_i][t],
                label=tag,
                ylabel=f'seq {row_i+1}\nGT')

            # PRED strip: only meaningful for the final frame
            ax_pr = fig.add_subplot(inner[1, t])
            if is_pred_frame:
                ax_pr.imshow(pred_seqs[row_i])
            else:
                ax_pr.imshow(gt_seqs[row_i][t])
                ax_pr.text(0.5, 0.5, '(input)', transform=ax_pr.transAxes,
                           ha='center', va='center', fontsize=4,
                           color=DIM, alpha=0.6)
            ax_pr.set_xticks([]); ax_pr.set_yticks([])
            for sp in ax_pr.spines.values():
                sp.set_edgecolor(border); sp.set_linewidth(lw)
            if t == 0:
                ax_pr.set_ylabel('PRED', fontsize=6, color=PR_C,
                                 rotation=0, labelpad=30, va='center')

            # ERR strip: only meaningful for the final frame
            ax_er = fig.add_subplot(inner[2, t])
            if is_pred_frame:
                ax_er.imshow(err_seqs[row_i], cmap=ERR_CMAP, vmin=0, vmax=60)
            else:
                ax_er.imshow(np.zeros_like(err_seqs[row_i]),
                             cmap=ERR_CMAP, vmin=0, vmax=60)
            ax_er.set_xticks([]); ax_er.set_yticks([])
            for sp in ax_er.spines.values():
                sp.set_edgecolor(SEP); sp.set_linewidth(0.4)
            if t == 0:
                ax_er.set_ylabel('|err|', fontsize=6, color=ERR_C,
                                 rotation=0, labelpad=30, va='center')

    # ── Per-frame MAE ribbon ──────────────────────────────────────────────────
    ax_mae = fig.add_subplot(outer[N])
    mae_val = (step_mae[NT - 1] / max(n_batches, 1)).item()

    # Single bar showing the t+1 prediction MAE, with per-sequence dots overlaid
    ax_mae.bar([NT - 1], [mae_val], color=PR_C, width=0.5, alpha=0.85, zorder=2)
    for seq_mae in vis_mae:
        ax_mae.scatter([NT - 1], [seq_mae], color=ERR_C,
                       s=18, zorder=3, alpha=0.7)

    # Annotate the bar
    ax_mae.text(NT - 1, mae_val + 0.3,
                f'{mae_val:.2f} px',
                ha='center', va='bottom', fontsize=7, color=PR_C)

    # Add a reference line for context
    ax_mae.axhline(mae_val, color=PR_C, linewidth=0.8, linestyle='--', alpha=0.4)

    ax_mae.set_xticks([NT - 1])
    ax_mae.set_xticklabels([f't={NT-1}  (predicted frame)'], fontsize=8, color=PR_C)
    ax_mae.yaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
    ax_mae.set_ylabel('MAE (px)', fontsize=7, color=DIM, labelpad=4)
    ax_mae.set_xlim(NT - 2, NT)
    ax_mae.tick_params(length=2)
    ax_mae.grid(axis='y', color=SEP, linewidth=0.5, alpha=0.5)
    ax_mae.spines['top'].set_visible(False)
    ax_mae.spines['right'].set_visible(False)

    # Explanatory caption
    ax_mae.text(0.02, 0.92,
                f'Input context: t=0 – t={NT-2}   |   dots = individual sequences',
                transform=ax_mae.transAxes,
                fontsize=6.5, color=DIM, va='top')

    # Colourbar
    cbar_ax = fig.add_axes([0.975, 0.22, 0.010, 0.68])
    sm = plt.cm.ScalarMappable(
        cmap=ERR_CMAP,
        norm=matplotlib.colors.Normalize(vmin=0, vmax=60),
    )
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('|err| px', fontsize=6, color=DIM, labelpad=4)
    cb.ax.yaxis.set_tick_params(color=DIM, labelsize=6)
    cb.outline.set_edgecolor(SEP)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=DIM)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()
    log.info(f"Mosaic saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info(f"Device : {DEVICE}")

    # ── Dataset & loader ──────────────────────────────────────────────────────
    dataset = StimulusDataset(DATA_DIR, NT)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    if not Path(CKPT_PATH).exists():
        log.error(f"Checkpoint not found: {CKPT_PATH}")
        log.error("Update CKPT_PATH in the CONFIG block.")
        sys.exit(1)

    model = PredNet(R_CHANNELS, A_CHANNELS, output_mode='prediction')
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    log.info(f"Loaded checkpoint  epoch={ckpt['epoch']+1}  "
             f"train_loss={ckpt['train_losses'][-1]:.4f}  "
             f"val_loss={ckpt['val_losses'][-1]:.4f}")

    model = model.to(DEVICE)
    model.eval()

    # ── Evaluation loop ───────────────────────────────────────────────────────
    # step_mae[t] accumulates MAE for frame t across all batches
    step_mae  = torch.zeros(NT)
    n_batches = 0

    vis_gt   = []   # list of [NT, H, W, 3] uint8 — all frames for context strip
    vis_pred = []   # list of [H, W, 3] uint8 — single predicted last frame
    vis_err  = []   # list of [H, W] float — pixel error at last frame
    vis_mae  = []   # list of float — scalar MAE at last frame

    with torch.no_grad():
        for inputs in loader:
            # inputs : [B, T, H, W, C]
            inputs_chw = inputs.permute(0, 1, 4, 2, 3).to(DEVICE)  # [B, T, C, H, W]

            pred = model(inputs_chw)   # [B, C, H, W]  — prediction for t=NT-1

            # Per-frame MAE: compare each frame's GT against the t+1 prediction
            # (the model predicts the next frame given all frames up to t,
            #  so the natural comparison is pred vs inputs[:, NT-1])
            gt_last = inputs_chw[:, NT - 1]   # [B, C, H, W]
            for t in range(1, NT):
                # For t+1 testing we have one prediction per sequence (the last frame).
                # Report it at position NT-1; other slots show GT reconstruction MAE = 0
                # to keep the per-frame bar chart honest.
                if t == NT - 1:
                    step_mae[t] += compute_mae(pred.cpu(), gt_last.cpu())

            n_batches += 1

            # Collect visualisation data
            if len(vis_gt) < N_VIS_SEQS:
                slots = min(N_VIS_SEQS - len(vis_gt), inputs_chw.shape[0])
                for s in range(slots):
                    gt_all  = [to_hwc_uint8(inputs_chw[s, t].cpu()) for t in range(NT)]
                    pr_img  = to_hwc_uint8(pred[s].cpu())
                    gt_img  = to_hwc_uint8(gt_last[s].cpu())
                    err_map = np.abs(gt_img.astype(np.float32) -
                                    pr_img.astype(np.float32)).mean(-1)

                    vis_gt.append(gt_all)
                    vis_pred.append(pr_img)
                    vis_err.append(err_map)
                    vis_mae.append(err_map.mean())

    # ── Summary table ─────────────────────────────────────────────────────────
    mae_t1 = (step_mae[NT - 1] / max(n_batches, 1)).item()

    print('\n' + '═' * 48)
    print('  PredNet  ·  Custom Stimulus  ·  t+1 MAE')
    print('═' * 48)
    print(f'  Predicted frame  : t={NT-1}')
    print(f'  Input context    : t=0 … t={NT-2}')
    print(f'  MAE (px)         : {mae_t1:.4f}')
    print(f'  Test sequences   : {len(dataset)}')
    print(f'  Batches evaluated: {n_batches}')
    print('═' * 48 + '\n')

    # ── Visualisation ─────────────────────────────────────────────────────────
    build_mosaic(vis_gt, vis_pred, vis_err, vis_mae,
                 step_mae, n_batches,
                 out_path='stimulus_t1_mosaic.png')

    log.info("✅  Done.")


if __name__ == '__main__':
    main()
