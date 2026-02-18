#!/usr/bin/env python3
"""
test_prednet_stimulus.py
------------------------
Runs the PredNet extrapolation test on a custom video stimulus prepared
by mp4_to_prednet.py.  Mirrors the KITTI extrapolation test but is
self-contained: no KITTI paths needed.

Usage (standalone):
    python test_prednet_stimulus.py

Or after editing the CONFIG block to point at your files.

Output:
    • Per-frame MAE table printed to stdout
    • stimulus_extrapolation_mosaic.png  — multi-sequence visualisation
    • stimulus_extrap_predicted_t{N}-t{M}.jpg  — predicted free-run strip
    • stimulus_extrap_groundtruth_t{N}-t{M}.jpg — matching GT strip
"""

import os
import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# ── your existing model files must be on the Python path ──────────────────────
# If running from the same directory as prednet.py, this just works.
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

DATA_DIR        = './prednet_data'          # output dir from mp4_to_prednet.py
WEIGHTS_DIR     = './prednet_checkpoints'   # directory containing your .pth files

# Use the fine-tuned extrapolation checkpoint
EXTRAP_CKPT     = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights_extrap_finetuned.pth')

# Architecture — must match training
A_CHANNELS      = (3, 48, 96, 192)
R_CHANNELS      = (3, 48, 96, 192)

# Sequence config — must match what was used in mp4_to_prednet.py
NT              = 15          # total frames per sequence
EXTRAP_START    = 10          # frames 0-9 = GT input; 10-14 = free-running

BATCH_SIZE      = 4
N_VIS_SEQS      = 4           # sequences shown in the mosaic

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ═════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class StimulusDataset(Dataset):
    """
    Reads X_test.h5 produced by mp4_to_prednet.py.
    Returns float32 tensors in [T, H, W, C] order (matches KITTI loader).
    """

    def __init__(self, data_dir: str, nt: int):
        data_dir = Path(data_dir)
        x_path = data_dir / 'X_test.h5'
        s_path = data_dir / 'sources_test.h5'

        if not x_path.exists():
            raise FileNotFoundError(
                f"X_test.h5 not found in {data_dir}. "
                "Run mp4_to_prednet.py first."
            )

        with h5py.File(x_path, 'r') as f:
            self.X = f['data'][:]                        # [S, NT, H, W, C]  uint8

        with h5py.File(s_path, 'r') as f:
            self.sources = np.array(
                [s.decode('utf-8') for s in f['data'][:]]
            )

        self.nt = nt

        assert self.X.shape[1] >= nt, (
            f"Sequences have length {self.X.shape[1]} but NT={nt}. "
            "Re-run mp4_to_prednet.py with matching --nt."
        )

        log.info(f"Loaded {len(self.X)} sequences  shape={self.X.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        clip = self.X[idx, :self.nt]              # [NT, H, W, C]  uint8
        return torch.from_numpy(clip.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper (identical to training)
# ─────────────────────────────────────────────────────────────────────────────

class PredNetExtrapolator(nn.Module):
    """
    Feeds predictions back as inputs after extrap_start_time.
    Identical to the wrapper used during fine-tuning.
    """

    def __init__(self, prednet_model: PredNet, extrap_start_time: int):
        super().__init__()
        self.prednet           = prednet_model
        self.extrap_start_time = extrap_start_time
        self.prednet.output_mode = 'prediction'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, C, H, W]
        returns : [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        preds          = []
        inp            = x.clone()

        for t in range(T):
            p = self.prednet(inp[:, :t + 1])   # [B, C, H, W]
            preds.append(p)
            if EXTRAP_START <= t < T - 1:
                inp[:, t + 1] = p.detach()

        return torch.stack(preds, dim=1)        # [B, T, C, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred.float() - target.float())).item()


def to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] float tensor → HWC uint8 numpy."""
    arr = t.cpu().float().permute(1, 2, 0).numpy()
    return np.clip(arr, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

BG     = '#0b0c10'
PANEL  = '#13151c'
GT_C   = '#4fc3f7'
FR_C   = '#ff6b35'
ERR_C  = '#ffd166'
SEP    = '#1f2230'
TXT    = '#d8d5cc'
DIM    = '#555870'

matplotlib.rcParams.update({
    'figure.facecolor':  BG,   'axes.facecolor':   PANEL,
    'savefig.facecolor': BG,   'text.color':       TXT,
    'axes.labelcolor':   TXT,  'xtick.color':      DIM,
    'ytick.color':       DIM,  'axes.edgecolor':   SEP,
    'font.family':       'monospace', 'font.size': 8,
})

ERR_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'err', ['#0b0c10', '#1a1030', '#7c2d12', '#ff6b35', '#ffd166'], N=256
)

SHOW_FRAMES = list(range(EXTRAP_START - 1, NT))   # seed frame + free-run frames


def build_mosaic(gt_seqs, pred_seqs, err_seqs, mae_seqs, out_path: str):
    """
    Render a multi-sequence mosaic:
      rows = sequences,  cols = frames,  strips = GT / PRED / |err|
      bottom panel = per-frame MAE ribbon.
    """
    F     = len(SHOW_FRAMES)
    N     = len(gt_seqs)
    FIG_W = 2.0 * F + 1.2
    FIG_H = 2.2 * N + 2.4

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=130)

    outer = gridspec.GridSpec(
        N + 1, 1, figure=fig,
        hspace=0.06, height_ratios=[1.0] * N + [0.85],
        left=0.06, right=0.96, top=0.93, bottom=0.05,
    )

    # Title
    fig.text(0.5, 0.965,
             'PredNet  ·  Custom Stimulus Extrapolation',
             ha='center', color=TXT, fontsize=11, fontweight='bold',
             fontfamily='monospace')
    fig.text(0.5, 0.950,
             f'seed t={SHOW_FRAMES[0]}  |  '
             f'GT input t={SHOW_FRAMES[0]}–{EXTRAP_START-1}  |  '
             f'free-running t={EXTRAP_START}–{SHOW_FRAMES[-1]}',
             ha='center', color=DIM, fontsize=7, fontfamily='monospace')

    for row_i in range(N):
        inner = gridspec.GridSpecFromSubplotSpec(
            3, F, subplot_spec=outer[row_i], hspace=0.04, wspace=0.03,
        )
        for col_i, abs_t in enumerate(SHOW_FRAMES):
            is_seed    = (abs_t == EXTRAP_START - 1)
            is_freerun = (abs_t >= EXTRAP_START)
            border     = ERR_C if is_seed else (FR_C if is_freerun else GT_C)
            lw         = 1.8 if (is_seed or is_freerun) else 0.6

            def _make_ax(strip_row, img_arr, label=None, ylabel=None):
                ax = fig.add_subplot(inner[strip_row, col_i])
                ax.imshow(img_arr)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor(border); sp.set_linewidth(lw)
                if row_i == 0 and label:
                    ax.set_title(label, fontsize=6, color=border, pad=2)
                if col_i == 0 and ylabel:
                    ax.set_ylabel(ylabel, fontsize=6, color=border,
                                  rotation=0, labelpad=30, va='center')
                return ax

            tag = (f't={abs_t}\n[seed]' if is_seed
                   else f't={abs_t}\n[free]' if is_freerun
                   else f't={abs_t}\n[gt]')

            _make_ax(0, gt_seqs[row_i][col_i],
                     label=tag,
                     ylabel=f'seq {row_i+1}\nGT')
            ax_pr = fig.add_subplot(inner[1, col_i])
            if is_seed:
                ax_pr.imshow(gt_seqs[row_i][col_i])
                ax_pr.text(0.5, 0.5, '(seed)', transform=ax_pr.transAxes,
                           ha='center', va='center', fontsize=5, color=ERR_C, alpha=0.7)
            else:
                ax_pr.imshow(pred_seqs[row_i][col_i])
            ax_pr.set_xticks([]); ax_pr.set_yticks([])
            for sp in ax_pr.spines.values():
                sp.set_edgecolor(border); sp.set_linewidth(lw)
            if row_i == 0:
                ax_pr.set_title('', fontsize=6)
            if col_i == 0:
                ax_pr.set_ylabel('PRED', fontsize=6, color=FR_C,
                                 rotation=0, labelpad=30, va='center')

            ax_er = fig.add_subplot(inner[2, col_i])
            ax_er.imshow(
                np.zeros_like(err_seqs[row_i][col_i]) if is_seed else err_seqs[row_i][col_i],
                cmap=ERR_CMAP, vmin=0, vmax=60,
            )
            ax_er.set_xticks([]); ax_er.set_yticks([])
            for sp in ax_er.spines.values():
                sp.set_edgecolor(SEP); sp.set_linewidth(0.4)
            if col_i == 0:
                ax_er.set_ylabel('|err|', fontsize=6, color=ERR_C,
                                 rotation=0, labelpad=30, va='center')

    # ── MAE ribbon ────────────────────────────────────────────────────────────
    ax_mae  = fig.add_subplot(outer[N])
    ft      = np.array(SHOW_FRAMES)
    mae_mat = np.stack(mae_seqs)            # [N, F]
    m_mean  = mae_mat.mean(0)
    m_std   = mae_mat.std(0)

    ax_mae.axvspan(EXTRAP_START - 0.5, ft[-1] + 0.5, color=FR_C, alpha=0.08)
    ax_mae.axvline(EXTRAP_START - 0.5, color=FR_C, linewidth=1.0,
                   linestyle='--', alpha=0.7)
    ax_mae.fill_between(ft, m_mean - m_std, m_mean + m_std,
                        color=GT_C, alpha=0.15)
    for i in range(N):
        ax_mae.plot(ft, mae_mat[i], color=GT_C, linewidth=0.7, alpha=0.3)
    ax_mae.plot(ft, m_mean, color=GT_C, linewidth=2.0, label='mean MAE')
    ax_mae.plot(ft[ft >= EXTRAP_START], m_mean[ft >= EXTRAP_START],
                color=FR_C, linewidth=2.5, label='free-run MAE')
    ax_mae.annotate(f'  t{ft[-1]}: {m_mean[-1]:.1f}',
                    xy=(ft[-1], m_mean[-1]),
                    xytext=(ft[-1] - 0.4, m_mean[-1] + m_std[-1] + 2),
                    color=ERR_C, fontsize=7,
                    arrowprops=dict(arrowstyle='->', color=ERR_C, lw=0.8))

    ax_mae.set_xlim(ft[0] - 0.5, ft[-1] + 0.5)
    ax_mae.set_xticks(ft)
    ax_mae.set_xticklabels([f't{t}' for t in ft], fontsize=7)
    ax_mae.yaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))
    ax_mae.set_ylabel('MAE\n(px)', fontsize=7, color=DIM, labelpad=4)
    ax_mae.set_xlabel('Frame', fontsize=7, color=DIM, labelpad=2)
    ax_mae.tick_params(length=2)
    ax_mae.grid(axis='y', color=SEP, linewidth=0.5, alpha=0.5)
    ax_mae.spines['top'].set_visible(False)
    ax_mae.spines['right'].set_visible(False)
    ax_mae.legend(loc='upper left', fontsize=7, framealpha=0.0,
                  labelcolor=[GT_C, FR_C])
    ax_mae.text(EXTRAP_START - 1.5, ax_mae.get_ylim()[1] * 0.9,
                '◀ GT input', ha='right', fontsize=6.5, color=GT_C)
    ax_mae.text(EXTRAP_START + 0.1, ax_mae.get_ylim()[1] * 0.9,
                'free-running ▶', ha='left', fontsize=6.5, color=FR_C)

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
    if not Path(EXTRAP_CKPT).exists():
        log.error(f"Checkpoint not found: {EXTRAP_CKPT}")
        log.error("Set EXTRAP_CKPT in the CONFIG block to your .pth file.")
        sys.exit(1)

    base    = PredNet(R_CHANNELS, A_CHANNELS, output_mode='prediction')
    ckpt    = torch.load(EXTRAP_CKPT, map_location=DEVICE)
    base.load_state_dict(ckpt['model_state_dict'])
    log.info(f"Loaded checkpoint  epoch={ckpt['epoch']+1}  "
             f"best_val_loss={ckpt.get('best_val_loss', float('nan')):.4f}")

    model = PredNetExtrapolator(base, extrap_start_time=EXTRAP_START).to(DEVICE)
    model.eval()

    # ── Evaluation loop ───────────────────────────────────────────────────────
    step_mae  = torch.zeros(NT)
    n_batches = 0

    # Collect per-sequence data for the mosaic
    vis_gt    = []   # list of [F, H, W, 3] uint8 arrays
    vis_pred  = []
    vis_err   = []
    vis_mae   = []   # list of [F] float arrays

    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            # inputs : [B, T, H, W, C]  float32
            # PredNet expects [B, T, C, H, W]
            inputs_chw = inputs.permute(0, 1, 4, 2, 3).to(DEVICE)

            preds = model(inputs_chw)   # [B, T, C, H, W]

            # Per-timestep MAE (skip t=0 to match training convention)
            for t in range(1, NT):
                step_mae[t] += compute_mae(
                    preds[:, t].cpu(), inputs_chw[:, t].cpu()
                )
            n_batches += 1

            # Collect visualisation data for the first N_VIS_SEQS sequences
            if len(vis_gt) < N_VIS_SEQS:
                slots = min(N_VIS_SEQS - len(vis_gt), inputs_chw.shape[0])
                for s in range(slots):
                    gt_frames   = inputs_chw[s].cpu().float()    # [T, C, H, W]
                    pred_frames = preds[s].cpu().float()

                    gt_show   = [to_hwc_uint8(gt_frames[t])   for t in SHOW_FRAMES]
                    pred_show = [to_hwc_uint8(pred_frames[t]) for t in SHOW_FRAMES]
                    err_show  = [
                        np.abs(
                            gt_show[i].astype(np.float32) -
                            pred_show[i].astype(np.float32)
                        ).mean(-1)
                        for i in range(len(SHOW_FRAMES))
                    ]
                    mae_show  = [e.mean() for e in err_show]

                    vis_gt.append(gt_show)
                    vis_pred.append(pred_show)
                    vis_err.append(err_show)
                    vis_mae.append(mae_show)

    # ── Per-frame MAE summary ─────────────────────────────────────────────────
    step_mae /= max(n_batches, 1)

    print('\n' + '═' * 52)
    print('  PredNet  ·  Custom Stimulus  ·  Extrapolation MAE')
    print('═' * 52)
    print(f'  {"Frame":>6}  {"MAE (px)":>10}  {"Mode":>14}')
    print('  ' + '─' * 36)
    for t in range(1, NT):
        mode   = 'free-running' if t >= EXTRAP_START else 'ground-truth'
        marker = '  ◄ start free-run' if t == EXTRAP_START else ''
        print(f'  {t:>6}  {step_mae[t].item():>10.4f}  {mode:>14}{marker}')

    avg_gt    = step_mae[1:EXTRAP_START].mean().item()
    avg_fr    = step_mae[EXTRAP_START:].mean().item()
    ratio     = avg_fr / max(avg_gt, 1e-9)

    print('  ' + '─' * 36)
    print(f'  Avg MAE  t1–t{EXTRAP_START-1}   (GT input)   : {avg_gt:.4f} px')
    print(f'  Avg MAE  t{EXTRAP_START}–t{NT-1}  (free-run)   : {avg_fr:.4f} px')
    print(f'  Degradation ratio                : {ratio:.3f}×')
    print('═' * 52 + '\n')

    # ── Visualisation ─────────────────────────────────────────────────────────
    build_mosaic(vis_gt, vis_pred, vis_err, vis_mae,
                 out_path='stimulus_extrapolation_mosaic.png')

    log.info("✅  Done.")


if __name__ == '__main__':
    main()
