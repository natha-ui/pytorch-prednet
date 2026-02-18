'''
Test PredNet model for both t+1 prediction and t+5 extrapolation.
Loads the fine-tuned extrapolation checkpoint and evaluates on the test set.
Converted to PyTorch 2026.
'''
import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader
from kitti_data import KITTI
from prednet import PredNet
import torchvision
from PIL import Image


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def save_image(tensor, filename):
    im = Image.fromarray(np.rollaxis(tensor.numpy(), 0, 3))
    im.save(filename)


# ─────────────────────────────────────────────
# PredNetExtrapolator  (mirrors the training wrapper)
# ─────────────────────────────────────────────

class PredNetExtrapolator(nn.Module):
    """
    Wraps PredNet so that after `extrap_start_time` the model feeds its own
    predictions back as inputs instead of ground-truth frames.
    Identical to the wrapper used during fine-tuning.
    """
    def __init__(self, prednet_model, extrap_start_time):
        super().__init__()
        self.prednet = prednet_model
        self.extrap_start_time = extrap_start_time
        self.prednet.output_mode = 'prediction'

    def forward(self, x):
        """
        Args:
            x: [batch, time_steps, C, H, W]
        Returns:
            predictions: [batch, time_steps, C, H, W]
        """
        batch_size, time_steps, channels, height, width = x.shape
        all_predictions = []
        modified_input = x.clone()

        for t in range(time_steps):
            current_sequence = modified_input[:, :t + 1]
            pred = self.prednet(current_sequence)   # [batch, C, H, W]
            all_predictions.append(pred)

            if t >= self.extrap_start_time and t < time_steps - 1:
                modified_input[:, t + 1] = pred.detach()

        return torch.stack(all_predictions, dim=1)  # [batch, T, C, H, W]


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

batch_size     = 4          # smaller batch for safety; increase if VRAM allows
A_channels     = (3, 48, 96, 192)
R_channels     = (3, 48, 96, 192)

DATA_DIR       = '/content/kitti_data'
WEIGHTS_DIR    = '/content/drive/MyDrive/prednet_checkpoints'

test_file      = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources   = os.path.join(DATA_DIR, 'sources_test.hkl')

# t+1 (original) checkpoint
orig_ckpt_path   = os.path.join(WEIGHTS_DIR, 'best_model.pth')
# t+5 fine-tuned checkpoint
extrap_ckpt_path = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights_extrap_finetuned.pth')

# Sequence length used during fine-tuning
nt_extrap      = 15
extrap_start   = 10   # frames 0-9 = ground truth; frames 10-14 = free-running

# For the plain t+1 test we only need a short sequence
nt_orig        = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ─────────────────────────────────────────────
# MAE helper
# ─────────────────────────────────────────────

def compute_mae(pred, target):
    """Mean Absolute Error over all elements."""
    return torch.mean(torch.abs(pred.float() - target.float())).item()


# ─────────────────────────────────────────────
# ① Standard t+1 test  (original model)
# ─────────────────────────────────────────────

print('\n' + '=' * 60)
print('① STANDARD t+1 PREDICTION TEST')
print('=' * 60)

kitti_test_orig = KITTI(test_file, test_sources, nt_orig)
loader_orig = DataLoader(kitti_test_orig, batch_size=batch_size, shuffle=False)

model_orig = PredNet(R_channels, A_channels, output_mode='prediction')
ckpt_orig  = torch.load(orig_ckpt_path, map_location=device)
model_orig.load_state_dict(ckpt_orig['model_state_dict'])
print(f"Loaded t+1 model from epoch {ckpt_orig['epoch'] + 1}")
print(f"  Train loss : {ckpt_orig['train_losses'][-1]:.4f}")
print(f"  Val   loss : {ckpt_orig['val_losses'][-1]:.4f}")

model_orig = model_orig.to(device)
model_orig.eval()

with torch.no_grad():
    for inputs in loader_orig:
        # [batch, T, H, W, C] → [batch, T, C, H, W]
        inputs = inputs.permute(0, 1, 4, 2, 3).to(device)

        pred_t1 = model_orig(inputs)                    # [batch, C, H, W]  (last-frame prediction)
        origin  = inputs[:, -1].cpu().byte()            # ground-truth last frame
        pred_t1 = pred_t1.cpu().byte()

        mae_t1 = compute_mae(pred_t1, origin)
        print(f'\nt+1 MAE on first batch: {mae_t1:.4f}')

        grid_origin = torchvision.utils.make_grid(origin,  nrow=4)
        grid_pred   = torchvision.utils.make_grid(pred_t1, nrow=4)

        save_image(grid_origin, 'origin_t1.jpg')
        save_image(grid_pred,   'predicted_t1.jpg')
        print('Saved: origin_t1.jpg  |  predicted_t1.jpg')
        break


# ─────────────────────────────────────────────
# ② t+5 extrapolation test  (fine-tuned model)
# ─────────────────────────────────────────────

print('\n' + '=' * 60)
print('② t+5 EXTRAPOLATION TEST  (fine-tuned model)')
print('=' * 60)

if not os.path.exists(extrap_ckpt_path):
    print(f'⚠  Extrapolation checkpoint not found: {extrap_ckpt_path}')
    print('   Run the fine-tuning script first, then re-run this test.')
else:
    kitti_test_ext = KITTI(test_file, test_sources, nt_extrap)
    loader_ext = DataLoader(kitti_test_ext, batch_size=batch_size, shuffle=False)

    base_model_ext = PredNet(R_channels, A_channels, output_mode='prediction')
    ckpt_ext       = torch.load(extrap_ckpt_path, map_location=device)
    base_model_ext.load_state_dict(ckpt_ext['model_state_dict'])
    print(f"Loaded extrap model from epoch {ckpt_ext['epoch'] + 1}")
    print(f"  Best val loss : {ckpt_ext.get('best_val_loss', float('nan')):.4f}")

    model_ext = PredNetExtrapolator(base_model_ext, extrap_start_time=extrap_start)
    model_ext = model_ext.to(device)
    model_ext.eval()

    # ── per-step MAE accumulators ──────────────────────────────────────
    # We care especially about the free-running frames (extrap_start … nt_extrap-1)
    step_mae   = torch.zeros(nt_extrap)
    n_batches  = 0

    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader_ext):
            inputs = inputs.permute(0, 1, 4, 2, 3).to(device)   # [B, T, C, H, W]

            # Full-sequence predictions
            preds = model_ext(inputs)   # [B, T, C, H, W]

            # Accumulate per-timestep MAE  (skip frame 0 to match training loss)
            for t in range(1, nt_extrap):
                step_mae[t] += compute_mae(preds[:, t].cpu(), inputs[:, t].cpu())

            n_batches += 1

            # ── Save visualisation grids for the FIRST batch ────────────
            if batch_idx == 5: #changed to see performance on non-blurry stimulus
                # Ground-truth strip: frames extrap_start … extrap_start+4
                gt_strip   = inputs[0, extrap_start:extrap_start + 5].cpu().byte()   # [5, C, H, W]
                pred_strip = preds [0, extrap_start:extrap_start + 5].cpu().byte()   # [5, C, H, W]

                # Last ground-truth frame before free-running starts
                last_gt    = inputs[0, extrap_start - 1].cpu().byte().unsqueeze(0)   # [1, C, H, W]

                grid_gt   = torchvision.utils.make_grid(gt_strip,   nrow=5)
                grid_pred = torchvision.utils.make_grid(pred_strip,  nrow=5)
                grid_seed = torchvision.utils.make_grid(last_gt,    nrow=1)

                save_image(grid_seed, 'extrap_last_seed_frame.jpg')
                save_image(grid_gt,   'extrap_groundtruth_t10_t14.jpg')
                save_image(grid_pred, 'extrap_predicted_t10_t14.jpg')
                print('\nSaved visualisations for first sequence:')
                print('  extrap_last_seed_frame.jpg       ← last GT frame before free-running (t=9)')
                print('  extrap_groundtruth_t10_t14.jpg   ← GT  frames t10 … t14')
                print('  extrap_predicted_t10_t14.jpg     ← PRED frames t10 … t14 (free-running)')

    # ── Print per-step summary ─────────────────────────────────────────
    step_mae /= max(n_batches, 1)

    print('\n── Per-timestep MAE (averaged over test set) ──')
    print(f'  {"Frame":>6}  {"MAE":>8}  {"Mode":>12}')
    print('  ' + '-' * 32)
    for t in range(1, nt_extrap):
        mode = 'free-running' if t >= extrap_start else 'ground-truth'
        marker = ' ◄' if t == extrap_start else ''
        print(f'  {t:>6}  {step_mae[t].item():>8.4f}  {mode:>12}{marker}')

    avg_gt_mae    = step_mae[1:extrap_start].mean().item()
    avg_extrap_mae = step_mae[extrap_start:].mean().item()

    print(f'\n  Avg MAE  frames  1 – {extrap_start-1}  (ground-truth  input): {avg_gt_mae:.4f}')
    print(f'  Avg MAE  frames {extrap_start} – {nt_extrap-1}  (free-running pred):   {avg_extrap_mae:.4f}')
    print(f'  Degradation ratio (extrap / gt): {avg_extrap_mae / max(avg_gt_mae, 1e-9):.3f}x')

print('\n✅ Testing complete!')
