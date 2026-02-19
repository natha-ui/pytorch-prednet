'''
Fine-tune PredNet model trained for t+1 prediction for up to t+5 prediction.
Converted to PyTorch 2026.
'''
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gc

np.random.seed(123)
torch.manual_seed(123)

from prednet import PredNet
from kitti_data import KITTI

# Settings
DATA_DIR = '/content/kitti_data'
WEIGHTS_DIR = '/content/drive/MyDrive/prednet_checkpoints'

# Parameters
nt = 15
extrap_start_time = 10  # starting at this time step, the prediction from the previous time step will be treated as the actual input
batch_size = 4
num_epochs = 150
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation

# Model architecture (match your training configuration)
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)

# File paths
orig_weights_file = os.path.join(WEIGHTS_DIR, 'best_model.pth')  # original t+1 weights
extrap_weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights_extrap_finetuned.pth')

# Data files
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define loss as MAE of frame predictions after t=0
def extrap_loss(y_true, y_hat):
    """
    Loss function for extrapolation.
    y_true: [batch, time, channels, height, width]
    y_hat: [batch, time, channels, height, width]
    """
    y_true = y_true[:, 1:]  # Skip first frame
    y_hat = y_hat[:, 1:]
    # 0.5 to match scale of loss when trained in error mode
    return 0.5 * torch.mean(torch.abs(y_true - y_hat))


class PredNetExtrapolator(nn.Module):
    """
    Wrapper for PredNet that implements extrapolation training.
    After extrap_start_time, uses predictions as input instead of ground truth.
    """
    def __init__(self, prednet_model, extrap_start_time):
        super(PredNetExtrapolator, self).__init__()
        self.prednet = prednet_model
        self.extrap_start_time = extrap_start_time
        # Set to prediction mode
        self.prednet.output_mode = 'prediction'

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, time_steps, channels, height, width]
        Returns:
            predictions: Tensor [batch, time_steps, channels, height, width]
        """
        batch_size, time_steps, channels, height, width = x.shape

        # We'll build predictions frame by frame
        all_predictions = []

        # Clone input so we can modify it
        modified_input = x.clone()

        # Process the sequence frame by frame
        for t in range(time_steps):
            # Prepare the input sequence up to current timestep
            if t < self.extrap_start_time:
                # Use ground truth for all frames up to t
                current_sequence = modified_input[:, :t+1]
            else:
                # Use ground truth up to extrap_start_time, then use predictions
                current_sequence = modified_input[:, :t+1]

            # Get prediction for the current timestep
            # PredNet returns only the last frame prediction
            pred = self.prednet(current_sequence)  # [batch, channels, height, width]
            all_predictions.append(pred)

            # For next iteration, replace the next frame with this prediction
            # (if we're past extrap_start_time)
            if t >= self.extrap_start_time and t < time_steps - 1:
                modified_input[:, t+1] = pred.detach()

        # Stack all predictions into [batch, time_steps, channels, height, width]
        return torch.stack(all_predictions, dim=1)


# Check for existing extrapolation checkpoints to continue training
def find_latest_extrap_checkpoint(weights_dir):
    """Find the most recent extrapolation checkpoint."""
    import glob

    # Look for both the best model and epoch checkpoints
    candidates = []

    # Check for best model
    best_model_path = os.path.join(weights_dir, 'prednet_kitti_weights_extrap_finetuned.pth')
    if os.path.exists(best_model_path):
        candidates.append(best_model_path)

    # Check for epoch checkpoints
    pattern = os.path.join(weights_dir, 'prednet_extrap_epoch_*.pth')
    epoch_checkpoints = glob.glob(pattern)
    candidates.extend(epoch_checkpoints)

    if not candidates:
        return None

    # Get the most recent by modification time
    latest = max(candidates, key=os.path.getmtime)
    return latest


def load_best_checkpoint(model, optimizer, extrap_weights_file, device):
    """Reload the best saved checkpoint into model and optimizer."""
    checkpoint = torch.load(extrap_weights_file, map_location=device)
    model.prednet.load_state_dict(checkpoint['model_state_dict'])
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except Exception:
        pass  # optimizer state mismatch is non-fatal
    print(f"  ↩ Rolled back to best checkpoint (epoch {checkpoint['epoch']+1}, "
          f"val_loss={checkpoint['best_val_loss']:.6f})")
    return checkpoint['best_val_loss']


# Try to find existing extrapolation checkpoint
extrap_checkpoint_path = find_latest_extrap_checkpoint(WEIGHTS_DIR)
start_epoch = 0
train_losses = []
val_losses = []
best_val_loss = float('inf')

if extrap_checkpoint_path and os.path.exists(extrap_checkpoint_path):
    print(f"Found existing extrapolation checkpoint: {extrap_checkpoint_path}")
    print("Resuming extrapolation fine-tuning from checkpoint...")

    checkpoint = torch.load(extrap_checkpoint_path, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    print(f"Resuming from epoch {start_epoch}")
    print(f"Previous best val loss: {best_val_loss:.4f}")
    if train_losses:
        print(f"Previous train loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"Previous val loss: {val_losses[-1]:.4f}")

    # Create base model and load weights
    base_model = PredNet(R_channels, A_channels, output_mode='prediction')
    base_model.load_state_dict(checkpoint['model_state_dict'])

    # Wrap with extrapolation capability
    model = PredNetExtrapolator(base_model, extrap_start_time)
    model = model.to(device)

    print("✓ Extrapolation checkpoint loaded successfully")

else:
    # No extrapolation checkpoint found, load pretrained t+1 model
    print("No extrapolation checkpoint found. Loading pretrained t+1 model...")
    if not os.path.exists(orig_weights_file):
        raise FileNotFoundError(f"Checkpoint not found: {orig_weights_file}")

    checkpoint = torch.load(orig_weights_file, map_location=device)
    print(f"Loaded t+1 checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"Previous train loss: {checkpoint['train_losses'][-1]:.4f}")
    print(f"Previous val loss: {checkpoint['val_losses'][-1]:.4f}")

    # Create base model in prediction mode
    base_model = PredNet(R_channels, A_channels, output_mode='prediction')
    base_model.load_state_dict(checkpoint['model_state_dict'])

    # Wrap with extrapolation capability
    model = PredNetExtrapolator(base_model, extrap_start_time)
    model = model.to(device)

    print("Starting extrapolation fine-tuning from scratch")

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Extrapolation starts at time step: {extrap_start_time}")
print(f"This means: frames 0-{extrap_start_time-1} use ground truth, frames {extrap_start_time}+ use predictions")

# Load data
print("\nLoading training data...")
kitti_train = KITTI(train_file, train_sources, nt)
train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

print("Loading validation data...")
kitti_val = KITTI(val_file, val_sources, nt)
val_loader = DataLoader(kitti_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(f"Training samples: {len(kitti_train)}")
print(f"Validation samples: {len(kitti_val)}")

# Optimizer — lower initial LR (0.001 → 0.0003) for fine-tuning stability;
# the extrapolation feedback loop amplifies large gradient steps.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# If resuming, load optimizer state
if extrap_checkpoint_path and 'optimizer_state_dict' in checkpoint:
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state loaded")
    except Exception as e:
        print(f"⚠ Could not load optimizer state: {e}")
        print("  Starting with fresh optimizer")


def adjust_learning_rate(optimizer, epoch, override_lr=None):
    """
    Sets LR to 0.0003 for epochs < 75, then 0.00003.
    Pass override_lr to set an explicit rate (used by spike recovery).
    """
    if override_lr is not None:
        lr = override_lr
    else:
        lr = 0.0003 if epoch < 75 else 0.00003
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train_epoch(model, train_loader, optimizer, device, samples_per_epoch, batch_size):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    max_batches = samples_per_epoch // batch_size

    for i, inputs in enumerate(train_loader):
        if num_batches >= max_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        # Convert from [batch, time, H, W, C] to [batch, time, C, H, W]
        inputs = inputs.permute(0, 1, 4, 2, 3)

        optimizer.zero_grad()

        # Forward pass - get predictions for all timesteps
        predictions = model(inputs)

        # Compute extrapolation loss
        loss = extrap_loss(inputs, predictions)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if i % 25 == 0:
            print(f'  Batch {i}/{max_batches}, Loss: {loss.item():.6f}')

        # Clear cache periodically
        if i % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return total_loss / num_batches if num_batches > 0 else 0


def validate(model, val_loader, device, N_seq_val, batch_size):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    max_batches = N_seq_val // batch_size

    with torch.no_grad():
        for i, inputs in enumerate(val_loader):
            if num_batches >= max_batches:
                break

            inputs = inputs.to(device, non_blocking=True)
            inputs = inputs.permute(0, 1, 4, 2, 3)

            # Forward pass
            predictions = model(inputs)

            # Compute loss
            loss = extrap_loss(inputs, predictions)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


# Create checkpoint directory
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Training loop
print("\n" + "=" * 70)
print("Starting fine-tuning for extrapolation")
if start_epoch > 0:
    print(f"Resuming from epoch {start_epoch + 1}")
print("=" * 70)

# Spike recovery config
SPIKE_FACTOR      = 1.5  # trigger rollback if val_loss > best_val_loss * this
RECOVERY_LR_SCALE = 0.5  # halve the LR after each rollback
MAX_RECOVERIES    = 5    # stop recovering after this many times

lr_was_switched = False  # track whether the epoch-75 best-weights reload has run
recovery_count  = 0
learning_rates  = []

try:
    for epoch in range(start_epoch, num_epochs):

        # ── LR schedule ──────────────────────────────────────────────────────
        # On the first epoch of the lower-LR phase, reload the best weights so
        # the reduced LR is applied to the best state, not a potentially drifted one.
        if epoch == 75 and not lr_was_switched and os.path.exists(extrap_weights_file):
            print("\n📉 LR drop at epoch 75 — reloading best weights before continuing")
            best_val_loss = load_best_checkpoint(model, optimizer, extrap_weights_file, device)
            lr_was_switched = True

        current_lr = adjust_learning_rate(optimizer, epoch)
        learning_rates.append(current_lr)

        print(f'\nEpoch {epoch+1}/{num_epochs}  |  LR: {current_lr:.2e}  |  '
              f'Recoveries used: {recovery_count}/{MAX_RECOVERIES}')

        # ── Train ─────────────────────────────────────────────────────────────
        train_loss = train_epoch(model, train_loader, optimizer, device,
                                 samples_per_epoch, batch_size)
        train_losses.append(train_loss)
        print(f'Training Loss: {train_loss:.6f}')

        # ── Validate ──────────────────────────────────────────────────────────
        val_loss = validate(model, val_loader, device, N_seq_val, batch_size)
        val_losses.append(val_loss)
        print(f'Validation Loss: {val_loss:.6f}')

        # ── Save best model ───────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.prednet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'extrap_start_time': extrap_start_time,
                'A_channels': A_channels,
                'R_channels': R_channels,
            }, extrap_weights_file)
            print(f'✓ Best model saved (val_loss: {val_loss:.6f})')

        # ── Spike recovery ────────────────────────────────────────────────────
        elif (val_loss > best_val_loss * SPIKE_FACTOR
              and recovery_count < MAX_RECOVERIES
              and os.path.exists(extrap_weights_file)):

            recovery_count += 1
            old_lr = get_current_lr(optimizer)
            new_lr = max(old_lr * RECOVERY_LR_SCALE, 1e-6)  # floor at 1e-6
            print(f'\n⚠ Loss spike detected! val_loss {val_loss:.4f} > '
                  f'{SPIKE_FACTOR}× best ({best_val_loss:.4f})')
            best_val_loss = load_best_checkpoint(model, optimizer,
                                                 extrap_weights_file, device)
            adjust_learning_rate(optimizer, epoch, override_lr=new_lr)
            print(f'  LR: {old_lr:.2e} → {new_lr:.2e}')

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if (epoch + 1) % 10 == 0:
            checkpoint_file = os.path.join(WEIGHTS_DIR,
                                           f'prednet_extrap_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.prednet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'extrap_start_time': extrap_start_time,
                'A_channels': A_channels,
                'R_channels': R_channels,
            }, checkpoint_file)
            print(f'💾 Backup checkpoint saved: epoch {epoch+1}')

        # Clear cache after each epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\n⚠ Training interrupted by user")
    print("Latest best model was saved!")

except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()

# Print summary
print("\n" + "=" * 70)
print("Training complete!")
print(f"Total epochs completed: {len(train_losses)}")
if train_losses:
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
print(f"Model saved to: {extrap_weights_file}")
print("=" * 70)

# Plot training history
if train_losses:
    plt.figure(figsize=(12, 5))

    # epochs_range covers the full loss history (all runs combined)
    epochs_range = range(1, len(train_losses) + 1)
    # lr_range covers only epochs trained in THIS run
    lr_range = range(start_epoch + 1, start_epoch + len(learning_rates) + 1)

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss', marker='o', markersize=3)
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='s', markersize=3)
    plt.axvline(x=75, color='r', linestyle='--', alpha=0.5, label='LR change')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Extrapolation Fine-tuning: Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot learning rate (only epochs trained in this run)
    plt.subplot(1, 2, 2)
    if learning_rates:
        plt.plot(lr_range, learning_rates, color='green', marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plot_path = os.path.join(WEIGHTS_DIR, 'extrap_training_history.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"\n📊 Training plot saved to: {plot_path}")
else:
    print("\n⚠ No training data to plot")

print("\n✅ Script finished successfully!")
