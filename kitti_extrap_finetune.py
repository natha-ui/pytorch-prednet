'''
Fine-tune PredNet model trained for t+1 prediction for up to t+5 prediction.
'''
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

np.random.seed(123)
torch.manual_seed(123)

from prednet import PredNet
from kitti_data import KITTI

# Settings
DATA_DIR = '/content/kitti_data'
WEIGHTS_DIR = '/content/drive/MyDrive/prednet_checkpoints'

# Define loss as MAE of frame predictions after t=0
# It doesn't make sense to compute loss on error representation, since the error isn't wrt ground truth when extrapolating.
def extrap_loss(y_true, y_hat):
    """
    Loss function for extrapolation.
    Args:
        y_true: ground truth frames [batch, time, channels, height, width]
        y_hat: predicted frames [batch, time, channels, height, width]
    """
    y_true = y_true[:, 1:]  # Skip first frame
    y_hat = y_hat[:, 1:]
    # 0.5 to match scale of loss when trained in error mode (positive and negative errors split)
    return 0.5 * torch.mean(torch.abs(y_true - y_hat))

# Parameters
nt = 15
extrap_start_time = 10  # starting at this time step, the prediction from the previous time step will be treated as the actual input
batch_size = 4
num_epochs = 150
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation

# Model architecture (should match your training)
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)

# File paths
orig_weights_file = os.path.join(WEIGHTS_DIR, 'latest_checkpoint.pth')  # original t+1 weights
extrap_weights_file = os.path.join(WEIGHTS_DIR, 'prednet_extrap_finetuned.pth')  # where new weights will be saved

# Data files
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load original t+1 model
print("Loading pretrained model...")
checkpoint = torch.load(orig_weights_file, map_location=device)
print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")

# Create new model for extrapolation
model = PredNet(R_channels, A_channels, output_mode='prediction', extrap_start_time=extrap_start_time)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Extrapolation starts at time step: {extrap_start_time}")

# Data loaders
print("\nLoading data...")
kitti_train = KITTI(train_file, train_sources, nt)
train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True, num_workers=2)

kitti_val = KITTI(val_file, val_sources, nt)
val_loader = DataLoader(kitti_val, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Training samples: {len(kitti_train)}")
print(f"Validation samples: {len(kitti_val)}")

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler: 0.001 for first 75 epochs, then 0.0001
def lr_lambda(epoch):
    return 1.0 if epoch < 75 else 0.1

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# Training function
def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for i, inputs in enumerate(train_loader):
        if num_batches >= samples_per_epoch // batch_size:
            break
            
        inputs = inputs.to(device)
        inputs = inputs.permute(0, 1, 4, 2, 3)  # [batch, time, H, W, C] -> [batch, time, C, H, W]
        
        optimizer.zero_grad()
        
        # Forward pass - model returns predictions
        predictions = model(inputs)
        
        # Compute extrapolation loss
        loss = extrap_loss(inputs, predictions)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if i % 50 == 0:
            print(f'  Batch {i}, Loss: {loss.item():.6f}')
    
    return total_loss / num_batches

# Validation function
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for i, inputs in enumerate(val_loader):
            if num_batches >= N_seq_val // batch_size:
                break
                
            inputs = inputs.to(device)
            inputs = inputs.permute(0, 1, 4, 2, 3)
            
            predictions = model(inputs)
            loss = extrap_loss(inputs, predictions)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

# Training loop
print("\n" + "=" * 50)
print("Starting fine-tuning for extrapolation")
print("=" * 50)

best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, device)
    train_losses.append(train_loss)
    print(f'Training Loss: {train_loss:.6f}')
    
    # Validate
    val_loss = validate(model, val_loader, device)
    val_losses.append(val_loss)
    print(f'Validation Loss: {val_loss:.6f}')
    
    # Update learning rate
    scheduler.step()
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'extrap_start_time': extrap_start_time,
        }, extrap_weights_file)
        print(f'✓ Best model saved (val_loss: {val_loss:.6f})')
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_file = os.path.join(WEIGHTS_DIR, f'prednet_extrap_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'extrap_start_time': extrap_start_time,
        }, checkpoint_file)
        print(f'💾 Checkpoint saved: epoch {epoch+1}')

print("\n" + "=" * 50)
print("Fine-tuning complete!")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Model saved to: {extrap_weights_file}")
print("=" * 50)

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Extrapolation Fine-tuning History')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(WEIGHTS_DIR, 'extrap_training_history.png'))
plt.show()
print(f"\nTraining plot saved to {os.path.join(WEIGHTS_DIR, 'extrap_training_history.png')}")
