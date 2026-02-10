from google.colab import drive
import os

drive.mount('/content/drive')
checkpoint_dir = '/content/drive/MyDrive/prednet_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"Checkpoints will be saved to: {checkpoint_dir}")

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import gc

from prednet import PredNet
from kitti_data import KITTI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

batch_size = 4
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
lr = 0.001
nt = 10
num_epochs = 150

train_file = 'kitti_data/X_train.hkl'
train_sources = 'kitti_data/sources_train.hkl'
val_file = 'kitti_data/X_val.hkl'
val_sources = 'kitti_data/sources_val.hkl'

print("Loading training data...")
kitti_train = KITTI(train_file, train_sources, nt)
train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True, num_workers=2)

print("Loading validation data...")
kitti_val = KITTI(val_file, val_sources, nt)
val_loader = DataLoader(kitti_val, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Training samples: {len(kitti_train)}")
print(f"Validation samples: {len(kitti_val)}")

model = PredNet(R_channels, A_channels, output_mode='error')
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for i, inputs in enumerate(train_loader):
        inputs = inputs.to(device)
        inputs = inputs.permute(0, 1, 4, 2, 3)
        
        optimizer.zero_grad()
        errors = model(inputs)
        loss = torch.mean(errors)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 50 == 0:
            print(f'  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(device)
            inputs = inputs.permute(0, 1, 4, 2, 3)
            
            errors = model(inputs)
            loss = torch.mean(errors)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

# Checkpoint file path
checkpoint_path = f'{checkpoint_dir}/latest_checkpoint.pth'

# Initialize or resume
start_epoch = 0
train_losses = []
val_losses = []

# Check if checkpoint exists and load it
if os.path.exists(checkpoint_path):
    print("=" * 50)
    print("Found existing checkpoint")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    
    print(f"✓ Resuming from epoch {start_epoch}")
    print(f"✓ Previous train loss: {train_losses[-1]:.4f}")
    print(f"✓ Previous val loss: {val_losses[-1]:.4f}")
    print("=" * 50)
else:
    print("New training - no checkpoint found")

print(f"\nTraining from epoch {start_epoch + 1} to {num_epochs}")
print("=" * 50)

# Training loop
try:
    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        print(f'Training Loss: {train_loss:.4f}')
        
        # Validate
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Save checkpoint after EVERY epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, checkpoint_path)
        print(f'Checkpoint saved')
        
        # Save numbered checkpoint every 10 epochs (for backup)
        if (epoch + 1) % 10 == 0:
            numbered_checkpoint = f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, numbered_checkpoint)
            print(f'💾 Backup checkpoint saved: epoch {epoch+1}')
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    print("Latest checkpoint was saved - you can resume later!")
    
except Exception as e:
    print(f"\nError during training: {e}")
    print("Latest checkpoint was saved - you can resume later!")

print("\n" + "=" * 50)
print("Training session complete!")
print(f"Completed epochs: {len(train_losses)}")
if train_losses:
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
print("=" * 50)

if train_losses:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{checkpoint_dir}/training_history.png')
    plt.show()
    print(f"Training plot saved to {checkpoint_dir}/training_history.png")
else:
    print("No training data to plot yet")
