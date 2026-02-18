import torch
import os
import numpy as np
import h5py
from torch.utils.data import DataLoader
from torch.autograd import Variable
from kitti_data import KITTI
from prednet import PredNet
import torchvision

def save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    from PIL import Image
    im = Image.fromarray(np.rollaxis(tensor.numpy(), 0, 3))
    im.save(filename)

batch_size = 16
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
DATA_DIR = '/content/kitti_data'
test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')
nt = 10

kitti_test = KITTI(test_file, test_sources, nt)
test_loader = DataLoader(kitti_test, batch_size=batch_size, shuffle=False)

model = PredNet(R_channels, A_channels, output_mode='prediction')

# Load checkpoint correctly
checkpoint_path = '/content/drive/MyDrive/prednet_checkpoints/best_model.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
print(f"Training loss: {checkpoint['train_losses'][-1]:.4f}")
print(f"Validation loss: {checkpoint['val_losses'][-1]:.4f}")

if torch.cuda.is_available():
    print('Using GPU.')
    model.cuda()

model.eval()  # Set to evaluation mode

with torch.no_grad():  # No gradient computation needed for testing
    for i, inputs in enumerate(test_loader):
        inputs = inputs.permute(0, 1, 4, 2, 3)  # batch x time_steps x channel x width x height
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        origin = inputs.cpu().byte()[:, nt-1]
        print('origin:')
        print(type(origin))
        print(origin.size())
        
        print('predicted:')
        pred = model(inputs)
        pred = pred.cpu().byte()
        print(type(pred))
        print(pred.size())
        
        origin = torchvision.utils.make_grid(origin, nrow=4)
        pred = torchvision.utils.make_grid(pred, nrow=4)
        
        save_image(origin, 'origin.jpg')
        save_image(pred, 'predicted.jpg')
        break

print("Test complete! Check origin.jpg and predicted.jpg")
