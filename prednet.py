import torch
import torch.nn as nn
from torch.nn import functional as F
from convlstmcell import ConvLSTMCell


class SatLU(nn.Module):
    """Saturating linear unit: clamps output to [lower, upper]."""
    def __init__(self, lower=0, upper=255):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper)

    def __repr__(self):
        return f'{self.__class__.__name__}(min_val={self.lower}, max_val={self.upper})'


class PredNet(nn.Module):
    def __init__(self, R_channels, A_channels, output_mode='error'):
        super(PredNet, self).__init__()
        assert len(R_channels) == len(A_channels), 'R_channels and A_channels must have the same length'

        valid_output_modes = ('prediction', 'error')
        if output_mode not in valid_output_modes:
            raise ValueError(f'Invalid output_mode "{output_mode}". Must be one of {valid_output_modes}')

        self.r_channels  = R_channels
        self.a_channels  = A_channels
        self.n_layers    = len(R_channels)
        self.output_mode = output_mode

        # ConvLSTM cells: input is E (2*A) concatenated with upsampled R from layer above
        # Top layer has no layer above, so input is just E
        r_channels_above = R_channels[1:] + (0,)
        for l in range(self.n_layers):
            cell = ConvLSTMCell(2 * A_channels[l] + r_channels_above[l], R_channels[l], kernel_size=3)
            setattr(self, f'cell{l}', cell)

        # Prediction convolutions: R -> A_hat
        for l in range(self.n_layers):
            conv = nn.Sequential(
                nn.Conv2d(R_channels[l], A_channels[l], kernel_size=3, padding=1),
                nn.ReLU(),
            )
            if l == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, f'conv{l}', conv)

        # A update convolutions: E -> A for next layer up (with pooling)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)
        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(
                nn.Conv2d(2 * A_channels[l], A_channels[l + 1], kernel_size=3, padding=1),
                self.maxpool,
            )
            setattr(self, f'update_A{l}', update_A)

        self.reset_parameters()

    def reset_parameters(self):
        for l in range(self.n_layers):
            getattr(self, f'cell{l}').reset_parameters()

    def forward(self, input):
        # input: (batch, time_steps, channels, H, W)
        batch_size, time_steps = input.size(0), input.size(1)
        H, W = input.size(-2), input.size(-1)
        device = input.device

        # Initialise E and R state to zeros
        E_seq = []
        R_seq = []
        H_seq = [None] * self.n_layers
        for l in range(self.n_layers):
            ds = 2 ** l
            E_seq.append(torch.zeros(batch_size, 2 * self.a_channels[l], H // ds, W // ds, device=device))
            R_seq.append(torch.zeros(batch_size, self.r_channels[l],     H // ds, W // ds, device=device))

        total_error = []

        for t in range(time_steps):
            A = input[:, t].float()

            # Top-down pass: update R states
            for l in reversed(range(self.n_layers)):
                cell = getattr(self, f'cell{l}')
                hx = H_seq[l] if H_seq[l] is not None else (R_seq[l], R_seq[l])

                if l == self.n_layers - 1:
                    lstm_input = E_seq[l]
                else:
                    lstm_input = torch.cat([E_seq[l], self.upsample(R_seq[l + 1])], dim=1)

                R_seq[l], H_seq[l] = cell(lstm_input, hx)

            # Bottom-up pass: compute predictions and errors
            for l in range(self.n_layers):
                A_hat = getattr(self, f'conv{l}')(R_seq[l])

                if l == 0:
                    frame_prediction = A_hat

                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E_seq[l] = torch.cat([pos, neg], dim=1)

                if l < self.n_layers - 1:
                    A = getattr(self, f'update_A{l}')(E_seq[l])

            if self.output_mode == 'error':
                mean_error = torch.cat(
                    [e.flatten(1).mean(1, keepdim=True) for e in E_seq], dim=1
                )
                total_error.append(mean_error)

        if self.output_mode == 'error':
            return torch.stack(total_error, dim=2)  # (batch, n_layers, time_steps)
        else:
            return frame_prediction
