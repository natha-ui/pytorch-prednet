import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1,
                 dilation=1, groups=1, bias=True):
        super(ConvLSTMCell, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        kernel_size = _pair(kernel_size)
        stride      = _pair(stride)
        padding     = _pair(padding)
        dilation    = _pair(dilation)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.groups       = groups

        # Input-to-hidden and hidden-to-hidden: all four gates in one weight
        self.weight_ih = Parameter(torch.Tensor(4 * out_channels, in_channels // groups, *kernel_size))
        self.weight_hh = Parameter(torch.Tensor(4 * out_channels, out_channels // groups, *kernel_size))

        # Peephole connections: elementwise per channel, broadcast over spatial dims
        self.weight_ci = Parameter(torch.Tensor(out_channels))
        self.weight_cf = Parameter(torch.Tensor(out_channels))
        self.weight_co = Parameter(torch.Tensor(out_channels))

        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * out_channels))
            self.bias_hh = Parameter(torch.Tensor(4 * out_channels))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = 4 * self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

        self.weight_ih.data.uniform_(-stdv, stdv)
        self.weight_hh.data.uniform_(-stdv, stdv)
        self.weight_ci.data.uniform_(-stdv, stdv)
        self.weight_cf.data.uniform_(-stdv, stdv)
        self.weight_co.data.uniform_(-stdv, stdv)

        if self.bias_ih is not None:
            self.bias_ih.data.uniform_(-stdv, stdv)
            self.bias_hh.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h_0, c_0 = hx

        wx = F.conv2d(input, self.weight_ih, self.bias_ih,
                      self.stride, self.padding, self.dilation, self.groups)
        wh = F.conv2d(h_0, self.weight_hh, self.bias_hh,
                      self.stride, self.padding, self.dilation, self.groups)
        wxh = wx + wh  # (batch, 4*out_channels, H, W)

        # Peepholes: weight_c* are (out_channels,), broadcast over batch and spatial dims
        ci = self.weight_ci.view(1, -1, 1, 1) * c_0
        cf = self.weight_cf.view(1, -1, 1, 1) * c_0

        i   = torch.sigmoid(wxh[:, :self.out_channels]                             + ci)
        f   = torch.sigmoid(wxh[:, self.out_channels:2 * self.out_channels]         + cf)
        g   = torch.tanh(   wxh[:, 2 * self.out_channels:3 * self.out_channels])
        c_1 = f * c_0 + i * g

        # o peephole uses updated cell state
        co  = self.weight_co.view(1, -1, 1, 1) * c_1
        o   = torch.sigmoid(wxh[:, 3 * self.out_channels:]                          + co)

        h_1 = o * torch.tanh(c_1)
        return h_1, (h_1, c_1)
