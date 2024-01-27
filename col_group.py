"""
© Felix O'Mahony
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import time


from collections.abc import Iterable

"""
This namespace is for the architecture of a (generalised) group convolutional network.

All classes in the network inherit nn.Module so that they may be included into a generalised PyTorch network.
In general modules accept and return data in the form (batch size, num groups * channels, *spatial dimensions) except special cases e.g. lifting.
The tensors are not separated into (batch size, num groups, channels, *spatial dimensions) because this allows the tensors to work with
functions which are not affected by group convolution (e.g. a 2D pool).

There are essentially X import classes in this namespace:

----------------
1. GroupConv
This group performs the group convolution.
The operation applied to the convolution filter between layers is defined by the filter_operation argument.
This should be a function with two inputs, x and k. x is the filter and k is the index of the filter which is being modified. 
$k \in \{0, \dots , n_groups - 1\}$.
$x \in \mathcal{R} ^{out channels, num groups, in channels, kernel size, kernel size}$
It should return a tensor with the same shape as x.
The default filter operation is to leave the filter unchanged (i.e. the identity operation).

By way of example, a function is defined spatial_rotation which performs the (standard) k * 90 degree spatial rotation of the filter.
This would be the appropriate filter for a spatial rotation equivariant group convolution.

----------------
2. GroupPool
This performs the final group pooling.

The method is a max pool, although this can be changed by editing the pool_operation function.

----------------

"""

class GroupConvHL(nn.Module):
    """
    Group Convolution Layer with hue and luminance group equivariance
    -----------------------
    This is a layer to be used within the global hue and luminance equivariance network. It is relatively simple, since no geometric transformation of the input tensor must take place. Rather, the input tensor has its group channels permuted so that each possible permutation of the color space (in hue, which we think of as rotation) and luminance (which we think of as scaling/permuting around radii of groups) occurs.

    This is described fully in the published paper.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        n_groups=4,
        n_groups_luminance=3,
        bias = False,
        rescale_luminance = True,
        ) -> None:
        super().__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if type(self.kernel_size) != int:
            self.kernel_size = self.kernel_size[0]  # we only permit square kernels

        self.n_groups = n_groups
        self.n_groups_luminance = n_groups_luminance

        self.bias = bias

        self.rescale_luminance = rescale_luminance

        self.conv_layer = nn.Conv2d(
            self.n_groups * self.n_groups_luminance * self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=bias, # it should be noted that bias = True is untested (since only resnets are considered), and may not function as expected.
        )
        
    
    def forward(self, x):
        """
        incoming tensor is of shape (batch_size, n_groups * n_groups_luminance * in_channels, height, width)
        outgoing tensor should be of shape (batch_size, n_groups * n_groups_luminance * out_channels, height, width)
        """
        # reshape input tensor to shape appropriate for transforming according to hlgcnn
        x = x.view(-1, self.n_groups, self.n_groups_luminance, self.in_channels, x.shape[-2], x.shape[-1])
        out_tensors = []
        for i in range(self.n_groups):
            for j in range(self.n_groups_luminance):
                roll = j - self.n_groups_luminance // 2
                # remodel y to appropriately model x at this luminance
                y = x.roll(roll, dims=2)
                # set y values to zero where rolling pushes them through to other side
                if roll  > 0:
                    y[:, :, :roll, :, :, :] = torch.zeros_like(y[:, :, :roll, :, :, :])
                elif roll < 0:
                    y[:, :, roll:, :, :, :] = torch.zeros_like(y[:, :, roll:, :, :, :])
                    

                # Apply network
                # first we must reshape x to our target input
                y = y.view(-1, self.n_groups * self.n_groups_luminance * self.in_channels, x.shape[-2], x.shape[-1])
                z = self.conv_layer(y)
                if self.rescale_luminance:
                    luminance_sf = (self.n_groups_luminance - abs(roll)) / self.n_groups_luminance
                    z /= luminance_sf
                out_tensors.append(z)
            x = x.roll(-1, dims=1)

        out_tensors = torch.stack(out_tensors, dim=1)
        out_tensors = out_tensors.view(-1, self.n_groups * self.n_groups_luminance * self.out_channels, out_tensors.shape[-2], out_tensors.shape[-1])

        
        # New method
        # conv_weights = []
        # for i in range(self.n_groups):
        #     for j in range(self.n_groups_luminance):
        #         roll = self.n_groups_luminance // 2 - j
        #         weight_reviewed = self.conv_layer.weight.data.view(self.out_channels, self.n_groups, self.n_groups_luminance,  self.in_channels, self.conv_layer.weight.shape[-2], self.conv_layer.weight.shape[-1])
        #         weight_reviewed = weight_reviewed.roll(i, dims=1)
        #         weight_reviewed = weight_reviewed.roll(roll, dims=2)
        #         if roll > 0:
        #             weight_reviewed[:, :, :roll, :, :, :] = torch.zeros_like(weight_reviewed[:, :, :roll, :, :, :])
        #         elif roll < 0:
        #             weight_reviewed[:, :, roll:, :, :, :] = torch.zeros_like(weight_reviewed[:, :, roll:, :, :, :])
        #         weight_reviewed = weight_reviewed.view(self.out_channels, self.n_groups * self.n_groups_luminance * self.in_channels, self.conv_layer.weight.shape[-2], self.conv_layer.weight.shape[-1])
        #         conv_weights.append(weight_reviewed)
        # weight = torch.stack(conv_weights, dim=0)
        # weight = weight.view(self.n_groups * self.n_groups_luminance * self.out_channels, self.n_groups * self.n_groups_luminance * self.in_channels, self.conv_layer.weight.shape[-2], self.conv_layer.weight.shape[-1])
        # out_tensors = F.conv2d(x, weight, stride=self.stride, padding=self.padding)
        
        return out_tensors


class GroupConv(nn.Module):
    """
    Group Convolution Layer
    -----------------------
    This layer is a group convolution layer. Its role is to perform a convolution on the input tensor in the group space.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        n_groups=4,
        bias = False,
        ) -> None:
        super().__init__()


        self.n_groups = n_groups
        self.kernel_size = kernel_size
        if type(self.kernel_size) != int:
            self.kernel_size = self.kernel_size[0]
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.conv_layer = nn.Conv2d(
            self.n_groups * self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=bias,
        )


    
    def forward_1(self, x):
        # """
        # incoming tensor is of shape (batch_size, n_groups * in_channels, height, width)
        # outgoing tensor should be of shape (batch_size, n_groups * out_channels, height, width)
        # """
        out_tensors = []
        
        for i in range(self.n_groups):
            out_tensors.append(self.conv_layer(x))
            x = x.view(-1, self.n_groups, self.in_channels, x.shape[-2], x.shape[-1])
            x = x.roll(-1, dims=1)
            x = x.view(-1, self.n_groups * self.in_channels, x.shape[-2], x.shape[-1])

        out_tensors = torch.stack(out_tensors, dim=1)
        out_tensors = out_tensors.view(-1, self.n_groups * self.out_channels, out_tensors.shape[-2], out_tensors.shape[-1])
        return out_tensors

    def forward_2(self, x):

        # New method
        conv_weight = torch.zeros((self.n_groups, self.out_channels, self.n_groups, self.in_channels, self.kernel_size, self.kernel_size), dtype=self.conv_layer.weight.dtype)
        # put on same device as x
        conv_weight = conv_weight.to(x.device)
        for i in range(self.n_groups):
            conv_weight[i, :, :, :, :, :] = self.conv_layer.weight.data.view(self.out_channels, self.n_groups, self.in_channels, self.kernel_size, self.kernel_size).roll(i, dims=1)
            conv_weight = self.conv_weight.view(self.n_groups * self.out_channels, self.n_groups * self.in_channels, self.kernel_size, self.kernel_size)
            out_tensors = F.conv2d(x, conv_weight, stride=self.stride, padding=self.padding)
        
        return out_tensors
    
    def forward(self, x):
        return self.forward_2(x)

class GroupPool(nn.Module):
    def __init__(
        self, n_groups, pool_operation=lambda x: torch.max(x, dim=1)[0], verbose=False, name=None
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.n_groups = n_groups
        if verbose:
            print("verbose is sent to True in Pooling Layer")
        self.name = name if name is not None else "GroupPool"
        self.pool_operation = pool_operation

    def forward(self, x):
        verbose_print(x, self.verbose, "{} input:".format(self.name))

        x = x.view(
            -1, self.n_groups, x.shape[1] // self.n_groups, x.shape[2], x.shape[3]
        )

        # incoming tensor is of shape (batch_size, n_groups * channels, height, width)
        # outgoing tensor should be of shape (batch_size, channels, height, width)
        y = self.pool_operation(x)

        verbose_print(y, self.verbose, "{} output:".format(self.name))
        return y


class GroupBatchNorm2d(nn.Module):
    def __init__(self, num_features, n_groups=4, n_groups_luminance=1, momentum = 0.0):
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(num_features, momentum=momentum)
        self.num_features = num_features
        self.n_groups = n_groups
        self.n_groups_luminance = n_groups_luminance

    def forward(self, x):
        """
        incoming tensor is of shape (batch_size, n_groups * channels, height, width)"""
        if x.shape[1] != self.n_groups * self.n_groups_luminance * self.num_features:
            raise ValueError(
                f"Expected {self.n_groups * self.n_groups_luminance * self.num_features} channels in tensor, but got {x.shape[1]} channels"
            )
        x = x.view(
            -1, self.n_groups, x.shape[-3] // (self.n_groups * self.n_groups_luminance), x.shape[-2], x.shape[-1]
        )
        x = x.permute(0, 2, 1, 3, 4)
        y = self.batch_norm(x)
        y = y.permute(0, 2, 1, 3, 4)
        y = y.reshape(-1, (self.n_groups * self.n_groups_luminance) * self.num_features, x.shape[-2], x.shape[-1])
        return y

def spatial_rotation(x, k):
    return torch.rot90(x, k, dims=(-2, -1))


def verbose_print(x, verbose, name):
    if verbose:
        print(name, x.shape)
    return x

if __name__=="__main__":
    # NB input tensor has shape (batch, group * channels, w, h)
    test_input = torch.randn(1, 4*3, 32, 32)
    group_conv = GroupConv(3, 8, 3, n_groups=4)
    group_conv(test_input)