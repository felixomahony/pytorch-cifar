import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

class LiftingLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        n_groups=3,
        verbose=False,
        name=None,
        bias=True,) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.verbose = verbose
        self.bias = bias
        self.name = name if name is not None else "GroupConv"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        conv_layer = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=bias,
        )
        conv_weight_base = conv_layer.weight.view(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        conv_bias_base = conv_layer.bias if bias else torch.zeros(self.out_channels)

        self.conv_bias_base = conv_bias_base.to(self.device)
        self.conv_weight_base = conv_weight_base.to(self.device)

        self.conv_weight_base = nn.Parameter(conv_weight_base)
        self.conv_bias_base = nn.Parameter(conv_bias_base)

    def composit_weight(self):

        conv_weight = [
            self.conv_weight_base.roll(k, dims=1) for k in range(self.n_groups)
        ]
        conv_weight = torch.stack(conv_weight, dim=0)
        conv_weight = conv_weight.view(
            self.n_groups * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )

        conv_bias = torch.stack(
            [self.conv_bias_base for _ in range(self.n_groups)], dim=0
        ).flatten()

        return conv_weight, conv_bias

    def forward(self, x):
        # incoming tensor is of shape (batch_size, 3, height, width)
        # outgoing tensor should be of shape (batch_size, n_groups * n_channels, height, width)
        conv_weight, conv_bias = self.composit_weight()

        y = F.conv2d(
            x,
            weight=conv_weight,
            bias=conv_bias if self.bias else None,
            padding=self.padding,
            stride=self.stride,
        )

        return y


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
    
    def forward(self, x):
        """
        incoming tensor is of shape (batch_size, n_groups * in_channels, height, width)
        outgoing tensor should be of shape (batch_size, n_groups * out_channels, height, width)
        """
        out_tensors = []
        for i in range(self.n_groups):
            out_tensors.append(self.conv_layer(x))
            x = x.view(-1, self.n_groups, self.in_channels, x.shape[-2], x.shape[-1])
            x = x.roll(-1, dims=1)
            x = x.view(-1, self.n_groups * self.in_channels, x.shape[-2], x.shape[-1])

        out_tensors = torch.stack(out_tensors, dim=1)
        out_tensors = out_tensors.view(-1, self.n_groups * self.out_channels, out_tensors.shape[-2], out_tensors.shape[-1])

        return out_tensors


class LegacyGroupConv(nn.Module):
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
        filter_operation=(lambda x, k: x),
        verbose=False,
        name=None,
        bias=True,
    ) -> None:
        super().__init__()

        self.n_groups = n_groups
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.verbose = verbose
        self.filter_operation = filter_operation
        self.name = name if name is not None else "GroupConv"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bias = bias

        conv_layer = nn.Conv2d(
            self.n_groups * self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=bias,
        )
        conv_weight_base = conv_layer.weight.view(
            self.out_channels,
            self.n_groups,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        conv_bias_base = conv_layer.bias if bias else torch.zeros(self.out_channels)

        self.conv_weight_base = conv_weight_base.to(self.device)
        self.conv_bias_base = conv_bias_base.to(self.device)

        self.conv_weight_base = nn.Parameter(conv_weight_base)
        self.conv_bias_base = nn.Parameter(conv_bias_base)

    def composit_weight(self):
        conv_weight = [
            self.filter_operation(self.conv_weight_base, k).roll(k, dims=1)
            for k in range(self.n_groups)
        ]  # shape (n_groups, out_channels, n_groups, in_channels, kernel_size, kernel_size)
        conv_weight = torch.stack(
            conv_weight, dim=0
        )  # shape (n_groups, out_channels, n_groups, in_channels, kernel_size, kernel_size)

        conv_weight = conv_weight.view(
            self.n_groups * self.out_channels,
            self.n_groups * self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )  # shape (n_groups * n_groups_d2 * out_channels, n_groups * n_groups_d2 * in_channels, kernel_size, kernel_size)

        conv_bias = torch.stack(
            [self.conv_bias_base for _ in range(self.n_groups)], dim=0
        ).flatten()

        return conv_weight, conv_bias

    def forward(self, x):
        verbose_print(x, self.verbose, "{} input:".format(self.name))
        # incoming tensor is of shape (batch_size, n_groups * in_channels, height, width)
        # outgoing tensor should be of shape (batch_size, n_groups * out_channels, height, width)

        conv_weight, conv_bias = self.composit_weight()

        y = F.conv2d(
            x,
            weight=conv_weight,
            bias=conv_bias if self.bias else None,
            padding=self.padding,
            stride=self.stride,
        )

        verbose_print(y, self.verbose, "{} output:".format(self.name))
        return y


class LiftConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        n_groups=4,
        filter_operation=(lambda x, k: x),
        verbose=False,
        name=None,
    ) -> None:
        super().__init__()

        self.n_groups = n_groups
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.verbose = verbose
        self.filter_operation = filter_operation
        self.name = name if name is not None else "LiftConv"

        conv_layer = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        self.conv_weight = conv_layer.weight
        self.conv_bias = conv_layer.bias

    def composit_weight(self):
        # First reform conv tensor
        conv_weight = [
            self.filter_operation(self.conv_weight, k) for k in range(self.n_groups)
        ]

        conv_weight = torch.stack(conv_weight, dim=0)

        conv_weight = conv_weight.view(
            self.n_groups * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )

        conv_bias = torch.stack(
            [self.conv_bias for _ in range(self.n_groups)], dim=0
        ).flatten()

        return conv_weight, conv_bias

    def forward(self, x):
        verbose_print(x, self.verbose, "{} input:".format(self.name))
        # incoming tensor is of shape (batch_size, in_channels, height, width)
        # outgoing tensor should be of shape (batch_size, n_groups, out_channels, height, width)
        conv_weight, conv_bias = self.composit_weight()
        y = F.conv2d(
            x,
            weight=conv_weight,
            bias=conv_bias,
            padding=self.padding,
            stride=self.stride,
        )
        verbose_print(y, self.verbose, "{} output:".format(self.name))
        return y


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
    def __init__(self, num_features, n_groups=4, momentum = 0.0):
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(num_features, momentum=momentum)
        self.num_features = num_features
        self.n_groups = n_groups

    def forward(self, x):
        """
        incoming tensor is of shape (batch_size, n_groups * channels, height, width)"""
        if x.shape[1] != self.n_groups * self.num_features:
            raise ValueError(
                f"Expected {self.n_groups * self.num_features} channels in tensor, but got {x.shape[1]} channels"
            )
        x = x.view(
            -1, self.n_groups, x.shape[-3] // self.n_groups, x.shape[-2], x.shape[-1]
        )
        x = x.permute(0, 2, 1, 3, 4)
        y = self.batch_norm(x)
        y = y.permute(0, 2, 1, 3, 4)
        y = y.reshape(-1, self.n_groups * self.num_features, x.shape[-2], x.shape[-1])
        return y


class GroupBatchNorm2d_legacy(nn.Module):
    def __init__(
        self,
        num_features,
        n_groups,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ) -> None:
        super().__init__()
        self.BatchNorm2d = nn.BatchNorm2d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.n_groups = n_groups

    def forward(self, x):
        # incoming tensor is of shape (batch_size, n_groups * channels, height, width)
        # outgoing tensor should be of shape (batch_size, n_groups, channels, height, width)
        # x = x.view(-1, self.n_groups * x.shape[2], x.shape[-2], x.shape[-1])

        x = self.BatchNorm2d(
            x.view(-1, x.shape[-3] // self.n_groups, x.shape[-2], x.shape[-1])
        )
        return x.view(-1, self.n_groups * x.shape[-3], x.shape[-2], x.shape[-1])


class Lifting(nn.Module):
    """
    Lifting Layer
    -------------
    This layer is a lifting layer. Its role is to raise the input tensor from the original space to a larger n dimensional group space.
    """

    def __init__(self, n_groups, space_transform, verbose=False) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.space_transform = space_transform
        self.verbose = verbose

    def forward(self, x):
        # incoming tensor is of shape (batch_size, 3, height, width)
        # outgoing tensor should be of shape (batch_size, n_groups, height, width)

        if len(x) != self.n_groups:
            raise ValueError(
                f"Expected {self.n_groups} images in tensor, but got {len(x)} images"
            )

        rotated_tensors = [self.space_transform(x[j]) for j in range(self.n_groups)]
        rotated_tensors = torch.stack(
            rotated_tensors, dim=1
        )  # shape (batch_size, n_groups, h, w)
        rotated_tensors = rotated_tensors.unsqueeze(
            2
        )  # shape (batch_size, n_groups, 1, h, w) 1 stands in for the channel dimension
        return rotated_tensors


def spatial_rotation(x, k):
    return torch.rot90(x, k, dims=(-2, -1))


def verbose_print(x, verbose, name):
    if verbose:
        print(name, x.shape)
    return x

if __name__=="__main__":
    test_input = torch.randn(1, 4*3, 32, 32)
    group_conv = GroupConv(3, 8, 3, n_groups=4)
    group_conv(test_input)