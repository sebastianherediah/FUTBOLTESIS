"""Minimal HRNet implementation tailored to load the homography checkpoint."""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


BN_MOMENTUM = 0.1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def _make_layer(
    block: type[nn.Module],
    inplanes: int,
    planes: int,
    blocks: int,
    stride: int = 1,
) -> nn.Sequential:
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
        )

    layers = [block(inplanes, planes, stride, downsample)]
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches: int,
        block: type[nn.Module],
        num_blocks: Sequence[int],
        num_inchannels: Sequence[int],
        num_channels: Sequence[int],
        fuse_method: str,
        multi_scale_output: bool = True,
    ) -> None:
        super().__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = list(num_inchannels)
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers(num_branches, num_inchannels, num_channels, multi_scale_output)
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(
        self,
        num_branches: int,
        num_blocks: Sequence[int],
        num_inchannels: Sequence[int],
        num_channels: Sequence[int],
    ) -> None:
        if num_branches != len(num_blocks):
            raise ValueError("NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(num_branches, len(num_blocks)))
        if num_branches != len(num_channels):
            raise ValueError("NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(num_branches, len(num_channels)))
        if num_branches != len(num_inchannels):
            raise ValueError("NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(num_branches, len(num_inchannels)))

    def _make_one_branch(
        self,
        branch_index: int,
        block: type[nn.Module],
        num_blocks: Sequence[int],
        num_channels: Sequence[int],
    ) -> nn.Sequential:
        layers = []
        inplanes = self.num_inchannels[branch_index]
        for _ in range(num_blocks[branch_index]):
            layers.append(block(inplanes, num_channels[branch_index]))
            inplanes = num_channels[branch_index] * block.expansion
        self.num_inchannels[branch_index] = inplanes
        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches: int,
        block: type[nn.Module],
        num_blocks: Sequence[int],
        num_channels: Sequence[int],
    ) -> nn.ModuleList:
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(
        self,
        num_branches: int,
        num_inchannels: Sequence[int],
        num_channels: Sequence[int],
        multi_scale_output: bool,
    ) -> nn.ModuleList | None:
        if num_branches == 1:
            return None

        num_branches_output = num_branches if multi_scale_output else 1
        fuse_layers = []
        for i in range(num_branches_output):
            fuse_layer = []
            for j in range(num_branches):
                if j == i:
                    fuse_layer.append(None)
                elif j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        )
                    )
                else:
                    convs = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv = num_inchannels[i]
                            convs.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv, momentum=BN_MOMENTUM),
                                )
                            )
                        else:
                            num_outchannels_conv = num_inchannels[j]
                            convs.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv, momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*convs))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> List[int]:
        return self.num_inchannels

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse: List[Tensor] = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if self.fuse_layers[i][0] is None else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if self.fuse_layers[i][j] is None:
                    y = y + x[j]
                else:
                    z = x[j]
                    if j > i:
                        z = self.fuse_layers[i][j](z)
                        z = F.interpolate(z, size=y.shape[-2:], mode="bilinear", align_corners=False)
                        y = y + z
                    else:
                        y = y + self.fuse_layers[i][j](z)
            x_fuse.append(self.relu(y))

        return x_fuse


def _make_transition_layer(
    num_channels_pre_layer: Sequence[int],
    num_channels_cur_layer: Sequence[int],
) -> nn.ModuleList:
    num_branches_cur = len(num_channels_cur_layer)
    num_branches_pre = len(num_channels_pre_layer)

    transition_layers = []
    for i in range(num_branches_cur):
        if i < num_branches_pre:
            if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                transition_layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            num_channels_pre_layer[i],
                            num_channels_cur_layer[i],
                            3,
                            1,
                            1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                transition_layers.append(nn.Identity())
        else:
            convs = []
            in_channels = num_channels_pre_layer[-1]
            out_channels = num_channels_cur_layer[i]
            for k in range(i + 1 - num_branches_pre):
                if k == i - num_branches_pre:
                    convs.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels,
                                out_channels,
                                3,
                                2,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    convs.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels,
                                in_channels,
                                3,
                                2,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                in_channels = out_channels
            transition_layers.append(nn.Sequential(*convs))

    return nn.ModuleList(transition_layers)


class HighResolutionNet(nn.Module):
    """Subset of HRNet for keypoint detection."""

    def __init__(self) -> None:
        super().__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)

        self.layer1 = _make_layer(Bottleneck, self.inplanes, 64, blocks=4)
        stage1_out_channel = 64 * Bottleneck.expansion

        # Stage 2
        num_channels = [18, 36]
        block = BasicBlock
        num_channels = [c * block.expansion for c in num_channels]
        self.transition1 = _make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            num_modules=1,
            num_branches=2,
            block=block,
            num_blocks=[4, 4],
            num_channels=[18, 36],
            fuse_method="SUM",
        )

        # Stage 3
        num_channels = [18, 36, 72]
        num_channels = [c * block.expansion for c in num_channels]
        self.transition2 = _make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            num_modules=4,
            num_branches=3,
            block=block,
            num_blocks=[4, 4, 4],
            num_channels=[18, 36, 72],
            fuse_method="SUM",
        )

        # Stage 4
        num_channels = [18, 36, 72, 144]
        num_channels = [c * block.expansion for c in num_channels]
        self.transition3 = _make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, self.pre_stage_channels = self._make_stage(
            num_modules=3,
            num_branches=4,
            block=block,
            num_blocks=[4, 4, 4, 4],
            num_channels=[18, 36, 72, 144],
            fuse_method="SUM",
            multi_scale_output=True,
        )

    def _make_stage(
        self,
        num_modules: int,
        num_branches: int,
        block: type[nn.Module],
        num_blocks: Sequence[int],
        num_channels: Sequence[int],
        fuse_method: str,
        multi_scale_output: bool = True,
    ) -> tuple[nn.Sequential, List[int]]:
        modules = []
        num_inchannels = [c * block.expansion for c in num_channels]

        for i in range(num_modules):
            # only the last module outputs single scale if needed
            reset_multi_scale_output = multi_scale_output or i != num_modules - 1
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        low_level = x.clone()

        x = self.layer1(x)

        x_list = []
        for i in range(len(self.transition1)):
            if isinstance(self.transition1[i], nn.Identity):
                x_list.append(x)
            else:
                x_list.append(self.transition1[i](x))
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(len(self.transition2)):
            if i < len(y_list):
                if isinstance(self.transition2[i], nn.Identity):
                    x_list.append(y_list[i])
                else:
                    x_list.append(self.transition2[i](y_list[i]))
            else:
                x_list.append(self.transition2[i](y_list[-1]))
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(len(self.transition3)):
            if i < len(y_list):
                if isinstance(self.transition3[i], nn.Identity):
                    x_list.append(y_list[i])
                else:
                    x_list.append(self.transition3[i](y_list[i]))
            else:
                x_list.append(self.transition3[i](y_list[-1]))
        y_list = self.stage4(x_list)
        return low_level, y_list


__all__ = ["HighResolutionNet", "BasicBlock", "Bottleneck"]
