"""HRNet-based keypoint predictor used for homography inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


BN_MOMENTUM = 0.1


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches: int,
        block: type[nn.Module],
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        fuse_method: str = "SUM",
        multi_scale_output: bool = True,
    ) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.block = block
        self.fuse_method = fuse_method
        self.num_inchannels = num_inchannels
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches()
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index: int) -> nn.Sequential:
        layers = []
        downsample = None
        in_channels = self.num_inchannels[branch_index]
        out_channels = self.num_channels[branch_index] * self.block.expansion
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            )

        layers.append(self.block(in_channels, self.num_channels[branch_index], downsample=downsample))
        in_channels = out_channels
        for _ in range(1, self.num_blocks[branch_index]):
            layers.append(self.block(in_channels, self.num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self) -> nn.ModuleList:
        branches = []
        for branch_index in range(self.num_branches):
            branches.append(self._make_one_branch(branch_index))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self) -> nn.ModuleList | None:
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = [c * self.block.expansion for c in self.num_channels]

        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_inchannels[j], num_inchannels[i], kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    convs: List[nn.Sequential] = []
                    for k in range(i - j):
                        in_channels = num_inchannels[j]
                        out_channels = num_inchannels[j] if k != i - j - 1 else num_inchannels[i]
                        convs.append(
                            nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True) if k != i - j - 1 else nn.Identity(),
                            )
                        )
                    fuse_layer.append(nn.Sequential(*convs))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](inputs[0])]

        outputs = []
        for branch, x in zip(self.branches, inputs):
            outputs.append(branch(x))

        fused = []
        for i in range(len(self.fuse_layers)):
            y = outputs[i]
            for j in range(self.num_branches):
                if i == j:
                    continue
                if j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](outputs[j]),
                        size=outputs[i].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    y = y + self.fuse_layers[i][j](outputs[j])
            fused.append(self.relu(y))
        return fused


def _make_transition_layer(
    prev_channels: List[int],
    curr_channels: List[int],
) -> nn.ModuleList:
    num_branches_prev = len(prev_channels)
    num_branches_curr = len(curr_channels)
    transition_layers: List[nn.Module | None] = []

    for i in range(num_branches_curr):
        if i < num_branches_prev:
            if curr_channels[i] != prev_channels[i]:
                transition_layers.append(
                    nn.Sequential(
                        nn.Conv2d(prev_channels[i], curr_channels[i], kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(curr_channels[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                transition_layers.append(None)
        else:
            convs = []
            in_channels = prev_channels[-1]
            for j in range(i + 1 - num_branches_prev):
                out_channels = curr_channels[i] if j == i - num_branches_prev else in_channels
                convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    )
                )
                in_channels = out_channels
            transition_layers.append(nn.Sequential(*convs))
    return nn.ModuleList(transition_layers)


def _make_stage(
    stage_config: Dict[str, Any],
    num_inchannels: List[int],
    multi_scale_output: bool = True,
) -> Tuple[nn.Sequential, List[int]]:
    num_modules = stage_config["num_modules"]
    num_branches = stage_config["num_branches"]
    num_blocks = stage_config["num_blocks"]
    num_channels = stage_config["num_channels"]
    block = BasicBlock if stage_config["block_type"] == "BASIC" else Bottleneck

    modules = []
    for i in range(num_modules):
        reset_multi_scale = multi_scale_output if i == num_modules - 1 else True
        module = HighResolutionModule(
            num_branches=num_branches,
            block=block,
            num_blocks=num_blocks,
            num_inchannels=num_inchannels,
            num_channels=num_channels,
            multi_scale_output=reset_multi_scale,
        )
        modules.append(module)
        num_inchannels = [c * block.expansion for c in num_channels]
    return nn.Sequential(*modules), num_inchannels


HRNET_W18_CONFIG: Dict[str, Any] = {
    "stem_width": 64,
    "stage1": {
        "num_modules": 1,
        "num_branches": 1,
        "block_type": "BOTTLENECK",
        "num_blocks": [4],
        "num_channels": [64],
    },
    "stage2": {
        "num_modules": 1,
        "num_branches": 2,
        "block_type": "BASIC",
        "num_blocks": [4, 4],
        "num_channels": [18, 36],
    },
    "stage3": {
        "num_modules": 4,
        "num_branches": 3,
        "block_type": "BASIC",
        "num_blocks": [4, 4, 4],
        "num_channels": [18, 36, 72],
    },
    "stage4": {
        "num_modules": 3,
        "num_branches": 4,
        "block_type": "BASIC",
        "num_blocks": [4, 4, 4, 4],
        "num_channels": [18, 36, 72, 144],
    },
    "upscale": 2,
    "internal_final_conv": 0,
    "final_conv_kernel": 1,
}


class HighResolutionNet(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        stem_width = config["stem_width"]
        self.conv1 = nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_width, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(stem_width, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        stage1_cfg = config["stage1"]
        block1 = Bottleneck if stage1_cfg["block_type"] == "BOTTLENECK" else BasicBlock
        num_channels1 = stage1_cfg["num_channels"][0]
        self.layer1 = self._make_layer(block1, stem_width, num_channels1, stage1_cfg["num_blocks"][0])
        stage1_out_channels = [num_channels1 * block1.expansion]

        stage2_cfg = config["stage2"]
        block2 = BasicBlock if stage2_cfg["block_type"] == "BASIC" else Bottleneck
        stage2_channels = [c * block2.expansion for c in stage2_cfg["num_channels"]]
        self.transition1 = _make_transition_layer(stage1_out_channels, stage2_channels)
        self.stage2, pre_stage_channels = _make_stage(stage2_cfg, stage2_channels)

        stage3_cfg = config["stage3"]
        block3 = BasicBlock if stage3_cfg["block_type"] == "BASIC" else Bottleneck
        stage3_channels = [c * block3.expansion for c in stage3_cfg["num_channels"]]
        self.transition2 = _make_transition_layer(pre_stage_channels, stage3_channels)
        self.stage3, pre_stage_channels = _make_stage(stage3_cfg, stage3_channels)

        stage4_cfg = config["stage4"]
        block4 = BasicBlock if stage4_cfg["block_type"] == "BASIC" else Bottleneck
        stage4_channels = [c * block4.expansion for c in stage4_cfg["num_channels"]]
        self.transition3 = _make_transition_layer(pre_stage_channels, stage4_channels)
        self.stage4, self.pre_stage_channels = _make_stage(stage4_cfg, stage4_channels, multi_scale_output=True)

        self.upscale = config.get("upscale", 1)
        self.last_inp_channels = sum(self.pre_stage_channels)
        if self.upscale > 1:
            self.last_inp_channels += stem_width

    def _make_layer(
        self,
        block: type[nn.Module],
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        expanded_out = out_channels * block.expansion
        if stride != 1 or in_channels != expanded_out:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, expanded_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(expanded_out, momentum=BN_MOMENTUM),
            )

        layers = [block(in_channels, out_channels, stride, downsample)]
        in_channels = expanded_out
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        stem = self.conv2(x)
        stem = self.bn2(stem)
        stem = self.relu(stem)

        x = self.layer1(stem)

        x_list = []
        for i in range(len(self.transition1)):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(len(self.transition2)):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[i if i < len(y_list) else -1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(len(self.transition3)):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[i if i < len(y_list) else -1]))
            else:
                x_list.append(y_list[i])
        branch_outputs = self.stage4(x_list)

        up_h = int(branch_outputs[0].shape[2] * self.upscale)
        up_w = int(branch_outputs[0].shape[3] * self.upscale)

        features: List[torch.Tensor] = []
        if self.upscale > 1:
            features.append(F.interpolate(stem, size=(up_h, up_w), mode="bilinear", align_corners=False))

        for branch in branch_outputs:
            if branch.shape[-2:] != (up_h, up_w):
                features.append(F.interpolate(branch, size=(up_h, up_w), mode="bilinear", align_corners=False))
            else:
                features.append(branch)

        concatenated = torch.cat(features, dim=1)
        return branch_outputs, concatenated


class HRNetHeatmapModel(nn.Module):
    """HRNet backbone with a lightweight prediction head for keypoint heatmaps."""

    def __init__(self, num_output_channels: int, config: Dict[str, Any] | None = None) -> None:
        super().__init__()
        config = config or HRNET_W18_CONFIG
        self.backbone = HighResolutionNet(config)

        if config.get("internal_final_conv", 0):
            internal_channels = config["internal_final_conv"]
            self.prediction_head = nn.Sequential(
                nn.Conv2d(self.backbone.last_inp_channels, internal_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(internal_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(internal_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    internal_channels,
                    num_output_channels,
                    kernel_size=config.get("final_conv_kernel", 1),
                    padding=1 if config.get("final_conv_kernel", 1) == 3 else 0,
                ),
            )
        else:
            final_kernel = config.get("final_conv_kernel", 1)
            self.prediction_head = nn.Sequential(
                nn.Conv2d(self.backbone.last_inp_channels, self.backbone.last_inp_channels, kernel_size=1, bias=True),
                nn.BatchNorm2d(self.backbone.last_inp_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.backbone.last_inp_channels,
                    num_output_channels,
                    kernel_size=final_kernel,
                    padding=1 if final_kernel == 3 else 0,
                    bias=True,
                ),
            )

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.num_output_channels = num_output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, features = self.backbone(x)
        logits = self.prediction_head(features)
        return self.log_softmax(logits)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device | str | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> HRNetHeatmapModel:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    num_channels = None
    if "num_output_channels" in checkpoint:
        num_channels = int(checkpoint["num_output_channels"])

    if num_channels is None:
        candidate_keys = [
            k for k in state_dict.keys() if k.startswith("prediction_head") and k.endswith("weight")
        ]
        head_weight_key = None
        if candidate_keys:
            # Select the layer with the highest index (e.g. prediction_head.3.weight)
            head_weight_key = max(
                candidate_keys,
                key=lambda key: int(key.split(".")[1]) if key.split(".")[1].isdigit() else -1,
            )
        if head_weight_key is None:
            raise RuntimeError("No se pudo determinar el número de canales de salida del head de predicción.")
        num_channels = state_dict[head_weight_key].shape[0]

    model = HRNetHeatmapModel(num_output_channels=int(num_channels))
    model.load_state_dict(state_dict)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model
