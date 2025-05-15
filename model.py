# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import math
import torch
from torch import nn
import torch.nn.functional as F

class EdgeGuidedLaplacian(nn.Module):
    def __init__(self, threshold=0.3, laplace_weight=0.5):
        super(EdgeGuidedLaplacian, self).__init__()
        self.threshold = threshold
        self.laplace_weight = laplace_weight

        # Sobel filters
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_x.weight.data.copy_(torch.tensor(sobel_x, dtype=torch.float32).view(1, 1, 3, 3))
        self.sobel_y.weight.data.copy_(torch.tensor(sobel_y, dtype=torch.float32).view(1, 1, 3, 3))

        # Laplacian filter
        lap_kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        self.laplace = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.laplace.weight.data.copy_(torch.tensor(lap_kernel, dtype=torch.float32).view(1, 1, 3, 3))

        # Freeze filters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_mag = grad_mag / (grad_mag.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)  # Normalize
        mask = (grad_mag > self.threshold).float()
        lap = self.laplace(x)
        enhanced = x + self.laplace_weight * (lap * mask)
        return enhanced.clamp(0, 1)

class SRCNN(nn.Module):
    def __init__(self, use_enhancer=True) -> None:
        super(SRCNN, self).__init__()
        self.use_enhancer = use_enhancer

        self.enhancer = EdgeGuidedLaplacian(threshold=0.3, laplace_weight=0.5)

        self.sobel = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        sobel_kernel = torch.tensor([[[-1, 0, 1],
                                      [-2, 0, 2],
                                      [-1, 0, 1]]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel.weight.data.copy_(sobel_kernel)
        self.sobel.weight.requires_grad = False

        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.ReLU(True)
        )

        self.map = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(True)
        )

        self.reconstruction = nn.Conv2d(32, 1, kernel_size=5, padding=2)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_enhancer:
            x = self.enhancer(x)

        edge = self.sobel(x)
        x_cat = torch.cat([x, edge], dim=1)
        feat = self.features(x_cat)
        feat = self.map(feat)
        out = self.reconstruction(feat)
        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0,
                                math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                if module.bias is not None:
                    nn.init.zeros_(module.bias.data)

        if self.reconstruction.bias is not None:
            nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
            nn.init.zeros_(self.reconstruction.bias.data)
        else:
            nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
