'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np


class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super(BasicBlock, self).__init__()
        # Uncomment the following lines, replace the ? with correct values.
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
           self.shortcut = nn.Sequential(
               nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(planes)
           )

    def forward(self, x):
        # 1. Go through conv1, bn1, relu
        # 2. Go through conv2, bn
        # 3. Combine with shortcut output, and go through relu
        
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += self.shortcut(identity)
        x = F.relu(x)
        
        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Uncomment the following lines and replace the ? with correct values
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, images):
        """ input images and output logits """
        x = self.conv1(images)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    def visualize(self, logdir):
        print("\n ---------------------------- Visualizing Model Kernels ----------------------------- \n")
        
        os.makedirs(logdir, exist_ok=True)

        # Extract kernels from conv1 layer
        kernels = self.conv1.weight.detach().cpu().numpy()  # Shape: (64, 3, 3, 3)

        # Average across the 3 input channels
        kernels = np.mean(kernels, axis=1)

        # Normalize kernels for visualization
        min_val = kernels.min()
        max_val = kernels.max()
        kernels = (kernels - min_val) / (max_val - min_val)

        num_kernels = kernels.shape[0]
        num_rows = 8
        num_cols = 8
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
        axes = axes.flatten()

        for i in range(num_kernels):
            kernel = kernels[i] 
            axes[i].imshow(kernel, cmap="gray") 
            axes[i].axis("off")

        for i in range(num_kernels, len(axes)): 
            fig.delaxes(axes[i])

        file_path = os.path.join(logdir, "resnet18_kernel.png")
        i = 1
        while os.path.exists(file_path):
            file_path = os.path.join(logdir, f"resnet18_kernel_{i}.png")
            i += 1

        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Kernels saved to {file_path}")

        
        
