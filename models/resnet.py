import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

# Linear Scheduler for Skip Connection
def linear_scheduler(step, total_steps, start=1.0, end=0.0):
    return max(end, start - (start - end) * (step / total_steps))

# Cosine Scheduler for Skip Connection
def cosine_scheduler(step, total_steps, start=1.0, end=0.0):
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * step / total_steps))

# Unified Bottleneck with configurable skip connection scheduler
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, scheduler_type='linear', total_epochs=100, final_skip=1.0, update_per_batch=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.skip_scale = 1  # Initialized to 1
        self.scheduler_type = scheduler_type
        self.total_epochs = total_epochs
        self.final_skip = final_skip
        self.update_per_batch = update_per_batch
    
    def update_skip_scale(self, step, total_steps, start_value=None):
        if start_value is None:
            start_value = 1.0

        if self.scheduler_type == 'linear':
            self.skip_scale = linear_scheduler(step, total_steps, start=start_value, end=self.final_skip)
        elif self.scheduler_type == 'cosine':
            self.skip_scale = cosine_scheduler(step, total_steps, start=start_value, end=self.final_skip)
        elif self.scheduler_type == 'none':
            self.skip_scale = 1.0
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")


    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply scaled skip connection
        out += self.skip_scale * identity
        out = F.relu(out)

        return out


#Bottleneck with BN after skip connection 
class Bottleneck_bn(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, scheduler_type='linear', total_epochs=100, final_skip=1.0):
        super(Bottleneck_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.bn4 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.skip_scale = 1  # Initialized to 1
        self.scheduler_type = scheduler_type
        self.total_epochs = total_epochs
        self.final_skip = final_skip
    
    def update_skip_scale(self, epoch, total_epochs, start_value=None):
        """Updates the skip scale dynamically during training."""
        if start_value is None:
            start_value = 1.0  # Normal training start

        if self.scheduler_type == 'linear':
            self.skip_scale = linear_scheduler(epoch, total_epochs, start=start_value, end=self.final_skip)
        elif self.scheduler_type == 'cosine':
            self.skip_scale = cosine_scheduler(epoch, total_epochs, start=start_value, end=self.final_skip)
        elif self.scheduler_type == 'none':
            self.skip_scale = 1.0
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")


    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply scaled skip connection
        out += self.skip_scale * identity
        out = self.bn4(out)
        out = F.relu(out)

        return out
     
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, pretrained=False, scheduler_type='none', total_epochs=100, final_skip_values=None, use_bn=False, update_per_batch=False):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.pretrained = pretrained
        self.scheduler_type = scheduler_type
        self.total_epochs = total_epochs
        self.final_skip_values = final_skip_values if final_skip_values else [1.0, 1.0, 1.0, 1.0]  # Default values
        self.use_bn = use_bn  # Determines which bottleneck to use
        self.update_per_batch = update_per_batch

        
        self.block = BottleneckBN if use_bn else block # Select the correct bottleneck type

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.block, 64, layers[0], stage=0, final_skip=self.final_skip_values[0])
        self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2, stage=1, final_skip=self.final_skip_values[1])
        self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2, stage=2, final_skip=self.final_skip_values[2])
        self.layer4 = self._make_layer(self.block, 512, layers[3], stride=2, stage=3, final_skip=self.final_skip_values[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1, stage=None, final_skip=0.0):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, scheduler_type=self.scheduler_type, total_epochs=self.total_epochs, final_skip=final_skip, update_per_batch=self.update_per_batch))

        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, scheduler_type=self.scheduler_type, total_epochs=self.total_epochs, final_skip=final_skip, update_per_batch=self.update_per_batch))

        return nn.Sequential(*layers)

    def update_skip_scale(self, step, total_steps):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                block.update_skip_scale(step, total_steps)


    def get_skip_scales(self):
        return {
            "stage1": self.layer1[0].skip_scale,
            "stage2": self.layer2[0].skip_scale,
            "stage3": self.layer3[0].skip_scale,
            "stage4": self.layer4[0].skip_scale,
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Factory function
def resnet50(num_classes=10, pretrained=False, scheduler_type='linear', total_epochs=100, final_skip_values=None, update_per_batch=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, pretrained=pretrained, scheduler_type=scheduler_type, total_epochs=total_epochs, final_skip_values=final_skip_values, update_per_batch=update_per_batch)

def resnet50bn(num_classes=10, pretrained=False, scheduler_type='linear', total_epochs=100, final_skip_values=None, update_per_batch=False):
    return ResNet(BottleneckBN, [3, 4, 6, 3], num_classes=num_classes, pretrained=pretrained, scheduler_type=scheduler_type, total_epochs=total_epochs, final_skip_values=final_skip_values, use_bn=True, update_per_batch=update_per_batch)