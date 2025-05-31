import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        '''Conv8->Conv32->Conv64->Conv64->Linear(outputs), BatchNorm, ReLU

        Args:
            h: 输入图像的高度
            w: 输入图像的宽度
            outputs: 动作空间的大小
        '''
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 卷积后的尺寸
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 64
        
        self.head = nn.Linear(linear_input_size, outputs)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (batch, channels, h, w)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.contiguous().view(x.size(0), -1)) # 展平为 (batch_size, features)