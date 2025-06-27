import torch 
import torch.nn as nn

# Conv+BN+SiLU 블록
class Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2 if p is None else p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()  # Swish

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# C2f (Cross Stage Partial Fusion)
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1)
        self.m = nn.ModuleList([Bottleneck(c_, c_) for _ in range(n)])
        
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# Bottleneck 블록 (C2f 내부)
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, 3, 1)
        self.add = shortcut

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return out + x if self.add else out

# SPPF
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2 * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


# Backbone: 여러 feature map 반환
class YOLOv8Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Conv(3, 32, 3, 2)              # 256x256 -> 128x128
        self.ds1 = Conv(32, 64, 3, 2)              # 128x128 -> 64x64
        self.stage1 = C2f(64, 64, n=1)
        self.ds2 = Conv(64, 128, 3, 2)             # 64x64 -> 32x32
        self.stage2 = C2f(128, 128, n=2)
        self.ds3 = Conv(128, 256, 3, 2)            # 32x32 -> 16x16
        self.stage3 = C2f(256, 256, n=3)
        self.sppf = SPPF(256, 256)                 # 16x16

    def forward(self, x):
        x = self.stem(x)
        p3 = self.ds1(x)
        p3 = self.stage1(p3)
        p4 = self.ds2(p3)
        p4 = self.stage2(p4)
        p5 = self.ds3(p4)
        p5 = self.stage3(p5)
        p5 = self.sppf(p5)
        return [p3, p4, p5]
    
# Neck: 채널수 일치
class Neck(nn.Module):
    def __init__(self, channels=[64, 128, 256]):
        super().__init__()
        self.reduce_conv1 = nn.Conv2d(channels[2], channels[1], 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f1 = C2f(channels[1]*2, channels[1], n=3)     # 128+128=256 in, 128 out
        
        self.reduce_conv2 = nn.Conv2d(channels[1], channels[0], 1)
        self.c2f2 = C2f(channels[0]+channels[0], channels[0], n=3)  # 64+64=128 in, 64 out
        
        self.downsample1 = nn.Conv2d(channels[0], channels[0], 3, stride=2, padding=1)
        self.c2f3 = C2f(channels[0]+channels[1], channels[1], n=3)  # 64+128=192 in, 128 out
        
        self.downsample2 = nn.Conv2d(channels[1], channels[1], 3, stride=2, padding=1)
        self.c2f4 = C2f(channels[1]+channels[2], channels[2], n=3)  # 128+256=384 in, 256 out

    def forward(self, features):
        P3, P4, P5 = features  # [64, 128, 256]
        
        # FPN top-down
        x = self.reduce_conv1(P5)  # 256->128
        x = self.upsample(x)
        x = torch.cat([x, P4], dim=1)  # 128+128=256
        x = self.c2f1(x)               # 128
        
        y = self.reduce_conv2(x)       # 128->64
        y = self.upsample(y)
        y = torch.cat([y, P3], dim=1)  # 64+64=128
        y = self.c2f2(y)               # 64
        
        z = self.downsample1(y)        # 64->64 (downsample)
        z = torch.cat([z, x], dim=1)   # 64+128=192
        z = self.c2f3(z)               # 128
        
        w = self.downsample2(z)        # 128->128 (downsample)
        w = torch.cat([w, P5], dim=1)  # 128+256=384
        w = self.c2f4(w)               # 256
        
        return w
    
# 전체 모델
class YOLOv8PoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = YOLOv8Backbone()
        self.neck = Neck([64, 128, 256])  # backbone의 output channel과 맞춤

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return feats