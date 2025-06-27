import cv2 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2
    return p

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

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None):  # ch_in, ch_out, kernel, stride, padding
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class C3k2(nn.Module):
    # YOLOv11의 C3 변형, 내부 구조에 따라 다름(이 예시는 일반적인 C3와 유사하게 작성)
    def __init__(self, c1, c2, shortcut=False, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *[Conv(c_, c_, 3) for _ in range(2)]  # repeats=2 기준
        )
        self.add = shortcut

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        if self.add:
            return self.cv3(torch.cat((y1 + y2, y2), dim=1))
        else:
            return self.cv3(torch.cat((y1, y2), dim=1))
    
class PSA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.fc(self.avg_pool(x))
        return x * attn
    
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))
    
class C2PSA(nn.Module):
    def __init__(self, c1, c2=None, e=0.5):
        super().__init__()
        c2 = c2 or c1
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0, bias=False)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, 0, bias=False)
        self.attn = PSA(c_)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.attn(self.cv1(x))
        x2 = self.cv2(x)
        return self.cv3(torch.cat((x1, x2), dim=1))
    
class YOLOv11MBackbone(nn.Module):
    def __init__(self, ch=3):
        super().__init__()
        self.layer0 = Conv(ch, 64, 3, 2)      # 0-P1/2
        self.layer1 = Conv(64, 128, 3, 2)     # 1-P2/4
        self.layer2 = C3k2(128, 256, shortcut=False, e=0.25)  # 2
        self.layer3 = Conv(256, 256, 3, 2)    # 3-P3/8
        self.layer4 = C3k2(256, 512, shortcut=False, e=0.25)  # 4
        self.layer5 = Conv(512, 512, 3, 2)    # 5-P4/16
        self.layer6 = C3k2(512, 512, shortcut=True)           # 6
        self.layer7 = Conv(512, 1024, 3, 2)   # 7-P5/32
        self.layer8 = C3k2(1024, 1024, shortcut=True)         # 8
        self.layer9 = SPPF(1024, 1024, 5)     # 9
        self.layer10 = C2PSA(1024)            # 10

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)        # ----> P3/8
        p3 = x
        x = self.layer4(x)
        x = self.layer5(x)        # ----> P4/16
        p4 = x
        x = self.layer6(x)
        x = self.layer7(x)        # ----> P5/32
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        p5 = x
        return [p3, p4, p5]   # 각 feature를 리스트로 반환

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, xs):
        return torch.cat(xs, self.dim)

class Pose(nn.Module):
    def __init__(self, nc, kpt_shape, ch_in=[256, 512, 1024]):  
        """
        nc: num_classes
        kpt_shape: (num_kpts, 3)  # (ex: 17, 3)
        ch_in: [P3, P4, P5] in-channel 수 (백본에 맞게 맞춰주세요)
        """
        super().__init__()
        self.nc = nc
        self.kpt_shape = kpt_shape
        self.num_kpts = kpt_shape[0]
        self.num_outputs = (nc + 1) + self.num_kpts * kpt_shape[1]  # +1=objectness

        # 각 feature별 output head
        self.detect_layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(c, self.num_outputs, 1)) for c in ch_in
        ])

    def forward(self, feats):
        # feats: [P3, P4, P5] feature maps
        # outputs shape: [(B, num_outputs, H, W), ...]
        out = []
        for i, f in enumerate(feats):
            y = self.detect_layers[i](f)
            # (B, num_outputs, H, W) 형태. 
            # 필요시 여기서 post-processing(e.g., y.sigmoid()) 가능
            out.append(y)
        return out  # [P3_out, P4_out, P5_out]


class YOLOv11MNeck(nn.Module):
    def __init__(self):
        super().__init__()
        # head에 쓸 블록
        self.active = nn.ReLU()
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.cat1 = Concat(1)
        self.c3k2_1 = C3k2(1024 + 512, 512, shortcut=False)  # cat backbone P4, 1024+512 -> 512
        
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.cat2 = Concat(1)
        self.c3k2_2 = C3k2(512 + 256, 256, shortcut=False)  # cat backbone P3, 512+256 -> 256

        self.down1 = Conv(256, 256, 3, 2)
        self.cat3 = Concat(1)
        self.c3k2_3 = C3k2(256 + 512, 512, shortcut=True)  # cat head P4, 256+512 -> 512

        self.down2 = Conv(512, 512, 3, 2)
        self.cat4 = Concat(1)
        self.c3k2_4 = C3k2(512 + 1024, 1024, shortcut=True)  # cat head P5, 512+1024 -> 1024

    def forward(self, feats):
        # feats: [P3, P4, P5] from backbone
        [P3, P4, P5] = feats  # ex: [256, 512, 1024 channel]
        # P5 (deepest) upsample & concat with P4
        x = self.up1(P5)
        x = self.cat1([x, P4])        # [P5_up, P4]
        x = self.active(x)
        x = self.c3k2_1(x)            # 512

        x2 = self.up2(x)
        x2 = self.cat2([x2, P3])      # [P4_up, P3]
        x2 = self.active(x2)
        P3_out = self.c3k2_2(x2)      # 256

        # Downsample, concat, C3k2
        x3 = self.down1(P3_out)
        x3 = self.cat3([x3, x])       # [P3_down, P4]
        x3 = self.active(x3)
        P4_out = self.c3k2_3(x3)      # 512

        x4 = self.down2(P4_out)
        x4 = self.cat4([x4, P5])      # [P4_down, P5]
        x4 = self.active(x4)
        P5_out = self.c3k2_4(x4)      # 1024
        # Pose (Detect) head
          # list of out heads
        return [P3_out, P4_out, P5_out]  # 각 output feature head (keypoint, cls, obj 등)



class PoseEstimator(nn.Module):
    def __init__(self, 
                 num_classes=5, 
                 kpt_shape=(4, 3), 
                 img_shape=(512, 512),
                 conf_thres=0.25, 
                 iou_thres=0.45):
        super().__init__()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_shape = img_shape
        self.num_cls = num_classes
        self.kpt_shape = kpt_shape
        self.backbone = YOLOv11MBackbone()
        self.neck     = YOLOv11MNeck()
        self.head     = Pose(self.num_cls, self.kpt_shape)  # P3/8, P4/16, P5/32
    
    def forward(self, x, train=False):
        feats = self.backbone(x)
        cv2.imshow('backbone', cv2.resize(feats[0][0].clone().detach().cpu().numpy().mean(0), (512, 512)))
        feats = self.neck(feats)
        outputs = self.head(feats)
        if not train:
            return self.poseprocess(outputs, 
                               strides=[8,16,32], 
                               img_shape=self.img_shape, 
                               conf_thres=self.conf_thres, 
                               iou_thres=self.iou_thres, 
                               nc=self.num_cls, 
                               kpt_shape=self.kpt_shape)
        return outputs 


    def poseprocess(self, outputs, strides, img_shape, conf_thres=0.25, iou_thres=0.45, nc=1, kpt_shape=(17,3)):
        """
        outputs: List of output tensors [P3_out, P4_out, P5_out], 각 shape (B, num_outputs, H, W)
        strides: ex) [8, 16, 32]
        img_shape: (height, width) - 원본 이미지 크기
        conf_thres: objectness threshold
        iou_thres: NMS threshold
        nc: num_classes
        kpt_shape: (num_kpts, 3)
        """
        device = outputs[0].device
        num_kpts = kpt_shape[0]
        B = outputs[0].size(0)
        all_poses = []

        for b in range(B):
            candidates = []
            for idx, (output, stride) in enumerate(zip(outputs, strides)):
                # output: (B, num_outputs, H, W)
                out = output[b]  # (num_outputs, H, W)
                num_outputs, H, W = out.shape
                # [cls..., obj, kpt1_x, kpt1_y, kpt1_score, ...]
                out = out.permute(1, 2, 0).reshape(-1, num_outputs)  # (H*W, num_outputs)

                # 활성화 함수
                cls_scores = torch.sigmoid(out[:, :nc])              # (N, nc)
                obj_scores = torch.sigmoid(out[:, nc:nc+1])          # (N, 1)
                kpt_pred   = out[:, nc+1:]                          # (N, num_kpts*3)
                kpt_pred   = kpt_pred.view(-1, num_kpts, 3)

                # 전체 confidence (obj * class)
                if nc == 1:
                    conf = obj_scores.squeeze(1) * cls_scores.squeeze(1)
                else:
                    conf, cls_idx = (obj_scores * cls_scores).max(1)

                # threshold
                mask = conf > conf_thres
                if mask.sum() == 0:
                    continue

                conf = conf[mask]
                if nc > 1:
                    cls_idx = cls_idx[mask]
                kpt_pred = kpt_pred[mask]

                # keypoint 좌표 복원: (sigmoid + grid) * stride
                kpts_xy = kpt_pred[..., :2]
                kpts_score = torch.sigmoid(kpt_pred[..., 2])
                # 이미지 범위로 clip
                kpts_xy[..., 0].clamp_(0, img_shape[1] - 1)
                kpts_xy[..., 1].clamp_(0, img_shape[0] - 1)
                # 결과 저장
                for i in range(conf.shape[0]):
                    res = {
                        "score": float(conf[i]),
                        "kpts": kpts_xy[i].detach().cpu().numpy(),      # (num_kpts, 2)
                        "kpt_score": kpts_score[i].detach().cpu().numpy(), # (num_kpts,)
                        "stride": stride,
                    }
                    if nc > 1:
                        res["class"] = int(cls_idx[i])
                    candidates.append(res)
            
            # NMS: 보통 bounding box 기반 NMS이지만, keypoints만 있을 때는 한정적으로 적용  
            # keypoint를 box로 감싸서 NMS 가능
            if len(candidates) == 0:
                all_poses.append([])
                continue

            # NMS를 위한 bbox 추출 (keypoint min/max box)
            boxes = []
            scores = []
            for pose in candidates:
                kpts = pose["kpts"]
                x1, y1 = kpts[:, 0].min(), kpts[:, 1].min()
                x2, y2 = kpts[:, 0].max(), kpts[:, 1].max()
                boxes.append([x1, y1, x2, y2])
                scores.append(pose["score"])
            boxes = torch.tensor(boxes, device=device, dtype=torch.float32)
            scores = torch.tensor(scores, device=device)
            keep = torchvision.ops.nms(boxes, scores, iou_thres)

            poses = [candidates[i] for i in keep.detach().cpu().numpy()]
            all_poses.append(poses)
        return all_poses
