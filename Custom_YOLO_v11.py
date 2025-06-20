import timm
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


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
    def __init__(self, nc, kpt_shape, ch_in=[256, 512, 768]):  
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
        self.detect_layers = nn.ModuleList([
            nn.Conv2d(c, self.num_outputs, 1) for c in ch_in
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


class YOLOv11MHead(nn.Module):
    def __init__(self, nc, kpt_shape):
        super().__init__()
        # head에 쓸 블록
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.cat1 = Concat(1)
        self.c3k2_1 = C3k2(256 + 256, 256, shortcut=False)  # cat backbone P4, 1024+512 -> 512

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.cat2 = Concat(1)
        self.c3k2_2 = C3k2(256 + 256, 256, shortcut=False)  # cat backbone P3, 512+256 -> 256

        self.down1 = Conv(256, 256, 3, 2)
        self.cat3 = Concat(1)
        self.c3k2_3 = C3k2(256 + 256, 512, shortcut=True)  # cat head P4, 256+512 -> 512

        self.down2 = Conv(512, 512, 3, 2)
        self.cat4 = Concat(1)
        self.c3k2_4 = C3k2(512 + 256, 768, shortcut=True)  # cat head P5, 512+1024 -> 1024

        self.pose_head = Pose(nc, kpt_shape)  # P3/8, P4/16, P5/32

    def forward(self, feats):
        # feats: [P3, P4, P5] from backbone
        [P3, P4, P5] = feats  # ex: [256, 512, 1024 channel]
        # P5 (deepest) upsample & concat with P4
        x = self.up1(P5)
        x = self.cat1([x, P4])        # [P5_up, P4]
        x = self.c3k2_1(x)            # 512

        x2 = self.up2(x)
        x2 = self.cat2([x2, P3])      # [P4_up, P3]
        P3_out = self.c3k2_2(x2)      # 256

        # Downsample, concat, C3k2
        x3 = self.down1(P3_out)
        x3 = self.cat3([x3, x])       # [P3_down, P4]
        P4_out = self.c3k2_3(x3)      # 512

        x4 = self.down2(P4_out)
        x4 = self.cat4([x4, P5])      # [P4_down, P5]
        P5_out = self.c3k2_4(x4)      # 1024

        # Pose (Detect) head
        outs = self.pose_head([P3_out, P4_out, P5_out])  # list of out heads

        return outs  # 각 output feature head (keypoint, cls, obj 등)


import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOPoseLoss(nn.Module):
    def __init__(self, nc=1, kpt_shape=(17,3), lambda_obj=3.0, lambda_cls=3.0, lambda_kpt=3.0):
        super().__init__()
        self.nc = nc
        self.num_kpts = kpt_shape[0]
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.lambda_kpt = lambda_kpt

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.smoothl1 = nn.SmoothL1Loss(reduction="none")
    
    def forward(self, preds, targets):
        """
        preds: [P3, P4, P5], 각 shape (B, num_outputs, H, W)
        targets: list of dicts [{boxes, labels, kpts}], 한 이미지(배치) 당 하나
        targets의 kpts: (N_obj, num_kpts, 3)
        """

        # 여러 FPN 스케일에서 concat (ex: YOLOv8 방식)
        loss_obj = 0.0
        loss_cls = 0.0
        loss_kpt = 0.0

        for pred, stride in zip(preds, [8,16,32]):  # ex: 각 output feature map
            B, num_outputs, H, W = pred.shape

            # For visualization
            ############################################################################################
            show_pred = pred[0].clone()
            show_cls_pred = show_pred[:self.nc].mean(0, keepdim=False)
            show_obj_pred = show_pred[self.nc:self.nc+1].mean(0, keepdim=False)
            show_kpt_pred = show_pred[self.nc+1:].mean(0, keepdim=False)
            feat_size = 244
            resized_cls_pred = cv2.resize(show_cls_pred.detach().cpu().numpy(), (feat_size, feat_size))
            resized_obj_pred = cv2.resize(show_obj_pred.detach().cpu().numpy(), (feat_size, feat_size))
            resized_kpt_pred = cv2.resize(show_kpt_pred.detach().cpu().numpy(), (feat_size, feat_size))
            concated_pred = cv2.hconcat([resized_cls_pred, resized_obj_pred, resized_kpt_pred])
            cv2.imshow(f'{stride} feature map', concated_pred)
            ############################################################################################

            pred = pred.permute(0,2,3,1).reshape(B, H*W, num_outputs)  # (B, HW, C)
            cls_pred = pred[..., :self.nc]            # (B, HW, nc)
            obj_pred = pred[..., self.nc:self.nc+1]   # (B, HW, 1)
            kpt_pred = pred[..., self.nc+1:]          # (B, HW, num_kpts*3)

            # 타겟에서 각 그리드/스케일에 해당하는 GT를 선택해 매칭해야 함
            # (아래는 매우 간단한 예시, 실제로는 assigner 사용 필요)
            for b in range(B):
                t = targets[b]
                # 여기서 t['kpts']는 (N_obj, num_kpts, 3)
                # t['boxes']: (N_obj, 4)  (cx, cy, w, h)
                # t['labels']: (N_obj,)
                N_obj = t['boxes'].shape[0]
                if N_obj == 0:
                    # GT 없는 배치 - 모든 그리드에 obj=0, loss만 계산
                    obj_tgt = torch.zeros((H*W, 1), device=pred.device)
                    loss_obj += self.bce(obj_pred[b], obj_tgt).mean()
                    continue

                # (실제로는 anchor assign 필요)
                # 여기선 GT 중심이 속한 grid cell에 positive sample로 할당하는 식의 예시
                gt_xy = t['boxes'][:, :2] / stride   # (N_obj, 2)
                gt_grid = (gt_xy*512).long()
                grid_idx = gt_grid[:,1]*W + gt_grid[:,0]    # (N_obj, )

                obj_tgt = torch.zeros((H*W, 1), device=pred.device)
                cls_tgt = torch.zeros((H*W, self.nc), device=pred.device)
                kpt_tgt = torch.zeros((H*W, self.num_kpts, 3), device=pred.device)
                kpt_mask = torch.zeros((H*W, self.num_kpts), device=pred.device) # visibility mask

                # 각 GT별로 해당 cell에 positive 할당
                for i, idx in enumerate(grid_idx):
                    if idx < 0 or idx >= H*W: continue
                    obj_tgt[idx] = 1.0
                    cls_tgt[idx, t['labels'][i]] = 1.0
                    kpt_tgt[idx] = t['kpts'][i]
                    kpt_mask[idx] = (t['kpts'][i,:,2]>0).float()  # visible만 loss

                # Loss 계산
                loss_obj += self.bce(obj_pred[b], obj_tgt).mean()
                if self.nc > 1:
                    loss_cls += self.bce(cls_pred[b], cls_tgt).mean()

                # keypoint: x, y는 SmoothL1 / BCE, mask로 visible한 점만 loss
                pred_kpt = kpt_pred[b].reshape(H*W, self.num_kpts, 3)
                tgt_kpt_xy = kpt_tgt[...,:2]   # (H*W, num_kpts, 2)
                pred_kpt_xy = pred_kpt[...,:2]
                loss_kpt_xy = self.smoothl1(pred_kpt_xy, tgt_kpt_xy) * kpt_mask.unsqueeze(-1)
                loss_kpt += loss_kpt_xy.sum() / (kpt_mask.sum()+1e-6)

                # score/visibility는 BCE loss 사용
                pred_kpt_score = pred_kpt[...,2]
                tgt_kpt_score = kpt_tgt[...,2]
                loss_kpt_score = self.bce(pred_kpt_score, tgt_kpt_score) * kpt_mask
                loss_kpt += loss_kpt_score.sum() / (kpt_mask.sum()+1e-6)

        # 최종 loss weighted sum
        total_loss = (self.lambda_obj * loss_obj + self.lambda_cls * loss_cls + self.lambda_kpt * loss_kpt)*30
        loss_items = {
            "total": total_loss,
            "obj": loss_obj,
            "cls": loss_cls,
            "kpt": loss_kpt,
        }
        return total_loss, loss_items

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
        self.head     = YOLOv11MHead(self.num_cls, self.kpt_shape)
    
    def forward(self, x, train=False):
        feats = self.backbone(x)
        cv2.imshow('backbone', cv2.resize(feats[0][0].clone().detach().cpu().numpy().mean(0), (512, 512)))
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

import os 
import cv2
import numpy as np 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 2. (가상) 입력/정답 데이터
class YoloPoseDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.img_dir = os.path.join(data_path, 'images')
        self.label_dir = os.path.join(data_path, 'labels')
        self.img_files = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def str2float(self, parts):
        return [float(x) for x in parts]

    def str2int(self, parts):
        return [int(x) for x in parts]

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        label_file = img_file.replace('.jpg', '.txt')

        # 이미지 로드
        img_path = os.path.join(self.img_dir, img_file)
        img = cv2.imread(img_path)
        orig_img = torch.from_numpy(img.copy())
        if self.transform is not None:
            img = self.transform(img)

        # 레이블 로드 (모든 객체)
        label_path = os.path.join(self.label_dir, label_file)
        labels = [] 
        with open(label_path, 'r') as f:
            classes   = []
            boxes     = []
            keypoints = []
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = self.str2int(parts[0])
                    box = self.str2float(parts[1:5])
                    keypoint = self.str2float(parts[5:])
                    keypoint.insert(2, 1.)
                    keypoint.insert(5, 1.)
                    keypoint.insert(8, 1.)
                    keypoint.append(1.)
        #             labels.append(cls + box + keypoint)
        # labels = torch.FloatTensor(labels)
                    boxes.append(box)
                    classes.append(cls)
                    keypoints.append(keypoint)
        classes   = torch.LongTensor(classes)
        boxes     = torch.FloatTensor(boxes)
        keypoints = torch.FloatTensor(keypoints).reshape(-1, 4, 3)
        labels = {
            "boxes": boxes,         # (N_obj, 4)
            "labels": classes,       # (N_obj,)
            "kpts": keypoints,           # (N_obj, num_kpts, 3)
        }


        return img, labels, orig_img

def custom_collate_fn(batch):
    images, poses, orig_img = zip(*batch)  
    images = torch.stack(images, dim=0)
    orig_img = torch.stack(orig_img, dim=0)
    poses = [det for det in poses]

    return images, poses, orig_img

img_size = (512, 512)
model = PoseEstimator(num_classes=5, kpt_shape=(4, 3), img_shape=img_size).cuda()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(img_size),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DataLoader
base = '/home/otter/dataset/pallet/dataset/train'
train_ds = YoloPoseDataset(base, transform=transform)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                          num_workers=8, pin_memory=False,
                          collate_fn=custom_collate_fn)

import numpy as np
from tqdm import tqdm 
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = AdamW(
    model.parameters(),
    lr=1e-3,           # 초기 학습률
    weight_decay=1e-2 # L2 정규화 강도
)

# 2) Scheduler 설정
# CosineAnnealingLR: T_max = 한 사이클(epoch) 길이, eta_min = 최소 학습률
num_epochs = 100
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6
)

loss_history = []
loss_fn = YOLOPoseLoss(nc=5, kpt_shape=(4, 3))
best = 1e+9

for epoch in range(num_epochs):
    total_loss = 0.0
    total_samples = 0   
    pbar = tqdm(train_loader)
    for i, (img, target, orig_img) in enumerate(pbar):
        img = img.cuda()
        result = model(img, True)
        loss, _ = loss_fn(result, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_samples += img.size(0)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        pbar.desc = f'Epoch {epoch}, Avg Loss: {avg_loss:.4f}'        
        if i%10 == 0:
            inference = model(img[0].unsqueeze(0))[0]
            frame = cv2.resize(orig_img[0].numpy(), (1080, 720)) # (C, H, W)

            for output in inference:
                kpts = output['kpts']
                for x, y in kpts:
                    x, y = int(x*1080), int(y*720)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            for tkpts in target[0]['kpts']:
                xy = tkpts[:, :2]
                for x, y in xy:
                    x, y = int(x*1080), int(y*720)
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('p'): break
    if best > avg_loss:
        best = avg_loss
        torch.save(model.state_dict(), 'best.pt')

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"  -> Learning Rate: {current_lr:.6f}\n")