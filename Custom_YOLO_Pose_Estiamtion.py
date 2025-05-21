import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.ops as ops

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Agg')
import cv2
import torch
import numpy as np


def visualize_yolov8_pose(
    image: torch.Tensor,        # [3, H, W]
    preds: torch.Tensor,        # [N_pred, 4+1+num_classes+3*num_keypoints]
    targets: torch.Tensor,      # [1, 5+2*num_keypoints]
    num_classes: int = 5,
    num_keypoints: int = 4,
    obj_conf_thres: float = 0.5,
    kpt_conf_thres: float = 0.3,
    iou_thres: float = 0.5,
    box_color: tuple = (0,255,0),
    kpt_color: tuple = (0,0,255)
):
    device = preds.device
    N, D = preds.shape
    _, H0, W0 = image.shape

    # 1) stride & feature sizes
    strides = [4, 8, 16]
    feature_sizes = [(H0//s, W0//s) for s in strides]

    # 2) decode
    decoded_boxes = []
    start = 0
    for (Hf,Wf), stride in zip(feature_sizes, strides):
        n = Hf*Wf
        p = preds[start:start+n]; start+=n

        gy, gx = torch.meshgrid(
            torch.arange(Hf, device=device, dtype=p.dtype),
            torch.arange(Wf, device=device, dtype=p.dtype),
            indexing='ij'
        )
        grid = torch.stack((gx, gy), dim=2).view(-1,2)*stride

        box_p = p[:,:4]

        xy = (torch.sigmoid(box_p[:,:2]) + grid)
        wh = (torch.sigmoid(box_p[:,2:4]))
        decoded_boxes.append(torch.cat([xy,wh],1))

    boxes = torch.cat(decoded_boxes,0)   # [N_pred,4]


    # 4) collect valid dets
    raw_kpt = preds[:, 5+num_classes:].view(-1, num_keypoints, 3)
    kpt_conf = torch.sigmoid(raw_kpt[..., 2])             # [N_pred, K]
    kpt_xy_raw = raw_kpt[..., :2]                         # [N_pred, K, 2]

    # 3) keypoint normalize → offset ([-1, +1] 범위)
    kpt_offset = torch.sigmoid(kpt_xy_raw)     # [N_pred, K, 2]

    # 4) assemble detections + NMS
    all_boxes, all_scores, all_kpts = [], [], []
    obj_conf  = torch.sigmoid(preds[:,4])
    cls_conf  = torch.sigmoid(preds[:,5:5+num_classes]).max(1)[0]
    combined  = obj_conf * cls_conf

    for i in range(N):
        if combined[i] < obj_conf_thres:
            continue
        cx, cy, w, h = boxes[i] 
        w, h = w*W0, h*H0
        score = float(combined[i])

        # box 좌표 int 형 변환
        x1 = int((cx - w/2).clamp(0, W0-1))
        y1 = int((cy - h/2).clamp(0, H0-1))
        x2 = int((cx + w/2).clamp(0, W0-1))
        y2 = int((cy + h/2).clamp(0, H0-1))

        all_boxes.append(torch.tensor([x1,y1,x2,y2], device=device))
        all_scores.append(score)

        # keypoints: 박스 중심+ offset*w/2,h/2
        pts = []
        for k in range(num_keypoints):
            off_x, off_y = kpt_offset[i,k]
            off_x, off_y = off_x*W0, off_y*H0
            px = off_x
            py = off_y
            pts.append((
                int(px.clamp(0, W0-1)),
                int(py.clamp(0, H0-1)),
                float(kpt_conf[i,k])
            ))
        all_kpts.append(pts)

    if not all_boxes:
        return image.cpu().permute(1,2,0).numpy()

    boxes_t  = torch.stack(all_boxes).float()
    scores_t = torch.tensor(all_scores, device=device)
    keep     = ops.nms(boxes_t, scores_t, iou_thres)[:1]  # top-1만 그릴 때

    # 5) draw
    out = image.permute(1,2,0).cpu().numpy().copy()
    for idx in keep:
        x1,y1,x2,y2 = boxes_t[idx].cpu().tolist()
        cv2.rectangle(out, (int(x1),int(y1)), (int(x2),int(y2)), box_color, 2)
        for px,py,pc in all_kpts[idx]:
            if pc < kpt_conf_thres: continue
            cv2.circle(out, (px,py), 5, kpt_color, -1)

    # (선택) GT
    t = targets.cpu().numpy()
    xy = t[6:].reshape(-1, 2)
    for x, y in xy:
        gx = int(x*W0)
        gy = int(y*H0)
        cv2.circle(out, (gx,gy), 2, (255, 0,0), -1)

    cv2.imshow('yolov8_pose', out)
    cv2.waitKey(1)
    return out



def plot_loss_curve(loss_history):
    fig = plt.figure(figsize=(6,4))
    plt.plot(loss_history, label='Avg Loss')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    fig.canvas.draw()
    saveimg = np.asarray(fig.canvas.buffer_rgba())
    saveimg = saveimg[..., :3]  # 알파 채널 제거
    saveimg = cv2.cvtColor(saveimg, cv2.COLOR_RGB2BGR)    
    cv2.imshow('loss_curve', saveimg)
    cv2.waitKey(1)


import math
import torch

def build_targets(preds, target_tensor, num_keypoints, img_size=(256,256)):
    """
    Multi-scale target assignment for YOLOv8-style pose head.

    Args:
        preds:        Tensor of shape [B, N_pred, 4+1+C+3K]
        target_tensor: either
           - Tensor [B, 1, 5+2K]  (one GT per batch), or
           - Tensor [B, N_gt, 5+2K] (multiple GTs)
        num_keypoints: int K
        img_size:     (H, W) of the model input

    Returns:
        obj_mask:  Tensor [B, N_pred] of 0/1 objectness targets
        box_list:  list of length B, each Tensor [N_i, 4] (cx,cy,w,h)
        cls_list:  list of length B, each Tensor [N_i] (class indices)
        kpt_list:  list of length B, each Tensor [N_i, K, 2] (normalized kpt XY)
        vis_list:  list of length B, each Tensor [N_i, K] (all ones)
        bvis_list: list of length B, each Tensor [N_i] (all ones)
        indices:   list of length B, each Tensor [N_i] (flattened pred indices)
    """
    B, N_pred, _ = preds.shape
    device = preds.device
    H, W = img_size

    # 1) unpack targets into a per-batch list
    targets_list = []
    # if shape [B, 1, M], squeeze that middle dim
    if target_tensor.dim() == 3 and target_tensor.size(1) == 1:
        target_tensor = target_tensor.squeeze(1)  # [B, M]
    for b in range(B):
        t = target_tensor[b]
        if t.dim() == 1:
            t = t.unsqueeze(0)  # [1, 5+2K]
        targets_list.append(t)

    # 2) define strides & scale ranges (by pixel diag)
    strides      = [4, 8, 16]
    scale_ranges = [(0, 64), (64, 128), (128, 1e8)]
    # compute number of preds per level & offsets for flattening
    level_sizes   = [(H//s)*(W//s) for s in strides]
    level_offsets = [0] + list(torch.cumsum(torch.tensor(level_sizes[:-1]), dim=0).tolist())
    assert sum(level_sizes) == N_pred, "preds.shape[1] must == sum of level grid sizes"

    # 3) prepare outputs
    obj_mask   = torch.zeros(B, N_pred, device=device)
    box_list   = [[] for _ in range(B)]
    cls_list   = [[] for _ in range(B)]
    kpt_list   = [[] for _ in range(B)]
    vis_list   = [[] for _ in range(B)]
    bvis_list  = [[] for _ in range(B)]
    indices    = [[] for _ in range(B)]

    # 4) assign each GT to exactly one scale
    for lvl, (stride, (mn, mx), offset) in enumerate(zip(strides, scale_ranges, level_offsets)):
        Hf, Wf = H // stride, W // stride
        for b, targets in enumerate(targets_list):
            for t in targets:
                cls_i = int(t[0].item())
                cx, cy, bw, bh = t[1].item(), t[2].item(), t[3].item(), t[4].item()
                # pixel-space diagonal of the box
                diag = math.hypot(bw*W, bh*H)
                if not (mn <= diag < mx):
                    continue

                # map normalized center to this level's grid
                gx = int((cx * Wf))
                gy = int((cy * Hf))
                idx = gy * Wf + gx + offset

                # set objectness target
                obj_mask[b, idx] = 1

                # record this pred index
                indices[b].append(idx)

                # append GT values
                box_list[b].append(torch.tensor([cx, cy, bw, bh], device=device))
                cls_list[b].append(torch.tensor(cls_i, device=device, dtype=torch.long))
                kpt_list[b].append(t[6:].view(num_keypoints, 2).to(device))
                vis_list[b].append(torch.ones(num_keypoints, device=device))
                bvis_list[b].append(torch.tensor(1.0, device=device))

    # 5) convert lists → tensors, handling empty cases
    for b in range(B):
        if len(indices[b]) > 0:
            box_list[b]  = torch.stack(box_list[b], dim=0)
            cls_list[b]  = torch.stack(cls_list[b], dim=0)
            kpt_list[b]  = torch.stack(kpt_list[b], dim=0)
            vis_list[b]  = torch.stack(vis_list[b], dim=0)
            bvis_list[b] = torch.stack(bvis_list[b], dim=0).view(-1)
            indices[b]   = torch.tensor(indices[b], dtype=torch.long, device=device)
        else:
            # no GT assigned → create zero-size tensors
            box_list[b]  = torch.zeros((0,4), device=device)
            cls_list[b]  = torch.zeros((0,),     dtype=torch.long, device=device)
            kpt_list[b]  = torch.zeros((0,num_keypoints,2), device=device)
            vis_list[b]  = torch.zeros((0,num_keypoints),      device=device)
            bvis_list[b] = torch.zeros((0,), device=device)
            indices[b]   = torch.zeros((0,), dtype=torch.long, device=device)

    return obj_mask, box_list, cls_list, kpt_list, vis_list, bvis_list, indices


def yolo_pose_loss(preds, targets, num_classes, num_keypoints,
                   box_weight=7.5, cls_weight=0.5,
                   obj_weight=1.0, kpt_weight=1.5, img_size=(512, 512)):
    """
    preds:   [B, N_pred, 4 + 1 + C + 3K]
    targets: list of length B, each [N_i, 5 + 2K] with format
             [ dummy?, cls, cx, cy, w, h, kpt1_x, kpt1_y, … ]
    """
    B, N_pred, _ = preds.shape
    device = preds.device

    # 1) multi-scale assignment → masks + GT lists
    #    build_targets는 cls=t[1], box=t[2:6], kpts=t[6:] 을 기준으로 할 것
    obj_mask, box_gt, cls_gt, kpt_gt, vis_gt, bvis_gt, indices = \
        build_targets(preds, targets, num_keypoints, img_size=img_size)

    # 2) obj loss (BCE)
    obj_logits = preds[..., 4]                           # [B, N_pred]
    obj_loss   = F.binary_cross_entropy_with_logits(
                    obj_logits, obj_mask, reduction='mean')

    # 3) prepare accumulators
    total_box      = 0.0
    total_box_conf = 0.0
    total_cls      = 0.0
    total_kpt_xy   = 0.0
    total_kpt_conf = 0.0

    # avoid zero‐division
    total_pos = obj_mask.sum().clamp_min(1.0)

    # 4) loop over batch
    for b in range(B):
        pos_idx = indices[b]       # Tensor of positive pred indices for image b
        if pos_idx.numel() == 0:
            continue

        p = preds[b, pos_idx]      # [N_i, 4+1+C+3K]

        # -- split predictions --
        pred_box       = torch.sigmoid(p[:, :4])               # center
        pred_box_conf = p[:, 4]                               # objectness logit

        cls_logits    = p[:, 5:5+num_classes]                 # [N_i, C]

        kpt_logits    = p[:, 5+num_classes:].view(-1, num_keypoints, 3)
        pred_kpt_xy   = torch.sigmoid(kpt_logits[..., :2])
        pred_kpt_conf = kpt_logits[..., 2]

        # -- ground‐truth from build_targets --
        gt_box        = box_gt[b][:, :4]      # [N_i, 2]
        gt_bvis       = bvis_gt[b]            # [N_i]  all ones
        gt_cls        = cls_gt[b]             # [N_i]  long
        gt_kpt_xy     = kpt_gt[b]             # [N_i, K, 2]
        gt_vis        = vis_gt[b]             # [N_i, K]

        # --- box regression ---
        total_box += F.l1_loss(pred_box, gt_box, reduction='mean')

        # box‐confidence vs GT objectness
        total_box_conf += (
            F.binary_cross_entropy_with_logits(
                pred_box_conf, gt_bvis, reduction='none'
            ).sum()
            / gt_bvis.sum()#.clamp_min(1.0)
        )

        # --- classification (CrossEntropy) ---
        total_cls += F.cross_entropy(cls_logits, gt_cls, reduction='mean')

        # --- keypoint xy regression ---
        # weight by visibility mask
        kpt_xy_loss = (
            F.l1_loss(pred_kpt_xy, gt_kpt_xy, reduction='none')
            .sum(-1)           # sum over K and coords
            * gt_vis           # mask invisible
        ).sum() / gt_vis.sum()#.clamp_min(1.0)
        total_kpt_xy += kpt_xy_loss

        # --- keypoint confidence ---
        total_kpt_conf += (
            F.binary_cross_entropy_with_logits(
                pred_kpt_conf, gt_vis, reduction='none'
            ).sum()
            / gt_vis.sum().clamp_min(1.0)
        )

    # 5) combine & weight
    box_loss = (total_box + 0.5 * total_box_conf) / total_pos
    cls_loss = total_cls / total_pos
    kpt_loss = (total_kpt_xy + 0.5 * total_kpt_conf) / total_pos

    total_loss = (
        box_weight * box_loss +
        cls_weight * cls_loss +
        obj_weight * obj_loss +
        kpt_weight * kpt_loss
    )

    return total_loss, {
        "obj_loss": obj_loss.item(),
        "box_loss": box_loss.item(),
        "cls_loss": cls_loss.item(),
        "kpt_loss": kpt_loss.item(),
    }

# (Conv, Bottleneck, C2f, SPPF 클래스 생략: 위 코드와 동일하게 사용)
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
        return [p3, p4, p5]  # 각각 64x64, 32x32, 16x16이 됨

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
        
        return [y, z, w]


# Head (detect-pose)
class DetectPoseHead(nn.Module):
    def __init__(self, num_classes, num_keypoints, ch):
        super().__init__()
        self.detect_layers = nn.ModuleList([
            nn.Conv2d(c, (num_classes + 5 + num_keypoints*3), 1) for c in ch
        ])
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints

    def forward(self, features):
        outputs = []
        for x, head in zip(features, self.detect_layers):
            y = head(x)  # [B, out_dim, H, W]
            cls_heatmap = y[0][5:10].mean(0).detach().cpu().numpy()
            kpt_heatmap = y[0][10:].mean(0).detach().cpu().numpy()
            box_heatmap = y[0][:4].mean(0).detach().cpu().numpy()
            t_heatmap = y[0].mean(0).detach().cpu().numpy()
            heatmap = [cls_heatmap, box_heatmap, kpt_heatmap, t_heatmap]
            B, out_dim, H, W = y.shape
            y = y.permute(0, 2, 3, 1).reshape(B, -1, out_dim)  # [B, N, out_dim]
            outputs.append(y)
        return torch.cat(outputs, dim=1), heatmap  # [B, N, out_dim]


# 전체 모델
class YOLOv8PoseModel(nn.Module):
    def __init__(self, num_classes=1, num_keypoints=4):
        super().__init__()
        self.backbone = YOLOv8Backbone()
        self.neck = Neck([64, 128, 256])  # backbone의 output channel과 맞춤
        self.head = DetectPoseHead(num_classes, num_keypoints, ch=[64, 128, 256])
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        preds, heatmap = self.head(feats)  # [B, N, 4+C+K*3]
        return preds, heatmap

# Loss function은 기존 yolo_pose_loss 함수 그대로 사용 가능!

# ========== 예시 학습 loop ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1. 모델 생성
model = YOLOv8PoseModel(num_classes=5, num_keypoints=4).to(device)

# 2. (가상) 입력/정답 데이터
class YoloPoseDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.img_dir = os.path.join(data_path, 'images')
        self.label_dir = os.path.join(data_path, 'labels')
        self.img_files = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_files)


    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        label_file = img_file.replace('.jpg', '.txt')

        # 이미지 로드
        img_path = os.path.join(self.img_dir, img_file)
        img = cv2.imread(img_path)
        if self.transform is not None:
            img = self.transform(img)

        # 레이블 로드 (모든 객체)
        label_path = os.path.join(self.label_dir, label_file)
        labels = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    pose = [float(x) for x in parts[:5]] + [1.0] + [float(x) for x in parts[5:]]# objectness = 1.0
                    labels.append(pose)

        labels = torch.tensor(labels, dtype=torch.float32)  # (num_objs, num_kps*2)

        return img, labels

def custom_collate_fn(batch):
    images, poses = zip(*batch)  # 튜플을 분리
    images = torch.stack(images, dim=0)  # 이미지만은 고정 크기이므로 stack 가능
    poses = torch.stack([det[0] if len(det) > 0 else torch.zeros(5) for det in poses], dim=0)

    return images, poses


from tqdm import tqdm 
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
img_size = (640, 640)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(img_size),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DataLoader
base = '/home/otter/dataset/pallet/dataset/train'
train_ds = YoloPoseDataset(base, transform=transform)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                          num_workers=2, pin_memory=True,
                          collate_fn=custom_collate_fn)

opt = AdamW(
    model.parameters(),
    lr=1e-3,           # 초기 학습률
    weight_decay=1e-2 # L2 정규화 강도
)

# 2) Scheduler 설정
# CosineAnnealingLR: T_max = 한 사이클(epoch) 길이, eta_min = 최소 학습률
num_epochs = 10000
scheduler = CosineAnnealingLR(
    opt,
    T_max=num_epochs,
    eta_min=1e-6
)

loss_history = []

for epoch in range(num_epochs):
    total_loss = 0.0
    total_samples = 0   
    pbar = tqdm(train_loader)
    for i, (imgs, targets) in enumerate(pbar):
        imgs = imgs.to(device)
        targets = targets.to(device)
        preds, heatmaps = model(imgs)  # [B, N_pred, 4+C+K*3]
        loss, _ = yolo_pose_loss(preds, targets, 5, 4, img_size=img_size)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        total_samples += imgs.size(0)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        pbar.desc = f'Epoch {epoch}, Avg Loss: {avg_loss:.4f}'
        if i%10 == 0:
            new_heatmaps = []
            for i in range(4):
                heatmap = heatmaps[i]
                heatmap = cv2.resize(heatmap, (img_size[0]//2, img_size[0]//2))
                new_heatmaps.append(heatmap)
                
            # vstack
            vstacked1 = np.vstack([new_heatmaps[0], new_heatmaps[1]])
            vstacked2 = np.vstack([new_heatmaps[2], new_heatmaps[3]])
            stacked_heatmap = np.hstack([vstacked1, vstacked2])
            # heatmap = cv2.resize(heatmap, img_size)
            cv2.imshow('heatmap', stacked_heatmap)
            visualize_yolov8_pose(imgs[0], preds[0], targets[0])
    loss_history.append(avg_loss)
    plot_loss_curve(loss_history)
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"  -> Learning Rate: {current_lr:.6f}\n")
    torch.save(model.state_dict(), 'yolov8_pose_last.pth')