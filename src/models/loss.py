
import cv2 
import torch 
import torch.nn as nn 

class YOLOPoseLoss(nn.Module):
    def __init__(self, nc=1, kpt_shape=(17,3), lambda_obj=1.0, lambda_cls=1.0, lambda_kpt=10.0):
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
        total_loss = self.lambda_obj * loss_obj + self.lambda_cls * loss_cls + self.lambda_kpt * loss_kpt
        loss_items = {
            "total": total_loss,
            "obj": loss_obj,
            "cls": loss_cls,
            "kpt": loss_kpt,
        }
        return total_loss, loss_items