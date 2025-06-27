
import cv2
import torch 

from tqdm import tqdm 
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import LoadDataset
from models import PoseEstimator
from models.loss import YOLOPoseLoss

img_size = (512, 512)
model = PoseEstimator(num_classes=5, kpt_shape=(4, 3), img_shape=img_size).cuda()
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
data_loader = LoadDataset(task='pallet', 
                          data_path='/home/otter/dataset/pallet/dataset',
                          img_size=(512, 512),
                          batch_size=4,
                          shuffle=True,
                          num_worker=8,
                          pin_memory=False)

train_loader = data_loader.get_train_loader(True)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        torch.save(model.state_dict(), 'output/best2.pt')

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"  -> Learning Rate: {current_lr:.6f}\n")