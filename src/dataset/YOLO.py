import os 
import cv2
import torch 
from torch.utils.data import Dataset

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