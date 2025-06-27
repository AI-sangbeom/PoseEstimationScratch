import os 
import torch 
from .YOLO import YoloPoseDataset
from torchvision import transforms
from torch.utils.data import DataLoader

class LoadDataset:
    def __init__(self, 
                 task='pallet', 
                 data_path='/home/otter/dataset/pallet/dataset',
                 img_size=(512, 512),
                 batch_size=4,
                 shuffle=True,
                 num_worker=8,
                 pin_memory=False):
        
        self.train_path = os.path.join(data_path, 'train')
        self.test_path = os.path.join(data_path, 'test')
        self.task = task 
        self.batch_size = batch_size
        self.shuffle=shuffle
        self.num_worker=num_worker
        self.pin_memory=pin_memory
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def custom_collate_fn(self, batch):
        images, poses, orig_img = zip(*batch)  
        images = torch.stack(images, dim=0)
        orig_img = torch.stack(orig_img, dim=0)
        poses = [det for det in poses]

        return images, poses, orig_img
    
    def get_train_loader(self, only_loader=False):
        dataset = YoloPoseDataset(self.train_path, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                    num_workers=self.num_worker, pin_memory=self.pin_memory,
                    collate_fn=self.custom_collate_fn)
        if only_loader:
            return dataloader 
        else:
            return dataset, dataloader 

    def get_test_loader(self, only_loader=False):
        dataset = YoloPoseDataset(self.test_path, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                    num_workers=self.num_worker, pin_memory=self.pin_memory,
                    collate_fn=self.custom_collate_fn)
        if only_loader:
            return dataloader 
        else:
            return dataset, dataloader 