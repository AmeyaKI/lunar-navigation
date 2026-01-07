import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os

class RCNNImageDataset(Dataset):
    IMAGE_WIDTH = 720
    IMAGE_HEIGHT = 480
    SCALE = 600 / IMAGE_HEIGHT
    def __init__(self, dataset_path, transforms=None):
        self.dataset_path = dataset_path
        self.img_path = os.path.join(dataset_path, 'images/render') # directory to images folder
        self.boxes_path = os.path.join(dataset_path, 'rcnn_bounding_boxes_final.csv') # directory to csv
        
        self.df = pd.read_csv(self.boxes_path) # bounding boxes df
        self.images = sorted(self.df['image'].unique()) # unique image name values: render0001.jpg, render9771.jpg
        
        self.transforms = transforms # transforms (if applicable)
                
    def __len__(self):
        return len(self.images)        
                
    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.img_path, f"{img_name}")
        image = Image.open(img_path).convert('RGB')
        
        # Find bounding boxes and class_id for selected image
        img_df = self.df[self.df['image'] == img_name]
        boxes = img_df[['x_min', 'y_min', 'x_max', 'y_max']].values
        class_id = img_df['class_id'].values
        

        # transforms + Manual resizing

        # width, height = image.size # 720, 480
        # scaler = 7/15 # 336 x 224
        new_width = int(self.IMAGE_WIDTH * self.SCALE)
        new_height = int(self.IMAGE_HEIGHT * self.SCALE)
        
        image = image.resize((new_width, new_height))
        boxes = boxes * self.SCALE
        
        if self.transforms:
            image = self.transforms(image) # turns image into tensor
            
        # converts boxes and class_id to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(class_id, dtype=torch.int64)       
        
        # calculates area and iscrowd pytorch variables
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(img_df),), dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": iscrowd
        }
        
        # return image and target dict
        return image, target