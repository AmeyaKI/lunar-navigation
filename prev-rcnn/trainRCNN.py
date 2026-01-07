# Notes from Ameya:

# ORDER OF OPERATIONS:
# 1. Get RCNN to work for object detection.
# 2. Integrate MiDaS Depth Estimation Model.

# Current issue: RCNN is too computationally intensive to run on Macbook itself.
import torch, torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from torchmetrics.detection import MeanAveragePrecision
import torch.optim as optim
from torch.cuda.amp import GradScaler

import os
from tqdm import tqdm
from RCNNDataset import RCNNImageDataset

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")


# Data Processing
dataset_path = os.path.join(os.getcwd(), 'dataset') # make sure you are in Vision/new_model/ folder


transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = RCNNImageDataset(dataset_path, transform)

# Train: 70%, Val: 15%, Test: 15%
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

def collate(batch):
    return tuple(zip(*batch))

NUM_WORKERS = 2
BATCH_SIZE = 4
PREFETCH = 4
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=NUM_WORKERS,pin_memory=True, 
                          persistent_workers=False, collate_fn=collate, 
                          prefetch_factor=PREFETCH) 
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=NUM_WORKERS, pin_memory=True, 
                        persistent_workers=False, collate_fn=collate, 
                        prefetch_factor=PREFETCH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=NUM_WORKERS, pin_memory=True, 
                         persistent_workers=False, collate_fn=collate, 
                         prefetch_factor=PREFETCH)



# Model Construction

model_weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=model_weights)
num_classes = 4 # 3 rock sizes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features # type: ignore

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(device)

# print(model) # List out RCNN layers
# loss_function = built-in, call while training

parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(
    parameters,
    lr=2e-4,
    weight_decay=0.0005
)
# optimizer = optim.SGD(
#     parameters,
#     lr=0.005,
#     momentum=0.9,
#     weight_decay=0.0005
# )

scaler = torch.amp.GradScaler("cuda")

# Model Training

def train_epoch(model, train_loader, optimizer, device, epoch, print_freq=100, accum_steps=1):
    model.train() # set model to train mode
    running_loss = 0.0
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    optimizer.zero_grad()
    for batch_index, (images, targets) in progress_bar:
        # Move images/targets to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        with torch.amp.autocast('cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values()) / accum_steps
        # loss_dict = model(images, targets)
        # losses = sum((loss for loss in loss_dict.values() if torch.is_tensor(loss)), torch.tensor(0.0, device=device))
        
        # Backward pass
        scaler.scale(losses).backward()
        # losses.backward()
        # optimizer.step()
        
        # Update metrics
        if (batch_index + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        batch_loss = losses.item() * accum_steps
        running_loss += batch_loss
        progress_bar.set_postfix(loss=batch_loss)

        if (batch_index + 1) % print_freq == 0:
            avg_loss = running_loss / print_freq
            print(f"[Batch {batch_index + 1}/{len(train_loader)}] "
                  f"Loss: {avg_loss:.4f}")
            running_loss = 0.0 # reset running loss every {print_freq} batches
            
    return running_loss
        
def evaluate(model, test_loader, device):
    model.eval() # swithc to eval
    metric = MeanAveragePrecision() # mAP
    # total_loss = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # no val loss in model.eval() mode
            # with torch.amp.autocast('cuda'):
            #     loss_dict = model(images, targets)
            #     losses = sum(loss_dict.values())
            #     total_loss += losses.item()
            # loss_dict = model(images, targets)
            # total_loss += sum(loss.item() if torch.is_tensor(loss) else float(loss) for loss in loss_dict.values())
    
            # get mAP predictions
            preds = model(images)
            metric.update(preds, targets)
            
    # avg_loss = total_loss / len(test_loader)
    metrics = metric.compute()

    return metrics # avg_loss
    
def main():
    num_epochs = 10
    best_mAP = 0.0

    print("==========Training==========")
    for epoch in range(1, num_epochs+1):
        print(f"\n==========Epoch: {epoch}==========")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validating
        val_metrics = evaluate(model, val_loader, device)
        # print(f"Validation Loss: {val_loss:.4f}")
        print("Validation Metrics")
        print(f"mAP: {val_metrics['map']:.4f} | mAP_50: {val_metrics['map_50']:.4f} | mAP_75: {val_metrics['map_75']:.4f}")

        if val_metrics['map'] > best_mAP:
            best_mAP = val_metrics['map']
            torch.save(model.state_dict(), "best_model.pth")
            
            
    # Testing
    print("\n==========Testing==========")
    test_metrics = evaluate(model, test_loader, device)
    # print(f"Test Loss: {val_loss:.4f}")
    print(f"mAP: {test_metrics['map']:.4f} | mAP_50: {test_metrics['map_50']:.4f} | mAP_75: {test_metrics['map_75']:.4f}")
    
if __name__ == '__main__':
    main()