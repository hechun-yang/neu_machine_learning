import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

CONFIG = {
    'data_dir': './detection',
    'train_csv': 'fovea_localization_train_GT.csv',
    'yolo_path': 'runs/detect/train/weights/best.pt',
    'crop_size': 512,       
    'roi_size': 400,       
    'batch_size': 8,
    'lr': 1e-4,
    'epochs': 15,          
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Using device: {CONFIG['device']}")

class CascadeDataset(Dataset):
    def __init__(self, img_dir, csv_file=None, mode='train', transform=None, yolo_model=None):
        self.img_dir = img_dir
        self.mode = mode
        self.transform = transform
        self.roi_half = CONFIG['roi_size'] // 2
        
        if mode == 'train':
            self.data = pd.read_csv(csv_file)
        else:
            self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
            self.yolo = yolo_model

    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        else:
            return len(self.img_files)

    def __getitem__(self, idx):
        
        if self.mode == 'train':
            row = self.data.iloc[idx]
            
            try:
                raw_id = row['data']
                img_id_str = str(int(float(raw_id))) # 33.0 -> 33 -> "33"
                
                possible_names = [
                    img_id_str.zfill(4) + '.jpg', 
                    img_id_str + '.jpg'          
                ]
                
                img_name = None
                for name in possible_names:
                    if os.path.exists(os.path.join(self.img_dir, name)):
                        img_name = name
                        break
                
                if img_name is None:
                   
                    img_name = possible_names[0]
                    
            except:
                
                img_name = str(row['data']).strip()
                if not img_name.lower().endswith('.jpg'):
                    img_name += '.jpg'

            img_path = os.path.join(self.img_dir, img_name)
            
            center_x = float(row['Fovea_X'])
            center_y = float(row['Fovea_Y'])
        else:
            img_name = self.img_files[idx]
            img_path = os.path.join(self.img_dir, img_name)
            
            results = self.yolo(img_path, verbose=False)
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0].xywh.cpu().numpy()[0] # x_c, y_c, w, h
                center_x, center_y = box[0], box[1]
            else:
                w_orig, h_orig = Image.open(img_path).size
                center_x, center_y = w_orig/2, h_orig/2

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            
            base = os.path.basename(img_path).split('.')[0] 
           
            new_path = os.path.join(os.path.dirname(img_path), base.zfill(4) + '.jpg')
            image = Image.open(new_path).convert('RGB')

        w_orig, h_orig = image.size
        
        left = max(0, int(center_x - self.roi_half))
        top = max(0, int(center_y - self.roi_half))
        right = min(w_orig, int(center_x + self.roi_half))
        bottom = min(h_orig, int(center_y + self.roi_half))
        
        img_cropped = image.crop((left, top, right, bottom))
        img_cropped = img_cropped.resize((CONFIG['crop_size'], CONFIG['crop_size']))
        
        if self.transform:
            img_tensor = self.transform(img_cropped)
            
        if self.mode == 'train':
            real_x = float(row['Fovea_X'])
            real_y = float(row['Fovea_Y'])
            
            local_x = real_x - left
            local_y = real_y - top
            
            crop_w_real = right - left
            crop_h_real = bottom - top
            
            scale_x = CONFIG['crop_size'] / crop_w_real
            scale_y = CONFIG['crop_size'] / crop_h_real
            
            local_x_final = local_x * scale_x
            local_y_final = local_y * scale_y
            
            label = torch.tensor([
                local_x_final / CONFIG['crop_size'],
                local_y_final / CONFIG['crop_size']
            ], dtype=torch.float32)
            
            return img_tensor, label
            
        else:
            crop_w_real = right - left
            crop_h_real = bottom - top
            info = torch.tensor([left, top, crop_w_real, crop_h_real])
            return img_tensor, info, w_orig, h_orig, img_name

if not os.path.exists(CONFIG['yolo_path']):
    print("Warning: 没找到训练好的YOLO模型，正在自动下载预训练模型兜底...")
    yolo_model = YOLO('yolov8n.pt') 
else:
    print("Loading YOLO model...")
    yolo_model = YOLO(CONFIG['yolo_path'])

train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class FoveaRegressor(nn.Module):
    def __init__(self):
        super(FoveaRegressor, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.backbone(x)

model = FoveaRegressor().to(CONFIG['device'])
optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
criterion = nn.MSELoss()

train_dataset = CascadeDataset(
    os.path.join(CONFIG['data_dir'], 'train'), 
    CONFIG['train_csv'], 
    mode='train', 
    transform=train_transforms
)
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)

print("Step 2: Training ResNet on Cropped Images...")

for epoch in range(CONFIG['epochs']):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(CONFIG['device'])
        labels = labels.to(CONFIG['device'])
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {running_loss/len(train_loader):.4f}")

print("Step 3: Prediction & Coordinate Restoration...")
model.eval()

test_dataset = CascadeDataset(
    os.path.join(CONFIG['data_dir'], 'test'), 
    mode='test', 
    transform=val_transforms,
    yolo_model=yolo_model 
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

submission_rows = []

with torch.no_grad():
    for images, info, w_orig, h_orig, img_name in test_loader:
        images = images.to(CONFIG['device'])
        
        preds = model(images).cpu().numpy()[0]
        
        
        left = info[0, 0].item()
        top = info[0, 1].item()
        crop_w = info[0, 2].item()
        crop_h = info[0, 3].item()
        
        pred_x_local = preds[0] * CONFIG['crop_size']
        pred_y_local = preds[1] * CONFIG['crop_size']
        
        pred_x_real_crop = pred_x_local * (crop_w / CONFIG['crop_size'])
        pred_y_real_crop = pred_y_local * (crop_h / CONFIG['crop_size'])
        
        pred_x_global = left + pred_x_real_crop
        pred_y_global = top + pred_y_real_crop
        
        pred_x_global = max(0, min(w_orig.item(), pred_x_global))
        pred_y_global = max(0, min(h_orig.item(), pred_y_global))
        
        fname = img_name[0]
        clean_id = fname.split('.')[0]
        try: clean_id = str(int(clean_id)) 
        except: pass
        
        submission_rows.append({'ImageID': f"{clean_id}_Fovea_X", 'value': pred_x_global})
        submission_rows.append({'ImageID': f"{clean_id}_Fovea_Y", 'value': pred_y_global})

df_sub = pd.DataFrame(submission_rows)
df_sub = df_sub[['ImageID', 'value']]
df_sub.to_csv('submission_cascade.csv', index=False)
print("Done! Saved to 'submission_cascade.csv'")
