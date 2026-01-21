import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import time

CONFIG = {
    'data_dir': './detection',  
    'train_csv': 'fovea_localization_train_GT.csv', 
    'img_size': 512,            
    'batch_size': 8,
    'lr': 1e-4,
    'epochs': 50,               
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Using device: {CONFIG['device']}")

class FoveaDataset(Dataset):
    def __init__(self, img_dir, csv_file=None, mode='train', transform=None):
        self.img_dir = img_dir
        self.mode = mode
        self.transform = transform
        
        if mode == 'train':
            self.data = pd.read_csv(csv_file, dtype={'data': str})
        else:
            self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        else:
            return len(self.img_files)

    def __getitem__(self, idx):
        if self.mode == 'train':
            row = self.data.iloc[idx]
            
            img_name = str(row['data']).strip()
            
            if not img_name.lower().endswith('.jpg'):
                img_name = img_name + '.jpg'
            
            raw_x = float(row['Fovea_X'])
            raw_y = float(row['Fovea_Y'])
            img_path = os.path.join(self.img_dir, img_name)
        else:
            img_name = self.img_files[idx]
            img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
           
            if self.mode == 'train':
                base_name = str(row['data']).strip()
                alt_name = base_name.zfill(4) 
                if not alt_name.lower().endswith('.jpg'):
                    alt_name += '.jpg'
                img_path = os.path.join(self.img_dir, alt_name)
                image = Image.open(img_path).convert('RGB')
            else:
                raise

        w_orig, h_orig = image.size

        image_resized = image.resize((CONFIG['img_size'], CONFIG['img_size']))
        
        if self.transform:
            image_tensor = self.transform(image_resized)
        
        if self.mode == 'train':
            
            x_resized = raw_x * (CONFIG['img_size'] / w_orig)
            y_resized = raw_y * (CONFIG['img_size'] / h_orig)
            
            label = torch.tensor([
                x_resized / CONFIG['img_size'], 
                y_resized / CONFIG['img_size']
            ], dtype=torch.float32)
            
            return image_tensor, label
        else:
            return image_tensor, w_orig, h_orig, img_name

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet 标准均值方差
])

train_img_dir = os.path.join(CONFIG['data_dir'], 'train')
test_img_dir = os.path.join(CONFIG['data_dir'], 'test')

full_dataset = FoveaDataset(train_img_dir, CONFIG['train_csv'], mode='train', transform=data_transforms)


train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

class FoveaRegressor(nn.Module):
    def __init__(self):
        super(FoveaRegressor, self).__init__()
    
        self.backbone = models.resnet18(pretrained=True)
        
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(128, 2)  
        )

    def forward(self, x):
        return self.backbone(x)

model = FoveaRegressor().to(CONFIG['device'])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

print("Start Training...")
best_val_loss = float('inf')

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
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / train_size
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
    
    avg_val_loss = val_loss / val_size
    
    print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Train Loss: {epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')

print("Training Complete.")

print("Generating Submission...")

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

test_dataset = FoveaDataset(test_img_dir, mode='test', transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

submission_rows = []

with torch.no_grad():
    for images, w_orig, h_orig, img_name in test_loader:
        images = images.to(CONFIG['device'])
        
        outputs = model(images).cpu().numpy()[0] # [x_pred, y_pred]
        
        pred_x = outputs[0] * (w_orig.item() / CONFIG['img_size'])
        pred_y = outputs[1] * (h_orig.item() / CONFIG['img_size'])
        
        pred_x = max(0, pred_x)
        pred_y = max(0, pred_y)
        
        fname = img_name[0]
        
        submission_rows.append({
            'ImageID': f"{fname}_Fovea_X",
            'value': pred_x
        })
        submission_rows.append({
            'ImageID': f"{fname}_Fovea_Y",
            'value': pred_y
        })

df_submission = pd.DataFrame(submission_rows)
df_submission = df_submission[['ImageID', 'value']] 
df_submission.to_csv('submission.csv', index=False)

print(f"Submission saved to submission.csv. Total rows: {len(df_submission)}")
print("Sample check:")
print(df_submission.head())