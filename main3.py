import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import copy

CONFIG = {
    'root_dir': os.getcwd(),
    'data_dir': './detection',
    'train_csv': 'fovea_localization_train_GT.csv',
    
    'output_dir': './output_exp_final',
    
    'img_size': 512,       
    'crop_size': 512,       
    'roi_ratio': 0.30,      
    'crop_size': 512,       
    'batch_size': 16, 
    'lr': 1e-4,      
    'epochs': 50,          
    'n_splits': 5,         
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
print(f"🚀 Using device: {CONFIG['device']}")

class FoveaDataset(Dataset):
    def __init__(self, img_dir, data_df, mode='train', transform=None, yolo_model=None):
        self.img_dir = img_dir
        self.data = data_df
        self.mode = mode
        self.transform = transform
        self.yolo = yolo_model
        
        self.img_sizes = {} 
        if self.mode == 'test':
            self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    def __len__(self):
        if self.mode == 'train' or self.mode == 'val':
            return len(self.data)
        else:
            return len(self.img_files)

    def _fix_filename(self, raw_name):
        
        try:
            base = str(int(float(raw_name))).zfill(4)
        except:
            base = str(raw_name).strip().split('.')[0]
        
        for name in [f"{base}.jpg", f"{int(base)}.jpg"]:
            if os.path.exists(os.path.join(self.img_dir, name)):
                return name
        return f"{base}.jpg" # Default

    def __getitem__(self, idx):
        
        if self.mode == 'train' or self.mode == 'val':
            row = self.data.iloc[idx]
            img_name = self._fix_filename(row['data'])
            img_path = os.path.join(self.img_dir, img_name)
            
            center_x = float(row['Fovea_X'])
            center_y = float(row['Fovea_Y'])
        else:
            img_name = self.img_files[idx]
            img_path = os.path.join(self.img_dir, img_name)
            
            results = self.yolo(img_path, verbose=False)
            if len(results[0].boxes) > 0:
                
                box = results[0].boxes[0].xywh.cpu().numpy()[0]
                center_x, center_y = box[0], box[1]
            else:
                
                img_tmp = Image.open(img_path)
                center_x, center_y = img_tmp.width / 2, img_tmp.height / 2

        image = np.array(Image.open(img_path).convert('RGB'))
        h_orig, w_orig = image.shape[:2]
        
        roi_w = int(w_orig * CONFIG['roi_ratio'])
        roi_h = int(h_orig * CONFIG['roi_ratio'])
        roi_half = roi_w // 2
        
        left = max(0, int(center_x - roi_half))
        top = max(0, int(center_y - roi_half))
        right = min(w_orig, left + roi_w)
        bottom = min(h_orig, top + roi_h)
        
        if right - left < roi_w:
            left = max(0, right - roi_w)
        if bottom - top < roi_h:
            top = max(0, bottom - roi_h)
           
        crop_img = image[top:bottom, left:right]
        
        if self.transform:
            augmented = self.transform(image=crop_img)
            img_tensor = augmented['image']
        else:
            img_tensor = transforms.ToTensor()(crop_img)

        if self.mode == 'train' or self.mode == 'val':
            real_x = float(row['Fovea_X'])
            real_y = float(row['Fovea_Y'])
            
            local_x = real_x - left
            local_y = real_y - top
            
            label = torch.tensor([
                local_x / (right - left),
                local_y / (bottom - top)
            ], dtype=torch.float32)
            
            return img_tensor, label
        else:
           
            info = torch.tensor([left, top, right-left, bottom-top]) # x, y, w, h
            return img_tensor, info, w_orig, h_orig, img_name

class FoveaRegressor(nn.Module):
    def __init__(self):
        super(FoveaRegressor, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),  
            nn.ReLU(),
            nn.Dropout(0.5),           
            nn.Linear(1024, 256),     
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.backbone(x)

def prepare_and_train_yolo():
    print(">>> Stage 1: Training YOLOv8 for ROI Detection...")
    
    yolo_dir = os.path.join(CONFIG['root_dir'], 'yolo_data_kfold')
    if os.path.exists(yolo_dir): shutil.rmtree(yolo_dir)
    
    os.makedirs(f"{yolo_dir}/images/train", exist_ok=True)
    os.makedirs(f"{yolo_dir}/labels/train", exist_ok=True)
    
    df = pd.read_csv(CONFIG['train_csv'])
    train_dir = os.path.join(CONFIG['data_dir'], 'train')
    
    for _, row in df.iterrows():
        try:
            fname = str(int(float(row['data']))).zfill(4) + '.jpg'
        except:
            fname = str(row['data'])
            
        src = os.path.join(train_dir, fname)
        if not os.path.exists(src): continue
        
        shutil.copy(src, f"{yolo_dir}/images/train/{fname}")
        
        w_img, h_img = Image.open(src).size
        norm_x = float(row['Fovea_X']) / w_img
        norm_y = float(row['Fovea_Y']) / h_img
        
        with open(f"{yolo_dir}/labels/train/{fname.replace('.jpg','.txt')}", 'w') as f:
            f.write(f"0 {norm_x} {norm_y} {CONFIG['roi_ratio']} {CONFIG['roi_ratio']}\n")
            
    # Config
    with open('yolo_kfold.yaml', 'w') as f:
        f.write(f"path: {yolo_dir}\ntrain: images/train\nval: images/train\nnames: {{0: fovea}}")
        
    
    model = YOLO('yolov8n.pt')
    
    model.train(data='yolo_kfold.yaml', epochs=100, imgsz=512, plots=False, verbose=False)
    return model

if __name__ == '__main__':
    
    yolo_model = prepare_and_train_yolo()
    
    print("\n>>> Stage 2: 5-Fold Cross Validation Training...")
    
    train_aug = A.Compose([
        A.Resize(CONFIG['crop_size'], CONFIG['crop_size']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussianBlur(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_aug = A.Compose([
        A.Resize(CONFIG['crop_size'], CONFIG['crop_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    full_df = pd.read_csv(CONFIG['train_csv'])
    kf = KFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=42)
    
    best_models = [] 
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_df)):
        print(f"\n------ Fold {fold+1}/{CONFIG['n_splits']} ------")
        
        train_sub = full_df.iloc[train_idx]
        val_sub = full_df.iloc[val_idx]
        
        train_ds = FoveaDataset(os.path.join(CONFIG['data_dir'], 'train'), train_sub, mode='train', transform=train_aug)
        val_ds = FoveaDataset(os.path.join(CONFIG['data_dir'], 'train'), val_sub, mode='val', transform=val_aug)
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
        
        
        model = FoveaRegressor().to(CONFIG['device'])
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
        criterion = nn.MSELoss() # 或者 nn.SmoothL1Loss()
        
        best_loss = float('inf')
        save_path = os.path.join(CONFIG['output_dir'], f'resnet50_fold{fold}.pth')
        
        for epoch in range(CONFIG['epochs']):
            model.train()
            train_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            scheduler.step()
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
                    preds = model(imgs)
                    loss = criterion(preds, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                # print(f"  Epoch {epoch+1} Saved Best: {best_loss:.5f}")
            
        print(f"Fold {fold+1} Finished. Best Val Loss: {best_loss:.5f}")
        best_models.append(save_path)

    print("\n>>> Stage 3: Ensemble Prediction with TTA...")
    
    test_ds = FoveaDataset(
        os.path.join(CONFIG['data_dir'], 'test'), 
        None, mode='test', 
        transform=val_aug, 
        yolo_model=yolo_model
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    models_list = []
    for path in best_models:
        m = FoveaRegressor().to(CONFIG['device'])
        m.load_state_dict(torch.load(path))
        m.eval()
        models_list.append(m)
        
    submission = []
    
    with torch.no_grad():
        for i, (imgs, info, w_orig, h_orig, img_name) in enumerate(test_loader):
            imgs = imgs.to(CONFIG['device'])
            
            # info: [batch, 4] -> [x, y, w, h]
            left = info[0, 0].item()
            top = info[0, 1].item()
            crop_w = info[0, 2].item()
            crop_h = info[0, 3].item()
            
            # Ensemble 预测
            fold_preds = []
            for m in models_list:
                # 1. 原图预测
                pred_orig = m(imgs).cpu().numpy()[0]
                
                # 2. 水平翻转预测 (TTA)
                imgs_flip = torch.flip(imgs, [3]) # Flip width dimension
                pred_flip = m(imgs_flip).cpu().numpy()[0]
                pred_flip[0] = 1.0 - pred_flip[0] # 翻转回坐标
                
                # 平均 TTA
                avg_pred = (pred_orig + pred_flip) / 2.0
                fold_preds.append(avg_pred)
            
            final_pred_local = np.mean(fold_preds, axis=0)
            
            px_local = final_pred_local[0] * crop_w
            py_local = final_pred_local[1] * crop_h
            
            px_global = left + px_local
            py_global = top + py_local
            
            fname = img_name[0].split('.')[0]
            try: fname = str(int(fname))
            except: pass
            
            submission.append({'ImageID': f"{fname}_Fovea_X", 'value': px_global})
            submission.append({'ImageID': f"{fname}_Fovea_Y", 'value': px_global})

            submission[-1]['value'] = py_global
            
            if i % 10 == 0: print(f"Predicted {i}/{len(test_ds)}")

    df_sub = pd.DataFrame(submission)
    df_sub.to_csv('submission_kfold_tta.csv', index=False)
    print("\n✅ All Done! Submission saved to 'submission_kfold_tta.csv'")