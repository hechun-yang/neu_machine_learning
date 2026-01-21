import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

CONFIG = {
    'root_dir': os.getcwd(),
    'data_dir': './detection',
    'train_csv': 'fovea_localization_train_GT.csv',
    'output_dir': './output_unet_heatmap',
    
    'img_size': 512,        
    'crop_size': 512,       
    'roi_ratio': 0.30,      
    'heatmap_sigma': 15,    
    
    'batch_size': 8,        
    'lr': 3e-4,
    'epochs': 40,
    'n_splits': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
print(f"🚀 Using device: {CONFIG['device']}")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512) # 512+512=1024 in -> 512 out
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)   # 256+256=512 in -> 256 out
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
       
        self.conv_up3 = DoubleConv(256, 128)   
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.conv_up4 = DoubleConv(128, 64)    
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1) # 128 + 128 = 256 channels here
        x = self.conv_up3(x)          # Now DoubleConv(256, 128) handles it correctly
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up4(x)
        
        logits = self.outc(x)
        return logits

class FoveaHeatmapDataset(Dataset):
    def __init__(self, img_dir, data_df, mode='train', transform=None, yolo_model=None):
        self.img_dir = img_dir
        self.data = data_df
        self.mode = mode
        self.transform = transform
        self.yolo = yolo_model
        
        if self.mode == 'test':
            self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.data) if self.mode != 'test' else len(self.img_files)

    def _fix_filename(self, raw_name):
        try: base = str(int(float(raw_name))).zfill(4)
        except: base = str(raw_name).strip().split('.')[0]
        for name in [f"{base}.jpg", f"{int(base)}.jpg"]:
            if os.path.exists(os.path.join(self.img_dir, name)): return name
        return f"{base}.jpg"

    def _generate_gaussian(self, size, center, sigma):
        w, h = size
        x = np.arange(0, w, 1, float)
        y = np.arange(0, h, 1, float)
        y = y[:, np.newaxis]
        x0, y0 = center
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    def __getitem__(self, idx):
        if self.mode != 'test':
            row = self.data.iloc[idx]
            img_path = os.path.join(self.img_dir, self._fix_filename(row['data']))
            cx, cy = float(row['Fovea_X']), float(row['Fovea_Y'])
        else:
            img_name = self.img_files[idx]
            img_path = os.path.join(self.img_dir, img_name)
            results = self.yolo(img_path, verbose=False)
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0].xywh.cpu().numpy()[0]
                cx, cy = box[0], box[1]
            else:
                tmp = Image.open(img_path)
                cx, cy = tmp.width / 2, tmp.height / 2

        image = np.array(Image.open(img_path).convert('RGB'))
        h_orig, w_orig = image.shape[:2]
        
        roi_w = int(w_orig * CONFIG['roi_ratio'])
        roi_h = int(h_orig * CONFIG['roi_ratio'])
        half = roi_w // 2
        
        left = max(0, int(cx - half))
        top = max(0, int(cy - half))
        right = min(w_orig, left + roi_w)
        bottom = min(h_orig, top + roi_h)
        
        if right - left < roi_w: left = max(0, right - roi_w)
        if bottom - top < roi_h: top = max(0, bottom - roi_h)
        
        crop_img = image[top:bottom, left:right]
        crop_h_real, crop_w_real = crop_img.shape[:2]

        mask = np.zeros((crop_h_real, crop_w_real), dtype=np.float32)
        
        if self.mode != 'test':
            local_x = float(row['Fovea_X']) - left
            local_y = float(row['Fovea_Y']) - top
            if 0 <= local_x < crop_w_real and 0 <= local_y < crop_h_real:
                mask = self._generate_gaussian((crop_w_real, crop_h_real), (local_x, local_y), CONFIG['heatmap_sigma'])

        if self.transform:
            transformed = self.transform(image=crop_img, mask=mask)
            img_tensor = transformed['image']
            mask_tensor = transformed['mask']
        else:
            img_tensor = transforms.ToTensor()(crop_img)
            mask_tensor = torch.from_numpy(mask)

        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
            
        if self.mode != 'test':
            return img_tensor, mask_tensor
        else:
            info = torch.tensor([left, top, crop_w_real, crop_h_real]) 
            return img_tensor, info, w_orig, h_orig, img_name

# ================= 3. YOLO 准备 =================
def prepare_and_train_yolo():
    
    best_weights = os.path.join(CONFIG['root_dir'], 'runs/detect/train5/weights/best.pt')
    # if os.path.exists(best_weights):
        # print(f">>> YOLO model found at {best_weights}, skipping training.")
        # return YOLO(best_weights)

    print(">>> Stage 1: Training YOLOv8...")
    yolo_dir = os.path.join(CONFIG['root_dir'], 'yolo_data_unet')
    if os.path.exists(yolo_dir): shutil.rmtree(yolo_dir)
    os.makedirs(f"{yolo_dir}/images/train", exist_ok=True)
    os.makedirs(f"{yolo_dir}/labels/train", exist_ok=True)
    
    df = pd.read_csv(CONFIG['train_csv'])
    df_sample = df.sample(frac=1.0, random_state=42) 
    
    for _, row in df_sample.iterrows():
        try: fname = str(int(float(row['data']))).zfill(4) + '.jpg'
        except: fname = str(row['data'])
        src = os.path.join(CONFIG['data_dir'], 'train', fname)
        if not os.path.exists(src): continue
        shutil.copy(src, f"{yolo_dir}/images/train/{fname}")
        w, h = Image.open(src).size
        nx, ny = float(row['Fovea_X'])/w, float(row['Fovea_Y'])/h
        with open(f"{yolo_dir}/labels/train/{fname.replace('.jpg','.txt')}", 'w') as f:
            f.write(f"0 {nx} {ny} {CONFIG['roi_ratio']} {CONFIG['roi_ratio']}\n")
            
    with open('yolo_unet.yaml', 'w') as f:
        f.write(f"path: {yolo_dir}\ntrain: images/train\nval: images/train\nnames: {{0: fovea}}")
    
    model = YOLO('yolov8n.pt')
    model.train(data='yolo_unet.yaml', epochs=100, imgsz=512, verbose=False, plots=False)
    return model

if __name__ == '__main__':
    # --- Step 1: YOLO ---
    
    yolo_model = prepare_and_train_yolo()

    # --- Step 2: K-Fold UNet ---
    print("\n>>> Stage 2: UNet Heatmap Training...")
    
    train_aug = A.Compose([
        A.Resize(CONFIG['crop_size'], CONFIG['crop_size']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5), 
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
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
        
        train_ds = FoveaHeatmapDataset(
            os.path.join(CONFIG['data_dir'], 'train'), 
            full_df.iloc[train_idx], mode='train', transform=train_aug
        )
        val_ds = FoveaHeatmapDataset(
            os.path.join(CONFIG['data_dir'], 'train'), 
            full_df.iloc[val_idx], mode='val', transform=val_aug
        )
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
        
        model = UNet(n_channels=3, n_classes=1).to(CONFIG['device'])
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        save_path = os.path.join(CONFIG['output_dir'], f'unet_fold{fold}.pth')
        
        for epoch in range(CONFIG['epochs']):
            model.train()
            train_loss = 0
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(CONFIG['device']), masks.to(CONFIG['device'])
                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, masks.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            scheduler.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(CONFIG['device']), masks.to(CONFIG['device'])
                    preds = model(imgs)
                    loss = criterion(preds, masks.float())
                    val_loss += loss.item()
            
            avg_val = val_loss/len(val_loader)
            if avg_val < best_loss:
                best_loss = avg_val
                torch.save(model.state_dict(), save_path)
        
        print(f"Fold {fold+1} Best Loss: {best_loss:.6f}")
        best_models.append(save_path)

    print("\n>>> Stage 3: Heatmap Decoding & Submission...")
    
    test_ds = FoveaHeatmapDataset(
        os.path.join(CONFIG['data_dir'], 'test'), None, mode='test', 
        transform=val_aug, yolo_model=yolo_model
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    loaded_models = [UNet().to(CONFIG['device']) for _ in range(len(best_models))]
    for m, p in zip(loaded_models, best_models):
        m.load_state_dict(torch.load(p))
        m.eval()
        
    submission = []
    
    with torch.no_grad():
        for i, (img, info, w_orig, h_orig, img_name) in enumerate(test_loader):
            img = img.to(CONFIG['device'])
            
            avg_heatmap = torch.zeros((1, 1, CONFIG['crop_size'], CONFIG['crop_size'])).to(CONFIG['device'])
            for m in loaded_models:
                pred = m(img)
                pred_flip = torch.flip(m(torch.flip(img, [3])), [3]) 
                avg_heatmap += (pred + pred_flip) / 2.0
            avg_heatmap /= len(loaded_models)
            
            hmap_np = avg_heatmap[0, 0].cpu().numpy()
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hmap_np)
            pred_x_local, pred_y_local = max_loc 
            
            left, top, crop_w, crop_h = info[0]
            left, top, crop_w, crop_h = left.item(), top.item(), crop_w.item(), crop_h.item()
            
            scale_x = crop_w / CONFIG['crop_size']
            scale_y = crop_h / CONFIG['crop_size']
            
            final_x = left + pred_x_local * scale_x
            final_y = top + pred_y_local * scale_y
            
            fname = img_name[0].split('.')[0]
            try: fname = str(int(fname))
            except: pass
            
            submission.append({'ImageID': f"{fname}_Fovea_X", 'value': final_x})
            submission.append({'ImageID': f"{fname}_Fovea_Y", 'value': final_y})
            
            if i % 50 == 0: print(f"Processing {i}...")

    df_sub = pd.DataFrame(submission)
    df_sub.to_csv('submission_unet_heatmap.csv', index=False)
    print("\n✅ Done! Saved to submission_unet_heatmap.csv")