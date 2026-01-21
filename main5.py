import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

CONFIG = {
    "train_img_dir": "./segmentation/train/image",
    "train_mask_dir": "./segmentation/train/label",
    "test_img_dir": "./segmentation/test/image",
    "output_dir": "./prediction_result/image", 
    "img_size": 512,        
    "batch_size": 4,        
    "lr": 1e-4,
    "epochs": 350,           
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

class RetinalDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None, is_test=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_test = is_test
        self.img_names = os.listdir(img_dir)
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 读取图像 (OpenCV默认BGR，转RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.is_test:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, img_name
        
        mask_name = img_name.replace('.jpg', '.png') 
        
        if not os.path.exists(os.path.join(self.mask_dir, mask_name)):
            mask_name = img_name 
            
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask

def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
           
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),

            A.OneOf([
              
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=1.0),
            ], p=0.7),

            A.OneOf([
                
                A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=1.0),
                
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.5),

            A.OneOf([
               
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
               
                A.GaussianBlur(blur_limit=3, p=1.0),
               
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, fill_value=0, mask_fill_value=0, p=0.3),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
      
        return A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


def train_model():
    # 4.1 准备数据
    train_dataset = RetinalDataset(CONFIG['train_img_dir'], CONFIG['train_mask_dir'], transform=get_transforms('train'))
    # 实际项目中建议划分验证集，这里因为数据太少(20张)，直接全部用于训练
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)

    # 4.2 定义模型 (U-Net + ResNet34)
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1
    )
    model.to(CONFIG['device'])

    # 4.3 优化器与损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    # 组合损失：BCE (像素级分类) + Dice (整体重合度)
    loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
    
    # 4.4 训练循环
    print("Start Training...")
    model.train()
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for images, masks in pbar:
            images = images.to(CONFIG['device'])
            masks = masks.to(CONFIG['device']).float().unsqueeze(1) # [B, 1, H, W]
            
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
    # 保存模型
    torch.save(model.state_dict(), "best_model.pth")
    print("Training Finished. Model saved.")
    return model

# ===========================
# 5. 推理与生成 CSV
# ===========================
def rle_encoding(x):
    """
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as string formatted
    """
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join([str(r) for r in run_lengths])

def predict_and_submit(model):
    print("Start Predicting...")
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
        
    test_dataset = RetinalDataset(CONFIG['test_img_dir'], is_test=True, transform=get_transforms('test'))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model.eval()
    submission_data = []

    with torch.no_grad():
        for image, name in tqdm(test_loader):
            image = image.to(CONFIG['device'])
            # 预测
            logits = model(image)
            prob = torch.sigmoid(logits) # 映射到 0-1
            pred_mask = (prob > 0.5).float().cpu().numpy().squeeze() # 阈值化
            
            # --- 保存预测图片 (可选，用于肉眼检查) ---
            # 必须保证血管是255，背景是0
            save_img = (pred_mask * 255).astype(np.uint8)
            save_path = os.path.join(CONFIG['output_dir'], name[0])
            # 注意：保存时如果原文件名是jpg，这里保存为png以防压缩损失
            if save_path.endswith('.jpg'):
                save_path = save_path.replace('.jpg', '.png')
            cv2.imwrite(save_path, save_img)
            
            # --- 直接生成 CSV 数据 (类似 segmentation_to_csv.py 的逻辑) ---
            # 我们的 pred_mask 已经是 (血管=1, 背景=0)，符合 rle_encoding 的输入
            rle_str = rle_encoding(pred_mask)
            
            # ID 需要去掉后缀
            img_id = name[0].split('.')[0]
            submission_data.append([img_id, rle_str])
            
    # 保存 CSV
    df = pd.DataFrame(submission_data, columns=['Id', 'Predicted'])
    df.to_csv('submission.csv', index=False)
    print("Submission file saved to 'submission.csv'.")

# ===========================
# 主程序
# ===========================
if __name__ == "__main__":
    # 1. 训练
    trained_model = train_model()
    
    # 2. 如果只是想推理（加载已有权重）：
    # trained_model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
    # trained_model.load_state_dict(torch.load("best_model.pth"))
    # trained_model.to(CONFIG['device'])
    
    # 3. 预测并生成CSV
    predict_and_submit(trained_model)