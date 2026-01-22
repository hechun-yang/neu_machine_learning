import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-v0_8')
sns.set_palette("muted")

def main():
    print("="*60)
    print("项目启动：乳腺癌诊断分类分析 (Classification Task)")
    print("="*60)

    print("\n[Step 1] 数据加载与探索...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target 

    class_names = data.target_names
    
    print(f"数据维度: {df.shape}")
    print(f"类别标签: {class_names} (0: 恶性, 1: 良性)")
 
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df)
    plt.title('Class Distribution (0: Malignant, 1: Benign)')
    plt.xlabel('Target Class')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x='mean radius', hue='target', fill=True, common_norm=False)
    plt.title('Distribution of Mean Radius by Diagnosis')
    plt.show()
    print("\n[Step 2] 建模与训练...")
    
    X = df.drop('target', axis=1)
    y = df['target']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    
    clf.fit(X_train, y_train)

    print("\n[Step 3] 深度评估与分析...")
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] # 获取预测为“良性”的概率


    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names))


    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:10] # 取前10个
    
    plt.figure(figsize=(10, 6))
    plt.title('Top 10 Feature Importances')
    plt.bar(range(10), importances[indices], align='center')
    plt.xticks(range(10), [data.feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
