import os
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class EyeDataset(Dataset):
    """Custom Dataset cho PyTorch"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor
        image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension
        label = torch.LongTensor([label])
        
        return image, label

class EyeCNN(nn.Module):
    """CNN Model cho nhận diện trạng thái mắt"""
    def __init__(self, img_size=(64, 64)):
        super(EyeCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Calculate size after convolutions
        conv_output_size = img_size[0] // 8 * img_size[1] // 8 * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 128)
        self.dropout5 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 2)  # Binary classification
        
    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout4(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout5(x)
        
        x = self.fc3(x)
        
        return x

class EyeStateDetectorPyTorch:
    def __init__(self, dataset_path="Eye_dataset", img_size=(64, 64), device=None):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"Sử dụng device: {self.device}")
        
    def load_dataset(self):
        """Load và parse dataset từ các thư mục con"""
        print("Đang tải dataset...")
        
        # Tìm tất cả file ảnh
        image_paths = []
        labels = []
        
        # Duyệt qua tất cả thư mục con s00XX
        for subject_dir in glob.glob(os.path.join(self.dataset_path, "s*")):
            if os.path.isdir(subject_dir):
                # Lấy tất cả file .png trong thư mục
                png_files = glob.glob(os.path.join(subject_dir, "*.png"))
                
                for img_path in png_files:
                    filename = os.path.basename(img_path)
                    
                    # Parse tên file: sXXX_XXXXX_gender_glasses_eyestate_reflections_lighting_sensortype.png
                    parts = filename.replace('.png', '').split('_')
                    
                    if len(parts) >= 5:
                        eye_state = int(parts[4])  # Vị trí thứ 4 là eye state
                        image_paths.append(img_path)
                        labels.append(eye_state)  # 0 = nhắm, 1 = mở
        
        print(f"Tổng số ảnh: {len(image_paths)}")
        print(f"Mắt nhắm (0): {labels.count(0)}")
        print(f"Mắt mở (1): {labels.count(1)}")
        
        return image_paths, labels
    
    def preprocess_images(self, image_paths, labels):
        """Tiền xử lý ảnh"""
        print("Đang tiền xử lý ảnh...")
        
        images = []
        valid_labels = []
        
        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
            try:
                # Đọc ảnh
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Chuyển sang grayscale
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Resize về kích thước cố định
                img = cv2.resize(img, self.img_size)
                
                # Normalize pixel values
                img = img.astype('float32') / 255.0
                
                images.append(img)
                valid_labels.append(label)
                
            except Exception as e:
                print(f"Lỗi xử lý ảnh {img_path}: {str(e)}")
                continue
        
        # Convert sang numpy arrays
        X = np.array(images)
        y = np.array(valid_labels)
        
        print(f"Hình dạng X: {X.shape}")
        print(f"Hình dạng y: {y.shape}")
        
        return X, y
    
    def create_model(self):
        """Tạo mô hình CNN PyTorch"""
        self.model = EyeCNN(img_size=self.img_size).to(self.device)
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001):
        """Train mô hình"""
        print("Bắt đầu training...")
        
        # Tạo datasets
        train_dataset = EyeDataset(X_train, y_train)
        val_dataset = EyeDataset(X_val, y_val)
        
        # Tạo data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss và optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        best_val_accuracy = 0.0
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for images, labels in train_bar:
                images = images.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_accuracy = 100. * train_correct / train_total
            train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for images, labels in val_bar:
                    images = images.to(self.device)
                    labels = labels.squeeze().to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    val_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            val_accuracy = 100. * val_correct / val_total
            val_loss = val_loss / len(val_loader)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            
            # Early stopping và save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_eye_model.pt')
                print(f'  ✅ Saved best model (Val Acc: {val_accuracy:.2f}%)')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        print(f'Training completed! Best validation accuracy: {best_val_accuracy:.2f}%')
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
    
    def evaluate_model(self, X_test, y_test, batch_size=32):
        """Đánh giá mô hình"""
        print("Đang đánh giá mô hình...")
        
        test_dataset = EyeDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing'):
                images = images.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        
        # Classification report
        print("\nBáo cáo phân loại:")
        print(classification_report(y_true, y_pred, target_names=['Mắt nhắm', 'Mắt mở']))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Mắt nhắm', 'Mắt mở'],
                   yticklabels=['Mắt nhắm', 'Mắt mở'])
        plt.title('Confusion Matrix - Phân loại trạng thái mắt')
        plt.ylabel('Thực tế')
        plt.xlabel('Dự đoán')
        plt.savefig('confusion_matrix_pytorch.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return y_pred, y_true
    
    def plot_training_history(self):
        """Vẽ biểu đồ quá trình training"""
        if not self.train_losses:
            print("Chưa có lịch sử training!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.train_losses, label='Training Loss')
        axes[0].plot(self.val_losses, label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(self.train_accuracies, label='Training Accuracy')
        axes[1].plot(self.val_accuracies, label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_pytorch.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def predict_single_image(self, image_path):
        """Dự đoán cho một ảnh đơn lẻ"""
        try:
            # Load và preprocess ảnh
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, self.img_size)
            img = img.astype('float32') / 255.0
            
            # Convert to tensor
            img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(self.device)  # Add batch and channel dims
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                predicted_class = predicted.item()
                confidence = probabilities[0][predicted_class].item() * 100
                
                if predicted_class == 1:
                    result = "Mắt mở"
                else:
                    result = "Mắt nhắm"
            
            return result, confidence
            
        except Exception as e:
            return f"Lỗi: {str(e)}", 0
    
    def save_model(self, filepath='eye_state_model.pt'):
        """Lưu mô hình"""
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'img_size': self.img_size,
                'device': str(self.device)
            }, filepath)
            print(f"Đã lưu mô hình tại: {filepath}")
    
    def load_model(self, filepath='eye_state_model.pt'):
        """Load mô hình đã lưu"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Tạo mô hình với cấu hình đã lưu
        self.img_size = checkpoint.get('img_size', (64, 64))
        self.model = EyeCNN(img_size=self.img_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Đã load mô hình từ: {filepath}")

def main():
    """Hàm chính để chạy toàn bộ pipeline"""
    
    # Khởi tạo detector
    detector = EyeStateDetectorPyTorch()
    
    # Load dataset
    image_paths, labels = detector.load_dataset()
    
    # Preprocess
    X, y = detector.preprocess_images(image_paths, labels)
    
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Training set: {X_train.shape[0]} ảnh")
    print(f"Validation set: {X_val.shape[0]} ảnh")
    print(f"Test set: {X_test.shape[0]} ảnh")
    
    # Tạo mô hình
    model = detector.create_model()
    print("\nKiến trúc mô hình:")
    print(model)
    
    # Train mô hình
    detector.train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    # Vẽ biểu đồ training
    detector.plot_training_history()
    
    # Load best model
    detector.load_model('best_eye_model.pt')
    
    # Đánh giá mô hình
    y_pred, y_true = detector.evaluate_model(X_test, y_test)
    
    # Lưu mô hình cuối cùng
    detector.save_model()
    
    # Test với một vài ảnh
    print("\nTest với vài ảnh mẫu:")
    test_images = image_paths[:5]  # Lấy 5 ảnh đầu tiên để test
    for img_path in test_images:
        result, confidence = detector.predict_single_image(img_path)
        filename = os.path.basename(img_path)
        print(f"{filename}: {result} (Confidence: {confidence:.1f}%)")

if __name__ == "__main__":
    main() 