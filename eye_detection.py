import os
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns

class EyeStateDetector:
    def __init__(self, dataset_path="Eye_dataset", img_size=(64, 64)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.model = None
        self.history = None
        
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
        
        for img_path, label in zip(image_paths, labels):
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
        
        # Reshape cho CNN (thêm channel dimension)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
        print(f"Hình dạng X: {X.shape}")
        print(f"Hình dạng y: {y.shape}")
        
        return X, y
    
    def create_model(self):
        """Tạo mô hình CNN"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train mô hình"""
        print("Bắt đầu training...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('best_eye_model.h5', monitor='val_accuracy', save_best_only=True)
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Đánh giá mô hình"""
        print("Đang đánh giá mô hình...")
        
        # Predict
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Classification report
        print("\nBáo cáo phân loại:")
        print(classification_report(y_test, y_pred, target_names=['Mắt nhắm', 'Mắt mở']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Mắt nhắm', 'Mắt mở'],
                   yticklabels=['Mắt nhắm', 'Mắt mở'])
        plt.title('Confusion Matrix - Phân loại trạng thái mắt')
        plt.ylabel('Thực tế')
        plt.xlabel('Dự đoán')
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return y_pred, y_pred_proba
    
    def plot_training_history(self):
        """Vẽ biểu đồ quá trình training"""
        if self.history is None:
            print("Chưa có lịch sử training!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0,0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0,0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0,0].set_title('Model Accuracy')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()
        
        # Loss
        axes[0,1].plot(self.history.history['loss'], label='Training Loss')
        axes[0,1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0,1].set_title('Model Loss')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        
        # Precision
        axes[1,0].plot(self.history.history['precision'], label='Training Precision')
        axes[1,0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1,0].set_title('Model Precision')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].legend()
        
        # Recall
        axes[1,1].plot(self.history.history['recall'], label='Training Recall')
        axes[1,1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1,1].set_title('Model Recall')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Recall')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def predict_single_image(self, image_path):
        """Dự đoán cho một ảnh đơn lẻ"""
        try:
            # Load và preprocess ảnh
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, self.img_size)
            img = img.astype('float32') / 255.0
            img = img.reshape(1, img.shape[0], img.shape[1], 1)
            
            # Predict
            prediction = self.model.predict(img)
            confidence = prediction[0][0]
            
            if confidence > 0.5:
                result = "Mắt mở"
                conf_percent = confidence * 100
            else:
                result = "Mắt nhắm"
                conf_percent = (1 - confidence) * 100
            
            return result, conf_percent
            
        except Exception as e:
            return f"Lỗi: {str(e)}", 0
    
    def save_model(self, filepath='eye_state_model.h5'):
        """Lưu mô hình"""
        if self.model:
            self.model.save(filepath)
            print(f"Đã lưu mô hình tại: {filepath}")
    
    def load_model(self, filepath='eye_state_model.h5'):
        """Load mô hình đã lưu"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Đã load mô hình từ: {filepath}")

def main():
    """Hàm chính để chạy toàn bộ pipeline"""
    
    # Khởi tạo detector
    detector = EyeStateDetector()
    
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
    model.summary()
    
    # Train mô hình
    detector.train_model(X_train, y_train, X_val, y_val, epochs=50)
    
    # Vẽ biểu đồ training
    detector.plot_training_history()
    
    # Đánh giá mô hình
    y_pred, y_pred_proba = detector.evaluate_model(X_test, y_test)
    
    # Lưu mô hình
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
