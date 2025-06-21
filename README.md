# 👁️ Hệ thống nhận diện trạng thái mắt (Mở/Nhắm)

Mô hình Deep Learning sử dụng CNN để phân loại trạng thái mắt từ ảnh.

## 📋 Mô tả Dataset

Dataset được tổ chức theo format:
```
sXXX_XXXXX_gender_glasses_eyestate_reflections_lighting_sensortype.png
```

Trong đó:
- **subject ID**: s001, s002, ...
- **image number**: số thứ tự ảnh
- **gender**: 0 = nam, 1 = nữ
- **glasses**: 0 = không đeo kính, 1 = có đeo kính
- **eye state**: **0 = mắt nhắm, 1 = mắt mở** ← Đây là nhãn chính
- **reflections**: 0 = không phản chiếu, 1 = phản chiếu thấp, 2 = phản chiếu cao
- **lighting**: 0 = ánh sáng kém, 1 = ánh sáng tốt
- **sensor type**: 01 = RealSense SR300, 02 = IDS Imaging, 03 = Aptina

## 🚀 Cài đặt

1. **Cài đặt các thư viện cần thiết:**
```bash
pip install -r requirements.txt
```

2. **Cấu trúc thư mục:**
```
CV-Helmet/
├── Eye_dataset/
│   ├── annotation.txt
│   ├── s0001/
│   ├── s0002/
│   └── ...
├── eye_detection.py
├── train_eye_detection.py
├── requirements.txt
└── README.md
```

## 🎯 Cách sử dụng

### 1. Training mô hình

```bash
python train_eye_detection.py
```

Script sẽ:
- Tự động load và parse tất cả ảnh từ dataset
- Chia dataset thành train/validation/test (70/15/15)
- Train mô hình CNN với Early Stopping
- Lưu mô hình tốt nhất
- Tạo biểu đồ và báo cáo đánh giá

### 2. Test với ảnh đơn lẻ

```bash
python train_eye_detection.py path/to/image.png
```

### 3. Sử dụng trong code

```python
from eye_detection import EyeStateDetector

# Khởi tạo detector
detector = EyeStateDetector()

# Load mô hình đã train
detector.load_model('eye_state_model.h5')

# Predict
result, confidence = detector.predict_single_image('path/to/image.png')
print(f"Kết quả: {result} (Độ tin cậy: {confidence:.1f}%)")
```

## 🏗️ Kiến trúc mô hình

- **Input**: Ảnh grayscale 64x64 pixels
- **CNN**: 3 khối Convolution + BatchNorm + MaxPool + Dropout
- **Dense**: 2 lớp fully connected với dropout
- **Output**: Sigmoid activation (binary classification)
- **Loss**: Binary crossentropy
- **Optimizer**: Adam

## 📊 Kết quả Training

Sau khi training, bạn sẽ có:
- `eye_state_model.h5`: Mô hình cuối cùng
- `best_eye_model.h5`: Mô hình tốt nhất (theo validation accuracy)
- `training_history.png`: Biểu đồ quá trình training
- `confusion_matrix.png`: Ma trận confusion

## 🔧 Tùy chỉnh

Bạn có thể tùy chỉnh các tham số trong class `EyeStateDetector`:

```python
detector = EyeStateDetector(
    dataset_path="Eye_dataset",  # Đường dẫn dataset
    img_size=(64, 64)           # Kích thước ảnh input
)

# Tùy chỉnh training
detector.train_model(
    X_train, y_train, X_val, y_val,
    epochs=100,     # Số epochs
    batch_size=64   # Batch size
)
```

## 📈 Đánh giá mô hình

Mô hình sẽ được đánh giá bằng:
- **Accuracy**: Độ chính xác tổng thể
- **Precision**: Độ chính xác cho từng class
- **Recall**: Độ nhạy cho từng class
- **F1-score**: Trung bình điều hòa precision/recall
- **Confusion Matrix**: Ma trận nhầm lẫn

## 🐛 Xử lý lỗi

### Lỗi thiếu thư viện:
```bash
pip install -r requirements.txt
```

### Lỗi không tìm thấy dataset:
Đảm bảo thư mục `Eye_dataset/` tồn tại và có cấu trúc đúng.

### Lỗi thiếu GPU:
Mô hình sẽ tự động chạy trên CPU nếu không có GPU.

## 📞 Hỗ trợ

Nếu gặp vấn đề, hãy kiểm tra:
1. Đã cài đặt đúng Python version (≥3.7)
2. Đã cài đặt đầy đủ thư viện
3. Dataset có cấu trúc đúng format
4. Đủ dung lượng ổ cứng cho training

---
*Mô hình được thiết kế để nhận diện trạng thái mắt với độ chính xác cao trong ứng dụng phát hiện tài xế buồn ngủ.* 