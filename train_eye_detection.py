#!/usr/bin/env python3
"""
Script đơn giản để train mô hình nhận diện trạng thái mắt (nhắm/mở)
Chạy bằng lệnh: python train_eye_detection.py
"""

from eye_detection import EyeStateDetector, main
import sys

def quick_train():
    """Chạy training nhanh với các tham số mặc định"""
    print("=== HỆ THỐNG NHẬN DIỆN TRẠNG THÁI MẮT ===")
    print("Bắt đầu quá trình training...")
    print("Dataset: Eye_dataset/")
    print("Format: s{subject}_{image_id}_{gender}_{glasses}_{eye_state}_{reflections}_{lighting}_{sensor}.png")
    print("Eye state: 0 = nhắm mắt, 1 = mở mắt")
    print("-" * 50)
    
    try:
        # Chạy toàn bộ pipeline
        main()
        
        print("\n" + "="*50)
        print("✅ TRAINING HOÀN THÀNH THÀNH CÔNG!")
        print("📁 Mô hình đã được lưu: eye_state_model.h5")
        print("📁 Best model: best_eye_model.h5")
        print("📊 Biểu đồ training: training_history.png")
        print("📊 Confusion matrix: confusion_matrix.png")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ LỖI TRONG QUÁ TRÌNH TRAINING: {str(e)}")
        print("Vui lòng kiểm tra:")
        print("1. Đã cài đặt đầy đủ thư viện? (pip install -r requirements.txt)")
        print("2. Thư mục Eye_dataset/ có tồn tại?")
        print("3. Có đủ quyền ghi file?")
        sys.exit(1)

def test_single_image(image_path):
    """Test với một ảnh đơn lẻ"""
    print(f"Testing với ảnh: {image_path}")
    
    # Load mô hình đã train
    detector = EyeStateDetector()
    try:
        detector.load_model('eye_state_model.h5')
        result, confidence = detector.predict_single_image(image_path)
        print(f"Kết quả: {result} (Confidence: {confidence:.1f}%)")
    except:
        print("❌ Chưa có mô hình được train. Hãy chạy training trước!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Nếu có tham số, coi như đường dẫn ảnh để test
        test_single_image(sys.argv[1])
    else:
        # Chạy training
        quick_train() 