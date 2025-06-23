#!/usr/bin/env python3
"""
Script đơn giản để train mô hình nhận diện trạng thái mắt với PyTorch
Đầu ra: file .pt
Chạy bằng lệnh: python train_pytorch.py
"""

from eye_detection_pytorch import EyeStateDetectorPyTorch, main
import sys

def quick_train():
    """Chạy training nhanh với PyTorch"""
    print("=== HỆ THỐNG NHẬN DIỆN TRẠNG THÁI MẮT - PYTORCH ===")
    print("Bắt đầu quá trình training...")
    print("Dataset: Eye_dataset/")
    print("Format: s{subject}_{image_id}_{gender}_{glasses}_{eye_state}_{reflections}_{lighting}_{sensor}.png")
    print("Eye state: 0 = nhắm mắt, 1 = mở mắt")
    print("Output: file .pt (PyTorch format)")
    print("-" * 50)
    
    try:
        # Chạy toàn bộ pipeline
        main()
        
        print("\n" + "="*50)
        print("✅ TRAINING HOÀN THÀNH THÀNH CÔNG!")
        print("📁 Mô hình đã được lưu: eye_state_model.pt")
        print("📁 Best model: best_eye_model.pt")
        print("📊 Biểu đồ training: training_history_pytorch.png")
        print("📊 Confusion matrix: confusion_matrix_pytorch.png")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ LỖI TRONG QUÁ TRÌNH TRAINING: {str(e)}")
        print("Vui lòng kiểm tra:")
        print("1. Đã cài đặt đầy đủ thư viện? (pip install -r requirements_pytorch.txt)")
        print("2. Thư mục Eye_dataset/ có tồn tại?")
        print("3. Có đủ quyền ghi file?")
        print("4. CUDA available (optional)?")
        sys.exit(1)

def test_single_image(image_path):
    """Test với một ảnh đơn lẻ"""
    print(f"Testing với ảnh: {image_path}")
    
    # Load mô hình đã train
    detector = EyeStateDetectorPyTorch()
    try:
        detector.create_model()
        detector.load_model('eye_state_model.pt')
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