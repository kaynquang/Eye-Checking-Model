#!/bin/bash

echo "===================================="
echo "   EYE STATE DETECTION - macOS"
echo "===================================="
echo

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 chưa được cài đặt!"
    echo "Vui lòng cài đặt Python3 từ: https://python.org"
    echo "Hoặc sử dụng Homebrew: brew install python"
    read -p "Nhấn Enter để thoát..."
    exit 1
fi

echo "✅ Python3 đã được cài đặt: $(python3 --version)"
echo

# Kiểm tra thư viện cần thiết
echo "📦 Kiểm tra thư viện..."
python3 -c "import torch, tkinter, PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Một số thư viện chưa được cài đặt"
    echo "🔄 Đang cài đặt thư viện cần thiết..."
    pip3 install -r requirements_pytorch.txt
    if [ $? -ne 0 ]; then
        echo "❌ Lỗi cài đặt thư viện!"
        read -p "Nhấn Enter để thoát..."
        exit 1
    fi
fi

echo "✅ Tất cả thư viện đã sẵn sàng"
echo

# Kiểm tra dataset
if [ ! -d "Eye_dataset" ]; then
    echo "⚠️  Không tìm thấy thư mục Eye_dataset"
    echo "Vui lòng đảm bảo thư mục Eye_dataset tồn tại"
fi

echo "🚀 Khởi chạy giao diện..."
echo
python3 eye_detection_gui.py

if [ $? -ne 0 ]; then
    echo
    echo "❌ Lỗi khởi chạy ứng dụng!"
    read -p "Nhấn Enter để thoát..."
fi

echo
echo "👋 Cảm ơn bạn đã sử dụng!" 