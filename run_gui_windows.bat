@echo off
echo ====================================
echo    EYE STATE DETECTION - WINDOWS
echo ====================================
echo.

REM Kiểm tra Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python chưa được cài đặt!
    echo Vui lòng cài đặt Python từ: https://python.org
    pause
    exit /b 1
)

echo ✅ Python đã được cài đặt
echo.

REM Kiểm tra thư viện cần thiết
echo 📦 Kiểm tra thư viện...
python -c "import torch, tkinter, PIL" >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Một số thư viện chưa được cài đặt
    echo 🔄 Đang cài đặt thư viện cần thiết...
    pip install -r requirements_pytorch.txt
    if %errorlevel% neq 0 (
        echo ❌ Lỗi cài đặt thư viện!
        pause
        exit /b 1
    )
)

echo ✅ Tất cả thư viện đã sẵn sàng
echo.

REM Kiểm tra dataset
if not exist "Eye_dataset" (
    echo ⚠️  Không tìm thấy thư mục Eye_dataset
    echo Vui lòng đảm bảo thư mục Eye_dataset tồn tại
)

echo 🚀 Khởi chạy giao diện...
echo.
python eye_detection_gui.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ Lỗi khởi chạy ứng dụng!
    pause
)

echo.
echo 👋 Cảm ơn bạn đã sử dụng!
pause 