# 👁️ Giao diện người dùng - Eye State Detection

Ứng dụng GUI đơn giản để nhận diện trạng thái mắt (mở/nhắm) với giao diện thân thiện người dùng.

## 🖥️ Tính năng

- ✅ **Training tự động** - Chỉ cần nhấn nút
- ✅ **Nhận diện ảnh** - Kéo thả hoặc chọn file
- ✅ **Hiển thị kết quả** - Trực quan và dễ hiểu  
- ✅ **Tương thích đa nền tảng** - Windows & macOS
- ✅ **Giao diện native** - Phù hợp với từng hệ điều hành
- ✅ **Tự động cài đặt** - Thư viện được cài tự động

## 🚀 Cách sử dụng

### 🪟 Windows

**Cách 1: Double-click (Khuyến nghị)**
```
👆 Double-click vào file: run_gui_windows.bat
```

**Cách 2: Command Prompt**
```cmd
run_gui_windows.bat
```

**Cách 3: PowerShell**
```powershell
.\run_gui_windows.bat
```

### 🍎 macOS

**Cách 1: Terminal (Khuyến nghị)**
```bash
./run_gui_macos.sh
```

**Cách 2: Finder**
```
👆 Right-click → Open with Terminal
```

**Cách 3: Python trực tiếp**
```bash
python3 eye_detection_gui.py
```

## 📋 Yêu cầu hệ thống

### 🔧 Phần mềm cần thiết

- **Python 3.7+** (Windows) / **Python 3.8+** (macOS)
- **pip** (Python package manager)

### 📦 Thư viện (tự động cài đặt)

- `torch` - PyTorch framework
- `torchvision` - Computer vision tools
- `opencv-python` - Image processing
- `PIL/Pillow` - Image handling
- `tkinter` - GUI framework (có sẵn trong Python)
- `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

## 🎯 Hướng dẫn từng bước

### Bước 1: Chuẩn bị Dataset
```
📁 Đảm bảo thư mục Eye_dataset/ tồn tại
├── s0001/
│   ├── s0001_xxxxx_x_x_0_x_x_xx.png  (mắt nhắm)
│   ├── s0001_xxxxx_x_x_1_x_x_xx.png  (mắt mở)
│   └── ...
├── s0002/
└── ...
```

### Bước 2: Khởi chạy ứng dụng
- **Windows**: Double-click `run_gui_windows.bat`
- **macOS**: Chạy `./run_gui_macos.sh` trong Terminal

### Bước 3: Training mô hình
1. 🎯 Nhấn nút **"BẮT ĐẦU TRAINING"**
2. ⏳ Chờ quá trình training hoàn thành (5-10 phút)
3. ✅ Mô hình sẽ được lưu tự động

### Bước 4: Nhận diện ảnh
1. 📁 Nhấn **"CHỌN ẢNH"** và chọn file ảnh
2. 🔮 Nhấn **"NHẬN DIỆN"** để phân tích
3. 📊 Xem kết quả trong phần **"KẾT QUẢ"**

## 🛠️ Tính năng giao diện

### 🚀 Training Section
- **Training Button**: Bắt đầu quá trình training
- **Progress Bar**: Hiển thị tiến trình
- **Status**: Trạng thái training hiện tại

### 🔍 Prediction Section  
- **File Browser**: Chọn ảnh để nhận diện
- **Image Preview**: Xem trước ảnh đã chọn
- **Predict Button**: Thực hiện nhận diện

### 📊 Results Section
- **Text Output**: Kết quả chi tiết
- **Image Display**: Hiển thị ảnh được phân tích
- **Confidence Score**: Độ tin cậy của kết quả

### 🛠️ Utilities Section
- **Open Dataset**: Mở thư mục dataset
- **Reset**: Làm mới giao diện
- **Exit**: Thoát ứng dụng

## 🎨 Giao diện theo từng OS

### 🪟 Windows
- **Font**: Segoe UI (Windows native)
- **Theme**: Vista/Windows 10-11 style
- **Colors**: Windows system colors

### 🍎 macOS
- **Font**: SF Pro Display (macOS native)  
- **Theme**: Aqua (macOS native)
- **Colors**: macOS system colors

## 🔧 Khắc phục sự cố

### ❌ Python không được cài đặt
```bash
# Windows
Download từ: https://python.org

# macOS  
brew install python
# hoặc download từ: https://python.org
```

### ❌ Lỗi thư viện
```bash
# Cài đặt thủ công
pip install -r requirements_pytorch.txt

# hoặc macOS
pip3 install -r requirements_pytorch.txt
```

### ❌ Không tìm thấy dataset
```
Đảm bảo thư mục Eye_dataset/ tồn tại trong cùng thư mục với GUI
```

### ❌ Lỗi hiển thị GUI
```bash
# macOS: Cài đặt tkinter
brew install python-tk

# Windows: Tkinter có sẵn trong Python
```

## 📁 Cấu trúc Files

```
CV-Helmet/
├── 📄 eye_detection_gui.py          # GUI chính
├── 🪟 run_gui_windows.bat           # Script Windows  
├── 🍎 run_gui_macos.sh             # Script macOS
├── 🤖 eye_detection_pytorch.py      # Model PyTorch
├── 📋 requirements_pytorch.txt      # Dependencies
├── 📖 README_GUI.md                # Hướng dẫn này
└── 📁 Eye_dataset/                 # Dataset
    ├── s0001/
    ├── s0002/
    └── ...
```

## 🎯 Output Files

Sau khi training, các file sau sẽ được tạo:

- `eye_state_model.pt` - Mô hình cuối cùng
- `best_eye_model.pt` - Mô hình tốt nhất  
- `training_history_pytorch.png` - Biểu đồ training
- `confusion_matrix_pytorch.png` - Ma trận confusion

## 💡 Tips & Tricks

1. **🚀 Performance**: Sử dụng GPU nếu có (CUDA)
2. **📊 Dataset**: Nhiều ảnh hơn = kết quả tốt hơn
3. **⚡ Speed**: Giảm epochs trong GUI để training nhanh hơn
4. **🔄 Re-training**: Có thể training lại bất cứ lúc nào
5. **📱 Batch**: Có thể nhận diện nhiều ảnh liên tục

## 🆘 Hỗ trợ

Nếu gặp vấn đề:

1. Kiểm tra Python version: `python --version`
2. Kiểm tra pip: `pip --version`  
3. Cài đặt thủ công: `pip install -r requirements_pytorch.txt`
4. Chạy GUI trực tiếp: `python eye_detection_gui.py`

---

**🎉 Chúc bạn sử dụng vui vẻ!** 

Made with ❤️ for Windows & macOS 