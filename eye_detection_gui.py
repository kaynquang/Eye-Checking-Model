import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import PhotoImage
import threading
import os
import sys
import platform
from PIL import Image, ImageTk
import cv2
import numpy as np

# Import mô hình
try:
    from eye_detection_pytorch import EyeStateDetectorPyTorch
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch không có sẵn. Vui lòng cài đặt: pip install torch torchvision")

class EyeDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.detector = None
        self.training_thread = None
        self.is_training = False
        
        # Thiết lập giao diện
        self.create_widgets()
        
    def setup_window(self):
        """Thiết lập cửa sổ chính"""
        self.root.title("👁️ Eye State Detection - Nhận diện trạng thái mắt")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Phát hiện hệ điều hành
        self.os_type = platform.system()
        
        if self.os_type == "Darwin":  # macOS
            self.root.configure(bg='#f0f0f0')
            # Thiết lập font cho macOS
            self.title_font = ("SF Pro Display", 18, "bold")
            self.button_font = ("SF Pro Display", 12)
            self.text_font = ("SF Pro Display", 11)
        elif self.os_type == "Windows":  # Windows
            self.root.configure(bg='#f0f0f0')
            # Thiết lập font cho Windows
            self.title_font = ("Segoe UI", 16, "bold")
            self.button_font = ("Segoe UI", 10)
            self.text_font = ("Segoe UI", 9)
        else:  # Linux
            self.title_font = ("Arial", 16, "bold")
            self.button_font = ("Arial", 10)
            self.text_font = ("Arial", 9)
    
    def create_widgets(self):
        """Tạo các widget"""
        # Frame chính
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Cấu hình grid weight
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Tiêu đề
        title_label = ttk.Label(main_frame, text="👁️ EYE STATE DETECTION", 
                               font=self.title_font)
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Thông tin hệ điều hành
        os_info = f"🖥️ Hệ điều hành: {self.os_type}"
        os_label = ttk.Label(main_frame, text=os_info, font=self.text_font)
        os_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        # Phần 1: Training
        training_frame = ttk.LabelFrame(main_frame, text="🚀 TRAINING MODEL", padding="10")
        training_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        training_frame.columnconfigure(1, weight=1)
        
        # Nút training
        self.train_button = ttk.Button(training_frame, text="🎯 BẮT ĐẦU TRAINING", 
                                      command=self.start_training,
                                      style="Accent.TButton")
        self.train_button.grid(row=0, column=0, padx=(0, 10), pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(training_frame, mode='indeterminate')
        self.progress.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        # Status training
        self.train_status = ttk.Label(training_frame, text="Sẵn sàng để training", 
                                     font=self.text_font)
        self.train_status.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Phần 2: Prediction
        predict_frame = ttk.LabelFrame(main_frame, text="🔍 NHẬN DIỆN ẢNH", padding="10")
        predict_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        predict_frame.columnconfigure(1, weight=1)
        
        # Nút chọn ảnh
        self.select_button = ttk.Button(predict_frame, text="📁 CHỌN ẢNH", 
                                       command=self.select_image)
        self.select_button.grid(row=0, column=0, padx=(0, 10), pady=5)
        
        # Đường dẫn file
        self.file_path = tk.StringVar()
        self.file_entry = ttk.Entry(predict_frame, textvariable=self.file_path, 
                                   state="readonly", font=self.text_font)
        self.file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        # Nút predict
        self.predict_button = ttk.Button(predict_frame, text="🔮 NHẬN DIỆN", 
                                        command=self.predict_image,
                                        state="disabled")
        self.predict_button.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Phần 3: Hiển thị kết quả
        result_frame = ttk.LabelFrame(main_frame, text="📊 KẾT QUẢ", padding="10")
        result_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Kết quả text
        self.result_text = tk.Text(result_frame, height=8, wrap=tk.WORD, 
                                  font=self.text_font, state="disabled")
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Scrollbar cho text
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S), pady=(0, 10))
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        # Frame cho ảnh
        self.image_frame = ttk.Frame(result_frame)
        self.image_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Label hiển thị ảnh
        self.image_label = ttk.Label(self.image_frame, text="Chưa có ảnh được chọn")
        self.image_label.pack(expand=True, fill="both")
        
        # Phần 4: Utilities
        util_frame = ttk.LabelFrame(main_frame, text="🛠️ TIỆN ÍCH", padding="10")
        util_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Nút mở thư mục
        self.open_folder_button = ttk.Button(util_frame, text="📂 MỞ THƯ MỤC DATASET", 
                                           command=self.open_dataset_folder)
        self.open_folder_button.grid(row=0, column=0, padx=(0, 10), pady=5)
        
        # Nút reset
        self.reset_button = ttk.Button(util_frame, text="🔄 RESET", 
                                      command=self.reset_all)
        self.reset_button.grid(row=0, column=1, padx=10, pady=5)
        
        # Nút exit
        self.exit_button = ttk.Button(util_frame, text="❌ THOÁT", 
                                     command=self.root.quit)
        self.exit_button.grid(row=0, column=2, padx=(10, 0), pady=5)
        
        # Khởi tạo detector nếu có PyTorch
        if PYTORCH_AVAILABLE:
            self.update_result("✅ PyTorch đã sẵn sàng!\n")
            self.init_detector()
        else:
            self.update_result("❌ PyTorch chưa được cài đặt!\n")
            self.update_result("Vui lòng chạy: pip install torch torchvision\n")
    
    def init_detector(self):
        """Khởi tạo detector"""
        try:
            self.detector = EyeStateDetectorPyTorch()
            self.update_result("🤖 Detector đã được khởi tạo!\n")
            
            # Kiểm tra xem có model đã train chưa
            if os.path.exists('eye_state_model.pt'):
                self.detector.create_model()
                self.detector.load_model('eye_state_model.pt')
                self.update_result("📦 Đã load model có sẵn: eye_state_model.pt\n")
                self.predict_button.configure(state="normal")
            else:
                self.update_result("⚠️  Chưa có model được train. Hãy training trước!\n")
                
        except Exception as e:
            self.update_result(f"❌ Lỗi khởi tạo detector: {str(e)}\n")
    
    def start_training(self):
        """Bắt đầu training"""
        if self.is_training:
            self.update_result("⚠️  Training đang chạy!\n")
            return
            
        if not PYTORCH_AVAILABLE:
            messagebox.showerror("Lỗi", "PyTorch chưa được cài đặt!")
            return
            
        # Kiểm tra dataset
        if not os.path.exists("Eye_dataset"):
            messagebox.showerror("Lỗi", "Không tìm thấy thư mục Eye_dataset!")
            return
        
        # Xác nhận training
        result = messagebox.askyesno("Xác nhận", 
                                   "Bắt đầu training? Quá trình này có thể mất vài phút.")
        if not result:
            return
        
        # Disable nút training
        self.train_button.configure(state="disabled", text="🔄 ĐANG TRAINING...")
        self.progress.start()
        self.is_training = True
        
        # Chạy training trong thread riêng
        self.training_thread = threading.Thread(target=self.run_training)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def run_training(self):
        """Chạy training trong background"""
        try:
            self.update_result("🚀 Bắt đầu training...\n")
            self.update_train_status("Đang load dataset...")
            
            # Khởi tạo detector mới
            detector = EyeStateDetectorPyTorch()
            
            # Load dataset
            image_paths, labels = detector.load_dataset()
            self.update_result(f"📊 Tổng số ảnh: {len(image_paths)}\n")
            
            self.update_train_status("Đang tiền xử lý ảnh...")
            # Preprocess
            X, y = detector.preprocess_images(image_paths, labels)
            
            self.update_train_status("Đang chia dataset...")
            # Split dataset
            from sklearn.model_selection import train_test_split
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
            
            self.update_result(f"📈 Training set: {X_train.shape[0]} ảnh\n")
            self.update_result(f"📊 Validation set: {X_val.shape[0]} ảnh\n")
            self.update_result(f"🧪 Test set: {X_test.shape[0]} ảnh\n")
            
            self.update_train_status("Đang tạo mô hình...")
            # Tạo mô hình
            model = detector.create_model()
            
            self.update_train_status("Đang training mô hình...")
            # Train với ít epochs hơn cho demo
            detector.train_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=32)
            
            self.update_train_status("Đang đánh giá mô hình...")
            # Load best model và đánh giá
            detector.load_model('best_eye_model.pt')
            y_pred, y_true = detector.evaluate_model(X_test, y_test)
            
            # Lưu mô hình cuối cùng
            detector.save_model()
            
            self.update_result("✅ Training hoàn thành!\n")
            self.update_result("📁 Đã lưu: eye_state_model.pt và best_eye_model.pt\n")
            
            # Cập nhật detector cho prediction
            self.detector = detector
            self.predict_button.configure(state="normal")
            
        except Exception as e:
            self.update_result(f"❌ Lỗi training: {str(e)}\n")
        finally:
            # Reset UI
            self.root.after(0, self.training_finished)
    
    def training_finished(self):
        """Gọi khi training xong"""
        self.progress.stop()
        self.train_button.configure(state="normal", text="🎯 BẮT ĐẦU TRAINING")
        self.update_train_status("Training hoàn thành!")
        self.is_training = False
    
    def select_image(self):
        """Chọn ảnh để predict"""
        file_types = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
            ("PNG files", "*.png"),
            ("JPG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Chọn ảnh để nhận diện",
            filetypes=file_types,
            initialdir="Eye_dataset" if os.path.exists("Eye_dataset") else "."
        )
        
        if filename:
            self.file_path.set(filename)
            self.display_image(filename)
            if self.detector and hasattr(self.detector, 'model') and self.detector.model:
                self.predict_button.configure(state="normal")
    
    def display_image(self, image_path):
        """Hiển thị ảnh đã chọn"""
        try:
            # Load và resize ảnh
            image = Image.open(image_path)
            
            # Tính toán kích thước hiển thị (max 300x300)
            display_size = (300, 300)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert sang PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Hiển thị
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Giữ reference
            
        except Exception as e:
            self.update_result(f"❌ Lỗi hiển thị ảnh: {str(e)}\n")
    
    def predict_image(self):
        """Nhận diện ảnh"""
        if not self.detector or not hasattr(self.detector, 'model') or not self.detector.model:
            messagebox.showerror("Lỗi", "Chưa có model! Hãy training trước!")
            return
            
        image_path = self.file_path.get()
        if not image_path:
            messagebox.showerror("Lỗi", "Chưa chọn ảnh!")
            return
        
        try:
            # Predict
            result, confidence = self.detector.predict_single_image(image_path)
            
            # Hiển thị kết quả
            filename = os.path.basename(image_path)
            self.update_result(f"\n🔍 NHẬN DIỆN: {filename}\n")
            self.update_result(f"📊 Kết quả: {result}\n")
            self.update_result(f"🎯 Độ tin cậy: {confidence:.1f}%\n")
            
            # Thêm emoji theo kết quả
            if "mở" in result.lower():
                self.update_result("😊 Mắt đang mở!\n")
            else:
                self.update_result("😴 Mắt đang nhắm!\n")
                
            self.update_result("-" * 40 + "\n")
            
        except Exception as e:
            self.update_result(f"❌ Lỗi predict: {str(e)}\n")
    
    def open_dataset_folder(self):
        """Mở thư mục dataset"""
        if self.os_type == "Darwin":  # macOS
            os.system("open Eye_dataset" if os.path.exists("Eye_dataset") else "open .")
        elif self.os_type == "Windows":  # Windows
            os.system("explorer Eye_dataset" if os.path.exists("Eye_dataset") else "explorer .")
        else:  # Linux
            os.system("xdg-open Eye_dataset" if os.path.exists("Eye_dataset") else "xdg-open .")
    
    def reset_all(self):
        """Reset tất cả"""
        self.file_path.set("")
        self.image_label.configure(image="", text="Chưa có ảnh được chọn")
        self.image_label.image = None
        self.predict_button.configure(state="disabled")
        
        # Clear result text
        self.result_text.configure(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.configure(state="disabled")
        
        self.update_result("🔄 Đã reset giao diện!\n")
    
    def update_result(self, text):
        """Cập nhật text kết quả"""
        self.result_text.configure(state="normal")
        self.result_text.insert(tk.END, text)
        self.result_text.see(tk.END)
        self.result_text.configure(state="disabled")
        self.root.update_idletasks()
    
    def update_train_status(self, text):
        """Cập nhật status training"""
        self.train_status.configure(text=text)
        self.root.update_idletasks()

def main():
    """Hàm main"""
    root = tk.Tk()
    app = EyeDetectionGUI(root)
    
    # Thiết lập style cho từng OS
    style = ttk.Style()
    
    if platform.system() == "Darwin":  # macOS
        try:
            # Sử dụng native macOS theme nếu có
            style.theme_use("aqua")
        except:
            style.theme_use("clam")
    elif platform.system() == "Windows":  # Windows
        try:
            # Sử dụng Windows 10/11 theme nếu có
            style.theme_use("vista")
        except:
            style.theme_use("clam")
    
    # Custom button style
    style.configure("Accent.TButton", font=app.button_font)
    
    root.mainloop()

if __name__ == "__main__":
    main() 