from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import UNet 

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    original_img = None  # Đường dẫn tới ảnh gốc
    predicted_img = None  # Đường dẫn tới ảnh dự đoán

    if request.method == 'POST':
        file = request.files['image']  # Nhận tệp tải lên
        if file and file.filename.endswith(('jpg', 'jpeg', 'png')):  # Kiểm tra định dạng
            # Lưu hình ảnh gốc
            image_path = os.path.join("static/", file.filename)
            file.save(image_path)

            # Đọc kích thước của ảnh gốc
            original_image = cv2.imread(image_path)
            original_size = original_image.shape[:2]  # Lưu kích thước gốc

            # Tiền xử lý để dự đoán
            resized_image = cv2.resize(original_image, (128, 128))  # Đổi kích thước cho mô hình
            normalized_image = resized_image / 255.0  # Chuẩn hóa
            image_batch = np.expand_dims(normalized_image, axis=0)  # Tạo batch

            # Tải mô hình và trọng số
            model = UNet(128)  # Mô hình với đầu vào 128x128
            model.load_weights("data/UNetW.weights.h5")  # Tải trọng số mô hình

            # Dự đoán
            prediction = model.predict(image_batch)
            prediction = (prediction[0] > 0.5).astype(np.uint8)  # Ngưỡng phân đoạn

            # Thay đổi kích thước ảnh dự đoán để khớp với kích thước gốc
            predicted_resized = cv2.resize(prediction, (original_size[1], original_size[0]))  # Khớp với kích thước gốc

            # Tạo và lưu ảnh dự đoán
            predicted_path = "static/predicted.png"
            cv2.imwrite(predicted_path, predicted_resized * 255)  # Lưu ảnh dự đoán (đổi về 255)

            original_img = image_path  # Đường dẫn tới ảnh gốc
            predicted_img = predicted_path  # Đường dẫn tới ảnh dự đoán

    # Trả về trang chính với đường dẫn tới ảnh gốc và ảnh dự đoán
    return render_template('index.html', original_img=original_img, predicted_img=predicted_img)

# Khởi động ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)
