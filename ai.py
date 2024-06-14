import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

def convert_url_to_local_path(url):
    # Sử dụng thư mục tạm thời, ví dụ /tmp trên Linux
    output_dir = '/tmp'  # Bạn có thể thay đổi thành thư mục khác tuỳ thuộc vào hệ điều hành và quyền hạn
    os.makedirs(output_dir, exist_ok=True)  # Đảm bảo thư mục tồn tại
    output_path = os.path.join(output_dir, 'CNN_GRAY.h5')  # Đường dẫn cuối cùng
    gdown.download(url, output_path, quiet=False)
    return output_path

def load_model_from_drive(url):
    filepath = convert_url_to_local_path(url)
    model = tf.keras.models.load_model(filepath)
    return model

# URL của mô hình trên Google Drive
url = "https://drive.google.com/file/d/13lld1eAsVI7iKOlCghOYN52BgBPOBbn7/view?usp=sharing"

# Tải và sử dụng mô hình
model = load_model_from_drive(url)
labels = ['Bình thường', 'Viêm phổi']  # Thay thế bằng nhãn của bạn

# Đọc ảnh đầu vào từ người dùng
uploaded_file = st.file_uploader("Chọn một tệp hình ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Chuyển tệp đã tải lên thành ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Chuyển sang ảnh xám
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Đổi kích thước ảnh xám về 150x150
    img_resize = cv2.resize(img_gray, (150, 150))

    # Thêm một chiều để khớp với đầu vào của mô hình và chuẩn hóa giá trị pixel về khoảng [0, 1]
    img_resize = img_resize / 255.0
    img_resize = np.expand_dims(img_resize, axis=-1)  # Thêm chiều kênh
    img_resize = np.expand_dims(img_resize, axis=0)  # Thêm chiều batch

    # Dự đoán
    yhat = model.predict(img_resize)
    max_index = np.argmax(yhat)

    # Hiển thị kết quả
    st.write(f'- Bình thường: {round(yhat[0][0]*100, 2)}%')
    st.write(f'- Viêm phổi: {round(yhat[0][1]*100, 2)}%')
    st.write(f"--> Nhãn dự đoán: {labels[max_index]}")

    # Hiển thị hình ảnh xám
    st.image(img_gray, caption="Hình ảnh xám", use_column_width=True, clamp=True, channels="GRAY")
else:
    st.write("Vui lòng tải lên một tệp hình ảnh.")
