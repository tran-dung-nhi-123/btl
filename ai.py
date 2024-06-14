import streamlit as st
import cv2 # type: ignore
import numpy as np
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image

# Giả sử bạn đã tải mô hình trước đó
model = tf.keras.models.load_model('C:\\Users\\DUNG NHI\\Desktop\\AI\\CNN_GRAY.h5')
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
