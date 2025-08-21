import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- CÀI ĐẶT CHUNG CHO TRANG ---
st.set_page_config(
    page_title="Trợ lý Chẩn đoán Viêm phổi",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH CHO GIAO DIỆN TỐI ---
st.markdown("""
<style>
/* Thay đổi font chữ toàn bộ ứng dụng */
html, body, [class*="css"] {
    font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
}

/* Tùy chỉnh sidebar để có màu nền tối nhưng khác biệt */
[data-testid="stSidebar"] {
    background-color: #1a1a2e;
}

/* Tiêu đề trong sidebar */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #e6e6e6;
}

/* Cảnh báo trong sidebar */
[data-testid="stAlert"] {
    border-radius: 8px;
}

/* Tùy chỉnh nút bấm chính */
div.stButton > button:first-child {
    background-color: #0078d4;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 4px 14px 0 rgba(0, 118, 212, 0.39);
}
div.stButton > button:first-child:hover {
    background-color: #005a9e;
    box-shadow: 0 6px 20px 0 rgba(0, 118, 212, 0.23);
}

/* Tùy chỉnh khu vực tải file cho theme tối */
[data-testid="stFileUploader"] {
    border: 2px dashed #0078d4;
    background-color: #1e1e3f;
    border-radius: 10px;
    padding: 20px;
}
[data-testid="stFileUploader"] label {
    font-weight: bold;
    color: #0078d4;
}
</style>
""", unsafe_allow_html=True)


# --- CACHING MODEL ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('H:\My Drive\Đồ án chuyên ngành Trí tuệ nhân tạo\saved_models\pneumonia_densenet121_model.h5')
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        st.info("Hãy đảm bảo tệp mô hình (ví dụ: 'your_model.h5') nằm cùng thư mục với tệp 'app.py'")
        st.stop()

model = load_model()

# --- HÀM XỬ LÝ VÀ DỰ ĐOÁN ---
def predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = image.convert('RGB')
    img_array = np.asarray(image)
    img_array_normalized = img_array / 255.0
    data = np.expand_dims(img_array_normalized, axis=0)
    prediction = model.predict(data)
    return prediction

# --- GIAO DIỆN ---

# --- SIDEBAR ---
with st.sidebar:
    st.title("👨‍⚕️ Về dự án")
    st.markdown("""
    **Trợ lý Chẩn đoán Viêm phổi** là một công cụ ứng dụng trí tuệ nhân tạo, được xây dựng để hỗ trợ các chuyên gia y tế trong việc phát hiện sớm các dấu hiệu viêm phổi từ ảnh chụp X-quang ngực.
    """)
    
    st.header("Hướng dẫn sử dụng")
    st.markdown("""
    1.  Nhấn vào nút **'Tải lên ảnh X-quang'** ở màn hình chính.
    2.  Chọn một tệp ảnh từ máy tính của bạn (.jpg, .png, .jpeg).
    3.  Nhấn nút **'Chẩn đoán'** và chờ kết quả.
    """)
    
    st.warning("**Lưu ý quan trọng:** Mô hình chỉ là công cụ hỗ trợ, không thay thế cho chẩn đoán chuyên nghiệp từ bác sĩ. Luôn tham khảo ý kiến của chuyên gia y tế.", icon="⚠️")
    
# --- TRANG CHÍNH ---
st.title("🩺 Trợ lý Chẩn đoán Viêm phổi")
st.markdown("### Tải lên hình ảnh X-quang ngực để mô hình phân tích và dự đoán.")

uploaded_file = st.file_uploader(
    "Kéo và thả hoặc nhấn để chọn ảnh", 
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.info("Hãy bắt đầu bằng cách tải lên một hình ảnh ở khung trên.", icon="⬆️")
else:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.header("Ảnh đầu vào")
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh X-quang đã tải lên.", use_container_width=True)

    with col2:
        st.header("Kết quả phân tích")
        st.write("Nhấn nút bên dưới để bắt đầu chẩn đoán.")
        
        if st.button('Chẩn đoán', use_container_width=True):
            with st.spinner('Mô hình đang làm việc, vui lòng chờ...'):
                prediction = predict(image, model)
                confidence_score = prediction[0][0]
                st.markdown("---")
                
                if confidence_score > 0.5:
                    percentage = confidence_score * 100
                    st.error(f"**Kết quả: PHÁT HIỆN DẤU HIỆU VIÊM PHỔI**", icon="🚨")
                    st.metric(label="Mức độ tin cậy của mô hình", value=f"{percentage:.2f}%")
                    # SỬA LỖI Ở ĐÂY
                    st.progress(float(confidence_score))
                    st.caption("Mức độ tin cậy thể hiện xác suất ảnh thuộc lớp 'VIÊM PHỔI'.")
                else:
                    percentage = (1 - confidence_score) * 100
                    st.success(f"**Kết quả: KHÔNG PHÁT HIỆN DẤU HIỆU VIÊM PHỔI**", icon="✅")
                    st.metric(label="Mức độ tin cậy của mô hình", value=f"{percentage:.2f}%")
                    # SỬA LỖI Ở ĐÂY
                    st.progress(float(1 - confidence_score))
                    st.caption("Mức độ tin cậy thể hiện xác suất ảnh thuộc lớp 'BÌNH THƯỜNG'.")