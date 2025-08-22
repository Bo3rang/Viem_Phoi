Chẩn đoán Viêm phổi bằng ảnh X-quang (Pneumonia Diagnosis from X-ray Images)
Dự án này sử dụng mô hình học sâu (Deep Learning) để phân loại ảnh X-quang ngực, giúp phát hiện bệnh viêm phổi. Một ứng dụng web sử dụng Streamlit được xây dựng để demo khả năng của mô hình.
Mục tiêu
Mục tiêu chính của dự án là xây dựng một mô hình có khả năng phân biệt giữa hai loại ảnh X-quang:
NORMAL: Phổi bình thường, không có dấu hiệu bệnh.
PNEUMONIA: Phổi có dấu hiệu bị viêm.
Tập dữ liệu (Dataset)
Mô hình được huấn luyện trên bộ dữ liệu công khai về ảnh X-quang ngực. Bạn có thể tìm và tải bộ dữ liệu này trên nền tảng Kaggle.
Link tham khảo: Chest X-Ray Images (Pneumonia) (Bạn hãy thay link này bằng link bộ dữ liệu chính xác bạn đã sử dụng nhé).
Mô hình và Hiệu suất
Kiến trúc
Dự án sử dụng kiến trúc mạng DenseNet121 được huấn luyện trước trên tập dữ liệu ImageNet, giúp tận dụng các đặc trưng đã học được từ một bộ dữ liệu lớn và đa dạng.
Hiệu suất
Mô hình đã được đánh giá trên tập kiểm tra (test set) và đạt được các kết quả sau:
Độ chính xác (Accuracy): 87.34%
Báo cáo phân loại (Classification Report)
Lớp	Precision	Recall	F1-Score
NORMAL	0.79	0.91	0.84
PNEUMONIA	0.94	0.86	0.90
Tổng thể	-	-	0.87 (Macro Avg)
-	-	0.88 (Weighted Avg)
Ma trận nhầm lẫn (Confusion Matrix)
Dự đoán: NORMAL	Dự đoán: PNEUMONIA
Thực tế: NORMAL	212 (Đúng)	22 (Sai)
Thực tế: PNEUMONIA	56 (Sai)	334 (Đúng)
Hướng dẫn cài đặt và sử dụng
1. Chuẩn bị môi trường
Clone kho lưu trữ này về máy của bạn:
code
Bash
git clone https://github.com/Bo3rang/Viem_Phoi.git
cd Viem_Phoi
2. Cài đặt các thư viện cần thiết
Dự án có một file requirements.txt trong thư mục StreamLit để quản lý các thư viện phụ thuộc. Chạy lệnh sau để cài đặt:
code
Bash
pip install -r StreamLit/requirements.txt
3. Chạy ứng dụng web
Sau khi cài đặt thành công, chạy ứng dụng Streamlit bằng lệnh:
code
Bash
streamlit run StreamLit/app.py
Mở trình duyệt và truy cập vào địa chỉ http://localhost:8501 để xem giao diện và thử nghiệm mô hình.
Tình trạng dự án
Đã hoàn thành.
