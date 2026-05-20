# Báo cáo Kỹ thuật: Ứng dụng Apache Spark trong Hệ thống Phân tích Dữ liệu Thời gian thực

Dưới đây là script/dàn ý chi tiết để bạn có thể sử dụng làm báo cáo kỹ thuật, thuyết trình hoặc viết vào document của dự án, tập trung vào cách triển khai Apache Spark và lý do tại sao Spark là sự lựa chọn tối ưu nhất cho bài toán này.

---

## Phần 1: Tổng quan vai trò của Spark trong Hệ thống
Trong dự án này, Apache Spark (cụ thể là **PySpark Structured Streaming**) đóng vai trò là "Trái tim" của luồng xử lý dữ liệu (Stream Processing Engine). Spark đứng giữa Kafka (nơi nhận dữ liệu thô) và hệ thống lưu trữ/cảnh báo, chịu trách nhiệm cho toàn bộ quá trình biến đổi dữ liệu, tính toán theo thời gian thực và chạy mô hình Trí tuệ Nhân tạo (Machine Learning/Deep Learning).

## Phần 2: Cách Spark được triển khai thực tế trong Code (`streaming_consumer.py`)
Spark được tận dụng tối đa các tính năng mạnh mẽ nhất của nó, cụ thể:

1. **Tiêu thụ dữ liệu Streaming & Schema Enforcement**: 
   - Spark kết nối trực tiếp với Kafka để đọc stream dữ liệu liên tục (`KAFKA_TOPIC_RESOURCE`).
   - Sử dụng định nghĩa Schema chặt chẽ (`StructType`) để parse chuỗi JSON từ Kafka thành dữ liệu có cấu trúc.

2. **Tính toán Stateful với Windowing & Watermarking**:
   - Sử dụng `window` của Spark để gom nhóm và tính toán các chỉ số thống kê (như `avg` CPU, RAM, Disk) theo các khung thời gian:
     - **Window 5 phút**: Tính toán cho từng máy chủ riêng biệt.
     - **Window 30 giây**: Tính toán tổng thể cho toàn bộ hệ thống (Global metrics).
   - Đặc biệt, sử dụng tính năng **Watermark** (`withWatermark`) của Spark để xử lý trơn tru các dữ liệu đến trễ (late arriving data) do độ trễ mạng.

3. **Rule-based Anomaly Detection & Sink ngược về Kafka**:
   - Bên cạnh AI, Spark còn đánh giá các luật logic (Ví dụ: RAM > 95% là CRITICAL, CPU > 90% là HIGH) dựa trên các window trung bình 2 phút. 
   - Kết quả cảnh báo này được Spark đẩy ngược (Write) trực tiếp vào một topic Kafka khác (`KAFKA_TOPIC_ANOMALY`) để phục vụ cảnh báo tức thời.

4. **Tích hợp Machine Learning qua kiến trúc Micro-batching (`foreachBatch`)**:
   - Đây là kỹ thuật quan trọng nhất. Spark chia dòng dữ liệu (stream) thành các mẻ nhỏ (micro-batches). 
   - Hàm `foreachBatch` cho phép chuyển đổi mỗi micro-batch sang **Pandas DataFrame** in-memory.
   - Tại đây, hệ thống gọi trực tiếp các mô hình AI đã được load sẵn (Singleton pattern) bằng Python:
     - Mô hình **Isolation Forest** (Scikit-Learn).
     - Mô hình **BiLSTM Autoencoder** (PyTorch).
   - Sau khi kết hợp (ensemble) kết quả dự đoán, Spark ghi dữ liệu ra các file dạng Parquet để Dashboard đọc.

5. **Đảm bảo tính chịu lỗi (Fault Tolerance) & Checkpointing**:
   - Ứng dụng sử dụng cơ chế lưu trữ trạng thái `checkpointLocation`, giúp Spark ghi lại chính xác offset của Kafka đã đọc đến đâu. Nếu Spark container bị sập và khởi động lại, nó sẽ tiếp tục xử lý chính xác từ điểm bị dừng, không gây lặp hoặc mất dữ liệu (Exactly-once / At-least-once semantics).
   - Có cơ chế vòng lặp auto-restart để tự động hồi phục khi mất kết nối mạng với Kafka.

---

## Phần 3: Tại sao lại chọn Apache Spark? (So sánh với các giải pháp khác)

Trong phần này, chúng ta sẽ bảo vệ quyết định kiến trúc: Tại sao dùng Spark thay vì Flink, Kafka Streams hay tự viết bằng Python?

### 1. Tại sao dùng Spark thay vì Apache Flink?
- **Flink** nổi tiếng với khả năng xử lý luồng thực thụ (Native Streaming / Event-by-event) với độ trễ cực thấp (tính bằng milli-giây).
- **Tuy nhiên**, đối với bài toán Phân tích tài nguyên (Resource monitoring), độ trễ **Micro-batching** của Spark (khoảng 30 giây/batch) là hoàn toàn chấp nhận được và đạt yêu cầu thực tế.
- **Lý do cốt lõi:** Việc tích hợp Flink với hệ sinh thái AI của Python (như PyTorch, Scikit-learn) vô cùng phức tạp, thường đòi hỏi phải thiết lập các REST API server bên ngoài để Flink gửi request sang, gây thắt cổ chai về mạng. Trong khi đó, **Spark + `foreachBatch`** cho phép chuyển đổi mượt mà luồng dữ liệu sang Pandas và gọi hàm PyTorch/Sklearn xử lý tại chỗ (in-memory) trong cùng một Executor của Python.

### 2. Tại sao dùng Spark thay vì Kafka Streams?
- **Kafka Streams** là thư viện rất nhẹ, không cần cài đặt cluster phức tạp, chạy trực tiếp trên Java/Scala.
- **Lý do cốt lõi:** Kafka Streams là hệ sinh thái của Java. Dự án của chúng ta đang sử dụng các mô hình AI tiên tiến (BiLSTM, Isolation Forest) được viết bằng Python. Việc gượng ép Kafka Streams gọi các thư viện AI của Python là điều không tự nhiên, tốn kém chi phí cấu hình (JNI hoặc API wrapper) và rất khó bảo trì. PySpark giải quyết triệt để vấn đề "ngôn ngữ" này.

### 3. Tại sao không tự viết một Python Kafka Consumer Script thông thường?
- Nếu chỉ tự viết vòng lặp `while True: kafka_consumer.poll()`, dự án sẽ gặp các rào cản kỹ thuật khổng lồ:
  - **Quản lý Window & Trạng thái (State):** Rất khó để code tay tính năng nhóm dữ liệu theo thời gian 5 phút, 30 giây, nhất là khi dữ liệu đến lộn xộn hoặc đến trễ. Spark giải quyết việc này bằng 1 dòng lệnh `groupBy(window(...))`.
  - **Khả năng mở rộng (Scalability):** Đoạn script tự viết thường chỉ chạy trên 1 process/1 máy. Spark hỗ trợ tính toán phân trực tiếp (Distributed computing), dễ dàng mở rộng thêm Worker Node khi khối lượng log từ Alibaba cluster phình to.
  - **Khả năng chịu lỗi (Fault Tolerance):** Quản lý offset thủ công và xử lý crash rất rủi ro. Cơ chế Checkpoint của Spark an toàn và chuẩn công nghiệp hơn rất nhiều.

### Kết luận
Apache Spark (PySpark) là lựa chọn hoàn hảo và cân bằng nhất cho hệ thống này vì nó cung cấp:
1. Sức mạnh xử lý dữ liệu lớn, windowing theo thời gian thực.
2. Cơ chế chịu lỗi mạnh mẽ.
3. **Quan trọng nhất:** Sự tương thích tuyệt đối với hệ sinh thái Machine Learning của Python (đáp ứng trọn vẹn nghiệp vụ chạy model PyTorch và Sklearn trực tiếp trên pipeline dữ liệu bằng hàm foreachBatch).
