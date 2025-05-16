# 📚 Personal-Project-NLP: Chatbot Tích Hợp Hệ Thống Gợi Ý và Dự Đoán

Bài tập cá nhân môn học **Xử lý Ngôn ngữ Tự nhiên (NLP)**, dự án này là một ứng dụng web tích hợp **Chatbot Q&A**, **Hệ thống gợi ý sách (Recommendation System)**, và **Mô hình dự đoán nhãn (Predict Model)**. Ứng dụng được xây dựng để thể hiện các kỹ thuật NLP, Machine Learning, và giao diện người dùng thân thiện, hướng tới việc cung cấp trải nghiệm tương tác thông minh và cá nhân hóa.

---

## 🎯 Giới Thiệu Dự Án

Dự án **Personal-Project-NLP** phát triển một hệ thống web với ba chức năng chính:
1. **Chatbot Q&A**: Trả lời câu hỏi người dùng dựa trên dữ liệu được huấn luyện trước, sử dụng mô hình SentenceTransformer để tìm kiếm câu trả lời phù hợp.
2. **Hệ Thống Gợi Ý Sách**: Đề xuất sách dựa trên sở thích của người dùng, sử dụng kỹ thuật phân tích ma trận và học máy.
3. **Mô Hình Dự Đoán Nhãn**: Xây dựng mô hình phân loại văn bản với khả năng tăng cường dữ liệu, tiền xử lý, và huấn luyện các mô hình học máy hoặc học sâu.

Ứng dụng được xây dựng bằng **Python** với framework **Flask** cho backend, giao diện người dùng sử dụng **HTML/CSS/JavaScript**, và tích hợp các thư viện NLP/ML như **SentenceTransformers**, **Transformers**, **Scikit-learn**, **NLTK**, và **Spacy**.

---

## 🚀 Các Kiến Thức và Kỹ Thuật Áp Dụng

Dự án áp dụng nhiều kỹ thuật tiên tiến trong **NLP**, **Machine Learning**, và phát triển **Chatbot**. Dưới đây là các kiến thức chính được sử dụng:

### 1. 🗣️ Xử Lý Ngôn Ngữ Tự Nhiên (NLP)
- **Nhúng Câu (Sentence Embeddings)**: Sử dụng mô hình **SentenceTransformer** (`all-MiniLM-L6-v2`) để biểu diễn câu hỏi và câu trả lời dưới dạng vector, hỗ trợ tìm kiếm ngữ nghĩa dựa trên độ tương đồng cosine.
- **Tiền Xử Lý Văn Bản**:
  - Loại bỏ ký tự đặc biệt, dấu câu, và stop words.
  - Chuẩn hóa văn bản (lemmatization, stemming).
  - Mở rộng từ viết tắt (contraction expansion).
  - Kiểm tra chính tả và nhận diện thực thể (NER) với **Spacy**.
- **Tăng Cường Dữ Liệu (Data Augmentation)**:
  - Thay thế từ đồng nghĩa, chèn/xóa từ ngẫu nhiên, hoán đổi từ.
  - Dịch xuôi ngược (back-translation) sử dụng **GoogleTranslator**.
  - Cân bằng nhãn (oversampling) để cải thiện hiệu suất mô hình.
- **Nhận Diện Thực Thể (NER)** và **POS Tagging**: Sử dụng **Spacy** và **NLTK** để phân tích cú pháp và ngữ nghĩa văn bản.

### 2. 🤖 Phát Triển Chatbot
- **Chatbot Q&A**: Xây dựng chatbot dựa trên dữ liệu câu hỏi - câu trả lời, sử dụng kỹ thuật tìm kiếm ngữ nghĩa để trả lời câu hỏi người dùng.
- **Giao Diện Tương Tác**: Giao diện chatbot (`index1.html`) được thiết kế với **HTML/CSS** và **JavaScript**, hỗ trợ hiển thị lịch sử trò chuyện và gửi tin nhắn qua API `/chat`.
- **Quản Lý Phiên (Session)**: Sử dụng **Flask-Session** để lưu trữ lịch sử trò chuyện, đảm bảo trải nghiệm liền mạch.

### 3. 🧠 Machine Learning
- **Hệ Thống Gợi Ý (Recommendation System)**:
  - Sử dụng **TruncatedSVD** để giảm chiều dữ liệu ma trận người dùng - sách.
  - Tính độ tương đồng cosine giữa người dùng để gợi ý sách dựa trên đánh giá cao từ người dùng tương tự.
  - Tải dữ liệu từ Google Drive sử dụng **gdown** và xử lý dữ liệu với **Pandas**.
- **Mô Hình Phân Loại Văn Bản**:
  - Hỗ trợ nhiều mô hình học máy: **Logistic Regression**, **Random Forest**, **Naive Bayes**, **SVM**, **KNN**.
  - Hỗ trợ mô hình học sâu: **DistilBERT**, **BERT-Tiny**, **RoBERTa**.
  - Các kỹ thuật vector hóa: **TF-IDF**, **Bag of Words**, **Bag of N-grams**, **One-Hot**, **Word2Vec**, **GloVe**.
  - Tạo biểu đồ đánh giá (metrics table, confusion matrix) với **Matplotlib** và **Seaborn**.
- **Vector Hóa Văn Bản**:
  - **TF-IDF**, **Bag of Words**, **Bag of N-grams**, **One-Hot Encoding**.
  - Nhúng phân tán (**Word2Vec**, **GloVe**).
  - Mô hình Transformer-based (**DistilBERT**, **BERT-Tiny**, **RoBERTa**).
- **Đánh Giá Mô Hình**: Tính toán **Accuracy**, **Precision**, **Recall**, **F1-Score** và vẽ confusion matrix.

### 4. 🌐 Phát Triển Web
- **Backend**: Sử dụng **Flask** để xử lý yêu cầu HTTP, quản lý API, và tích hợp các mô hình NLP/ML.
- **Frontend**: Giao diện thân thiện, responsive với **HTML/CSS/JavaScript**, hỗ trợ tương tác thời gian thực.
- **Quản Lý Dữ Liệu**: Tải và xử lý dữ liệu từ Google Drive, lưu trữ lịch sử đánh giá trong file JSON.
- **Triển Khai**: Chạy ứng dụng trên server cục bộ với cấu hình `host='0.0.0.0', port=8000`.

---

## 📽️ Video Demo

Video demo dự án được đăng tải tại:  
🔗 [Link Video Demo](https://drive.google.com/file/d/1VpxPtb0_KY4tywr1jeTJ0kr2rWxoKbmo/view?usp=sharing)

---

## 🛠️ Yêu Cầu
- Python 3.8+
- Các thư viện: `flask`, `sentence-transformers`, `transformers`, `scikit-learn`, `pandas`, `nltk`, `spacy`, `gdown`, `matplotlib`, `seaborn`, `textblob`, `deep-translator`, `gensim`, v.v.

## 📋 Cấu Trúc Dự Án

```plaintext
Personal-Project-NLP/
├── app.py                # File chính điều khiển ứng dụng Flask
├── chatbot.py            # Lớp Chatbot xử lý câu hỏi - câu trả lời
├── recommend.py          # Lớp Recommender cho hệ thống gợi ý sách
├── predict.py            # Lớp Predictor cho mô hình dự đoán nhãn
├── templates/
│   ├── index1.html       # Giao diện Chatbot
│   ├── index2.html       # Giao diện Recommendation System
│   ├── index3.html       # Giao diện Predict Model
├── static/               # Thư mục chứa tài nguyên tĩnh (CSS, favicon, hình ảnh)
├── README.md             # Tài liệu hướng dẫn
```

---

## 🌟 Điểm Nổi Bật

- **Tích Hợp Đa Chức Năng**: Kết hợp Chatbot, Recommendation System, và Predict Model trong một ứng dụng duy nhất.
- **Giao Diện Thân Thiện**: Thiết kế hiện đại, dễ sử dụng, hỗ trợ tương tác thời gian thực.
- **Tùy Chỉnh Linh Hoạt**: Người dùng có thể chọn kỹ thuật tiền xử lý, tăng cường dữ liệu, vector hóa, và mô hình học máy/học sâu.
- **Hiệu Suất Cao**: Tối ưu hóa xử lý dữ liệu và huấn luyện mô hình với các thư viện tiên tiến.
- **Đánh Giá Trực Quan**: Hiển thị kết quả đánh giá mô hình qua biểu đồ và bảng số liệu.

---
