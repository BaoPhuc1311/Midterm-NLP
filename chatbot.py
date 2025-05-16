import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

class Chatbot:
    def __init__(self, data_source):
        self.questions = []
        self.answers = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_data(data_source)
        self.train()

    def load_data(self, data_source):
        try:
            if data_source.startswith('http'):
                df = pd.read_csv(data_source)
                if 'question' not in df.columns or 'answer' not in df.columns:
                    raise ValueError("CSV file must contain 'question' and 'answer' columns")
                # Loại bỏ giá trị NaN và chuyển thành danh sách
                df = df.dropna(subset=['question', 'answer'])
                if df.empty:
                    raise ValueError("CSV file is empty or contains no valid question-answer pairs")
                self.questions = df['question'].astype(str).tolist()
                self.answers = df['answer'].astype(str).tolist()
            else:
                # Xử lý file JSON cục bộ (giữ tương thích)
                import json
                with open(data_source, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if not data:
                        raise ValueError("JSON file is empty")
                    for item in data:
                        self.questions.append(str(item['question']))
                        self.answers.append(str(item['answer']))
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def train(self):
        if not self.questions or not self.answers or len(self.questions) != len(self.answers):
            raise ValueError("Invalid data: No questions or answers available for training")
        self.question_embeddings = self.model.encode(self.questions, convert_to_tensor=True)

    def get_response(self, user_question):
        if not self.question_embeddings.size:
            return "Error: Chatbot has not been trained with any data"
        user_embedding = self.model.encode(user_question, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(user_embedding, self.question_embeddings)[0]
        max_score_idx = np.argmax(cos_scores)
        return self.answers[max_score_idx]