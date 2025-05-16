import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from bs4 import BeautifulSoup
import requests
import os
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import Word, TextBlob
from deep_translator import GoogleTranslator
import spacy
import random
import string
import io
import json
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# Tự động tải tài nguyên nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

class Predictor:
    def __init__(self):
        self.data = None
        self.columns = []
        self.text_column = None
        self.label_column = None
        self.labels = []
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self.vectorization_type = None
        self.report_history = []
        self.user_input = None
        self.predicted_label = None
        self.error = None
        self.text_augmentation = []
        self.label_augmentation = []
        self.text_preprocessing = []
        self.vectorization = None
        self.train_ratio = None
        self.augment_preview = None
        self.preprocess_preview = None
        self.action = ''
        self.augment_random = False
        os.makedirs('static', exist_ok=True)
        # Tải report_history từ file JSON nếu tồn tại
        self.load_report_history()

    def load_report_history(self):
        """Tải report_history từ file JSON."""
        try:
            if os.path.exists('report_history.json'):
                with open('report_history.json', 'r') as f:
                    self.report_history = json.load(f)
        except Exception as e:
            print(f"Error loading report_history: {str(e)}")
            self.report_history = []

    def save_report_history(self):
        """Lưu report_history vào file JSON."""
        try:
            with open('report_history.json', 'w') as f:
                json.dump(self.report_history, f)
        except Exception as e:
            print(f"Error saving report_history: {str(e)}")

    def reset(self):
        self.data = None
        self.columns = []
        self.text_column = None
        self.label_column = None
        self.labels = []
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self.vectorization_type = None
        self.report_history = []
        self.user_input = None
        self.predicted_label = None
        self.error = None
        self.text_augmentation = []
        self.label_augmentation = []
        self.text_preprocessing = []
        self.vectorization = None
        self.train_ratio = None
        self.augment_preview = None
        self.preprocess_preview = None
        self.action = ''
        self.augment_random = False
        # Lưu report_history rỗng để xóa file
        self.save_report_history()

    def load_file(self, file, file_type, is_zip):
        try:
            if is_zip:
                with zipfile.ZipFile(file) as z:
                    with z.open(z.namelist()[0]) as f:
                        file_content = io.BytesIO(f.read())
                        if file_type == 'csv':
                            self.data = pd.read_csv(file_content)
                        elif file_type == 'tsv':
                            self.data = pd.read_csv(file_content, sep='\t')
                        elif file_type == 'xlsx':
                            self.data = pd.read_excel(file_content)
                        elif file_type == 'json':
                            self.data = pd.read_json(file_content)
                        elif file_type == 'txt':
                            self.data = pd.read_csv(file_content, sep=r'\s+', header=None)
            else:
                if file_type == 'csv':
                    self.data = pd.read_csv(file)
                elif file_type == 'tsv':
                    self.data = pd.read_csv(file, sep='\t')
                elif file_type == 'xlsx':
                    self.data = pd.read_excel(file)
                elif file_type == 'json':
                    self.data = pd.read_json(file)
                elif file_type == 'txt':
                    self.data = pd.read_csv(file, sep=r'\s+', header=None)
            self.columns = self.data.columns.tolist()
            self.auto_select_columns()
            self.error = None
            self.action = 'load'
        except Exception as e:
            self.error = f"Error loading file: {str(e)}"
            self.action = 'load'

    def crawl_data(self, url):
        try:
            quotes = []
            labels = []
            page = 1
            while True:
                response = requests.get(f"{url}/page/{page}/")
                soup = BeautifulSoup(response.text, 'html.parser')
                page_quotes = [quote.text.strip() for quote in soup.find_all('span', class_='text')]
                if not page_quotes:
                    break
                quotes.extend(page_quotes)
                for quote in page_quotes:
                    blob = TextBlob(quote)
                    polarity = blob.sentiment.polarity
                    if polarity > 0.1:
                        label = 'positive'
                    elif polarity < -0.1:
                        label = 'negative'
                    else:
                        label = 'neutral'
                    labels.append(label)
                page += 1
            if not quotes:
                raise ValueError("No quotes found on the provided URL.")
            self.data = pd.DataFrame({'text': quotes, 'label': labels})
            self.columns = self.data.columns.tolist()
            self.auto_select_columns()
            self.error = None
            self.action = 'crawl'
        except Exception as e:
            self.error = f"Error crawling URL: {str(e)}"
            self.action = 'crawl'

    def auto_select_columns(self):
        if self.data is None or not self.columns:
            return
        try:
            text_scores = {}
            for col in self.columns:
                try:
                    avg_length = self.data[col].astype(str).apply(len).mean()
                    text_scores[col] = avg_length
                except:
                    continue
            if text_scores:
                self.text_column = max(text_scores, key=text_scores.get)
            else:
                self.text_column = None
                self.error = "Could not identify text column."

            label_candidates = []
            for col in self.columns:
                try:
                    col_data = self.data[col].fillna('').astype(str)
                    unique_values = col_data.nunique()
                    if 2 <= unique_values <= 5:
                        label_candidates.append((col, unique_values))
                except:
                    continue
            if label_candidates:
                self.label_column = min(label_candidates, key=lambda x: abs(x[1] - 3.5))[0]
                self.labels = self.data[self.label_column].dropna().unique().tolist()
            else:
                for col in self.columns:
                    try:
                        col_data = self.data[col].fillna('').astype(str)
                        unique_values = col_data.nunique()
                        if 1 <= unique_values <= 6:
                            label_candidates.append((col, unique_values))
                    except:
                        continue
                if label_candidates:
                    self.label_column = min(label_candidates, key=lambda x: abs(x[1] - 3.5))[0]
                    self.labels = self.data[self.label_column].dropna().unique().tolist()
                else:
                    self.label_column = None
                    self.labels = []
                    self.error = "Could not identify label column with 2-5 unique values."
        except Exception as e:
            self.error = f"Error auto-selecting columns: {str(e)}"

    def select_columns(self, text_column, label_column):
        if self.data is None:
            self.error = "No data loaded. Please load data first."
            self.action = 'select_columns'
            return
        self.text_column = text_column
        self.label_column = label_column
        if not text_column or not label_column:
            self.error = "Please select both text and label columns."
        else:
            self.labels = self.data[label_column].dropna().unique().tolist()
            self.error = None
        self.action = 'select_columns'

    def augment_data(self, text_augmentation, label_augmentation, augment_rows, augment_random):
        if self.data is None or self.text_column is None or self.label_column is None:
            self.error = "Please load data and select columns first."
            self.action = 'augment'
            return
        self.text_augmentation = text_augmentation
        self.label_augmentation = label_augmentation
        self.augment_random = augment_random
        try:
            augmented_data = self.data.copy()
            techniques = text_augmentation
            if augment_random:
                techniques = random.sample(text_augmentation, k=max(1, len(text_augmentation)//2))
            for technique in techniques:
                if technique == 'synonym':
                    augmented_data[self.text_column] = augmented_data[self.text_column].apply(lambda x: self.synonym_replacement(str(x)))
                elif technique == 'random_insert':
                    augmented_data[self.text_column] = augmented_data[self.text_column].apply(lambda x: self.random_word_insertion(str(x)))
                elif technique == 'back_translation':
                    augmented_data[self.text_column] = augmented_data[self.text_column].apply(self.back_translation)
                elif technique == 'word_dropout':
                    augmented_data[self.text_column] = augmented_data[self.text_column].apply(lambda x: self.word_dropout(str(x)))
                elif technique == 'entity_replacement':
                    augmented_data[self.text_column] = augmented_data[self.text_column].apply(self.entity_replacement)
                elif technique == 'random_word_swap':
                    augmented_data[self.text_column] = augmented_data[self.text_column].apply(lambda x: self.random_word_swap(str(x)))
                elif technique == 'random_word_deletion':
                    augmented_data[self.text_column] = augmented_data[self.text_column].apply(lambda x: self.random_word_deletion(str(x)))
            if 'balance' in label_augmentation:
                max_count = augmented_data[self.label_column].value_counts().max()
                balanced_data = []
                for label in self.labels:
                    label_data = augmented_data[augmented_data[self.label_column] == label]
                    balanced_data.append(label_data.sample(max_count, replace=True, random_state=42))
                augmented_data = pd.concat(balanced_data)
            if augment_rows > 0:
                additional_data = augmented_data.sample(n=augment_rows, replace=True, random_state=42)
                augmented_data = pd.concat([augmented_data, additional_data]).reset_index(drop=True)
            else:
                additional_data = augmented_data.sample(frac=1.0, replace=True, random_state=42)
                augmented_data = pd.concat([augmented_data, additional_data]).reset_index(drop=True)
            self.data = augmented_data
            self.augment_preview = self.data.head().to_dict(orient='records')
            self.error = None
            self.action = 'augment'
        except Exception as e:
            self.error = f"Error during augmentation: {str(e)}"
            self.action = 'augment'

    def preprocess_data(self, text_preprocessing):
        if self.data is None or self.text_column is None or self.label_column is None:
            self.error = "Please load data and select columns first."
            self.action = 'preprocess'
            return
        self.text_preprocessing = text_preprocessing
        try:
            processed_data = self.data.copy()
            for technique in text_preprocessing:
                if technique == 'remove_special':
                    processed_data[self.text_column] = processed_data[self.text_column].apply(self.remove_special_chars)
                elif technique == 'remove_stopwords':
                    processed_data[self.text_column] = processed_data[self.text_column].apply(self.remove_stopwords)
                elif technique == 'lemmatize':
                    processed_data[self.text_column] = processed_data[self.text_column].apply(self.lemmatize)
                elif technique == 'remove_punctuation':
                    processed_data[self.text_column] = processed_data[self.text_column].apply(self.remove_punctuation)
                elif technique == 'stemming':
                    processed_data[self.text_column] = processed_data[self.text_column].apply(self.stemming)
                elif technique == 'pos_tagging':
                    processed_data[self.text_column] = processed_data[self.text_column].apply(self.pos_tagging)
                elif technique == 'expand_contraction':
                    processed_data[self.text_column] = processed_data[self.text_column].apply(self.expand_contraction)
                elif technique == 'spellcheck':
                    processed_data[self.text_column] = processed_data[self.text_column].apply(self.spellcheck)
                elif technique == 'ner':
                    processed_data[self.text_column] = processed_data[self.text_column].apply(self.ner)
            processed_data[self.text_column] = processed_data[self.text_column].fillna("").astype(str)
            self.data = processed_data
            self.preprocess_preview = self.data.head().to_dict(orient='records')
            self.error = None
            self.action = 'preprocess'
        except Exception as e:
            self.error = f"Error during preprocessing: {str(e)}"
            self.action = 'preprocess'

    def train_model(self, model_type, train_ratio, vectorization):
        if self.data is None or self.text_column is None or self.label_column is None:
            self.error = "Please complete data loading, column selection, and preprocessing first."
            self.action = 'train'
            return
        self.model_type = model_type
        self.vectorization_type = vectorization
        self.train_ratio = train_ratio
        try:
            X = self.data[self.text_column]
            y = self.data[self.label_column]
            valid_idx = X.str.strip() != ""
            X = X[valid_idx]
            y = y[valid_idx]
            if len(X) == 0:
                raise ValueError("No valid data after preprocessing.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)

            if vectorization in ['distilbert', 'bert-tiny', 'roberta']:
                model_name = {
                    'distilbert': 'distilbert-base-uncased',
                    'bert-tiny': 'prajjwal1/bert-tiny',
                    'roberta': 'roberta-base'
                }[vectorization]
                tokenizer_class = RobertaTokenizer if vectorization == 'roberta' else DistilBertTokenizer
                model_class = RobertaForSequenceClassification if vectorization == 'roberta' else DistilBertForSequenceClassification
                self.tokenizer = tokenizer_class.from_pretrained(model_name)
                self.model = model_class.from_pretrained(model_name, num_labels=len(self.labels))

                train_encodings = self.tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128)
                test_encodings = self.tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=128)

                label_map = {label: i for i, label in enumerate(self.labels)}
                y_train_idx = [label_map[label] for label in y_train]
                y_test_idx = [label_map[label] for label in y_test]

                class Dataset(torch.utils.data.Dataset):
                    def __init__(self, encodings, labels):
                        self.encodings = encodings
                        self.labels = labels
                    def __getitem__(self, idx):
                        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                        item['labels'] = torch.tensor(self.labels[idx])
                        return item
                    def __len__(self):
                        return len(self.labels)

                train_dataset = Dataset(train_encodings, y_train_idx)
                test_dataset = Dataset(test_encodings, y_test_idx)

                training_args = TrainingArguments(
                    output_dir='./results',
                    num_train_epochs=3,
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    warmup_steps=500,
                    weight_decay=0.01,
                    logging_dir='./logs',
                    logging_steps=10,
                )

                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset
                )

                trainer.train()

                predictions = trainer.predict(test_dataset)
                y_pred = np.argmax(predictions.predictions, axis=1)
                y_pred = [self.labels[idx] for idx in y_pred]
            else:
                if vectorization == 'tfidf':
                    vectorizer = TfidfVectorizer(max_features=5000)
                elif vectorization == 'bag_of_ngrams':
                    vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
                elif vectorization == 'one_hot':
                    vectorizer = CountVectorizer(max_features=5000, binary=True)
                elif vectorization == 'glove':
                    glove_path = 'glove.6B.100d.txt'
                    if not os.path.exists(glove_path):
                        raise FileNotFoundError("GloVe embeddings not found. Please download glove.6B.100d.txt.")
                    glove_vectors = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
                    def get_glove_features(text):
                        words = word_tokenize(text.lower())
                        vectors = [glove_vectors[word] for word in words if word in glove_vectors]
                        return np.mean(vectors, axis=0) if vectors else np.zeros(100)
                    X_train_vec = np.array([get_glove_features(text) for text in X_train])
                    X_test_vec = np.array([get_glove_features(text) for text in X_test])
                    self.tokenizer = glove_vectors
                elif vectorization == 'word2vec':
                    sentences = [word_tokenize(text.lower()) for text in X_train]
                    w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
                    def get_w2v_features(text):
                        words = word_tokenize(text.lower())
                        vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
                        return np.mean(vectors, axis=0) if vectors else np.zeros(100)
                    X_train_vec = np.array([get_w2v_features(text) for text in X_train])
                    X_test_vec = np.array([get_w2v_features(text) for text in X_test])
                    self.tokenizer = w2v_model
                else:
                    vectorizer = CountVectorizer(max_features=5000)

                if vectorization in ['tfidf', 'bag_of_words', 'bag_of_ngrams', 'one_hot']:
                    X_train_vec = vectorizer.fit_transform(X_train)
                    X_test_vec = vectorizer.transform(X_test)
                    self.tokenizer = vectorizer

                if model_type == 'logistic':
                    self.model = LogisticRegression(max_iter=1000)
                elif model_type == 'random_forest':
                    self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_type == 'naive_bayes':
                    self.model = MultinomialNB()
                elif model_type == 'svm':
                    self.model = SVC(kernel='linear', random_state=42)
                elif model_type == 'knn':
                    self.model = KNeighborsClassifier(n_neighbors=5)

                self.model.fit(X_train_vec, y_train)
                y_pred = self.model.predict(X_test_vec)

            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            }

            cm = confusion_matrix(y_test, y_pred, labels=self.labels)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_file = f'metrics_{timestamp}.png'
            plt.figure(figsize=(8, 2))
            table = plt.table(cellText=[[f'{v:.4f}' for v in metrics.values()]],
                              colLabels=list(metrics.keys()),
                              loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            plt.axis('off')
            plt.savefig(f'static/{metrics_file}', bbox_inches='tight', dpi=100)
            plt.close()

            cm_file = f'cm_{timestamp}.png'
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.labels, yticklabels=self.labels, cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(f'static/{cm_file}', bbox_inches='tight', dpi=100)
            plt.close()

            self.report_history.append({'timestamp': timestamp, 'metrics_file': metrics_file, 'cm_file': cm_file})
            self.save_report_history()  # Lưu report_history sau khi thêm báo cáo mới
            self.error = None
            self.action = 'train'
        except Exception as e:
            self.error = f"Error during training: {str(e)}"
            self.action = 'train'

    def predict(self, user_input):
        if self.model is None or self.tokenizer is None or self.model_type is None:
            self.error = "Please train a model first."
            self.action = 'predict'
            return
        if not user_input or not user_input.strip():
            self.error = "Input text cannot be empty."
            self.action = 'predict'
            return
        if not self.labels:
            self.error = "No labels available. Please select a valid label column."
            self.action = 'predict'
            return
        self.user_input = user_input
        try:
            if self.vectorization_type in ['distilbert', 'bert-tiny', 'roberta']:
                inputs = self.tokenizer(user_input, return_tensors='pt', truncation=True, padding=True, max_length=128)
                outputs = self.model(**inputs)
                predicted_idx = torch.argmax(outputs.logits, dim=1).item()
                self.predicted_label = self.labels[predicted_idx]
            else:
                if self.vectorization_type == 'word2vec':
                    words = word_tokenize(user_input.lower())
                    vectors = [self.tokenizer.wv[word] for word in words if word in self.tokenizer.wv]
                    if not vectors:
                        self.error = "Input text contains no recognizable words for Word2Vec model."
                        self.action = 'predict'
                        return
                    input_vec = np.mean(vectors, axis=0).reshape(1, -1)
                elif self.vectorization_type == 'glove':
                    words = word_tokenize(user_input.lower())
                    vectors = [self.tokenizer[word] for word in words if word in self.tokenizer]
                    if not vectors:
                        self.error = "Input text contains no recognizable words for GloVe model."
                        self.action = 'predict'
                        return
                    input_vec = np.mean(vectors, axis=0).reshape(1, -1)
                else:
                    input_vec = self.tokenizer.transform([user_input])
                self.predicted_label = self.model.predict(input_vec)[0]
            self.error = None
            self.action = 'predict'
        except Exception as e:
            self.error = f"Error during prediction: {str(e)}"
            self.action = 'predict'
            print(f"Prediction error: {str(e)}")

    def data_preview(self):
        return self.data.head().to_dict(orient='records') if self.data is not None else None

    def synonym_replacement(self, text):
        words = word_tokenize(text)
        new_words = words.copy()
        for i in range(len(words)):
            if random.random() < 0.1:
                synonyms = [syn.lemmas()[0].name() for syn in Word(words[i]).synsets]
                if synonyms:
                    new_words[i] = random.choice(synonyms)
        return ' '.join(new_words)

    def random_word_insertion(self, text):
        words = word_tokenize(text)
        new_words = words.copy()
        for _ in range(int(len(words) * 0.1)):
            pos = random.randint(0, len(new_words))
            new_words.insert(pos, random.choice(words))
        return ' '.join(new_words)

    def back_translation(self, text):
        try:
            translated = GoogleTranslator(source='en', target='fr').translate(text)
            back_translated = GoogleTranslator(source='fr', target='en').translate(translated)
            return back_translated
        except:
            return text

    def word_dropout(self, text):
        words = word_tokenize(text)
        new_words = [word for word in words if random.random() > 0.1]
        return ' '.join(new_words)

    def entity_replacement(self, text):
        doc = nlp(text)
        new_text = text
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG']:
                new_text = new_text.replace(ent.text, f"{ent.label_}_{random.randint(1, 100)}")
        return new_text

    def random_word_swap(self, text):
        words = word_tokenize(text)
        if len(words) < 2:
            return text
        new_words = words.copy()
        i, j = random.sample(range(len(words)), 2)
        new_words[i], new_words[j] = new_words[j], new_words[i]
        return ' '.join(new_words)

    def random_word_deletion(self, text):
        words = word_tokenize(text)
        new_words = [word for word in words if random.random() > 0.2]
        return ' '.join(new_words) if new_words else text

    def remove_special_chars(self, text):
        return re.sub(r'[^\w\s]', '', str(text))

    def remove_stopwords(self, text):
        return ' '.join(word for word in word_tokenize(str(text)) if word.lower() not in stop_words)

    def lemmatize(self, text):
        return ' '.join(Word(word).lemmatize() for word in word_tokenize(str(text)))

    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def stemming(self, text):
        stemmer = PorterStemmer()
        return ' '.join(stemmer.stem(word) for word in word_tokenize(str(text)))

    def pos_tagging(self, text):
        tokens = word_tokenize(str(text))
        pos_tags = nltk.pos_tag(tokens)
        return ' '.join(f"{word}_{tag}" for word, tag in pos_tags)

    def expand_contraction(self, text):
        contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "won't": "will not", "wouldn't": "would not", "can't": "cannot",
            "i'm": "i am", "you're": "you are", "he's": "he is"
        }
        for contraction, expansion in contractions.items():
            text = text.lower().replace(contraction, expansion)
        return text

    def spellcheck(self, text):
        return str(TextBlob(text).correct())

    def ner(self, text):
        doc = nlp(text)
        new_text = text
        for ent in doc.ents:
            new_text = new_text.replace(ent.text, f"[{ent.label_}]")
        return new_text