from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from chatbot import Chatbot
from recommend import Recommender
from predict import Predictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'x7k9m2p8q3z6w5'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

chatbot = Chatbot('https://drive.google.com/uc?export=download&id=1qOP2PTbaDCfHOEgeyr1ebNtA_L2XSXFA')
recommender = Recommender()
predictor = Predictor()

@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index1.html', chat_history=session['chat_history'])

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json['message'].lower()
    if 'recommend' in user_question or 'recommend system' in user_question:
        response = 'This is Recommendation System": <a href="/recommend-system">link</a>'
    elif 'predict' in user_question or 'predict model' in user_question:
        response = 'This is Prediction Model": <a href="/predict-model">link</a>'
    else:
        response = chatbot.get_response(user_question)
    
    if 'chat_history' not in session:
        session['chat_history'] = []
    session['chat_history'].append({'user': request.json['message'], 'bot': response})
    session.modified = True
    
    return jsonify({'response': response})

@app.route('/recommend-system', methods=['GET', 'POST'])
def recommend_system():
    error = None
    recommended_books = []
    user_id = None
    num_books = 10

    if request.method == 'POST':
        user_id = request.form.get('user_id', type=int)
        num_books = request.form.get('num_books', type=int, default=10)
        recommended_books, error = recommender.recommend_books(user_id, num_books)

    return render_template('index2.html', error=error, recommended_books=recommended_books, user_id=user_id, num_books=num_books)

@app.route('/predict-model', methods=['GET', 'POST'])
def predict_model():
    if request.method == 'POST':
        if 'reset' in request.form:
            predictor.reset()
        elif 'file' in request.files:
            file = request.files['file']
            file_type = request.form['file_type']
            is_zip = 'is_zip' in request.form
            predictor.load_file(file, file_type, is_zip)
        elif 'crawl_url' in request.form:
            crawl_url = request.form['crawl_url']
            predictor.crawl_data(crawl_url)
        elif 'select_columns' in request.form:
            text_column = request.form['text_column']
            label_column = request.form['label_column']
            predictor.select_columns(text_column, label_column)
        elif 'augment' in request.form:
            text_augmentation = request.form.getlist('text_augmentation')
            label_augmentation = request.form.getlist('label_augmentation')
            augment_rows = request.form.get('augment_rows', type=int, default=0)
            augment_random = request.form.get('augment_random', '') == 'random'
            predictor.augment_data(text_augmentation, label_augmentation, augment_rows, augment_random)
        elif 'preprocess' in request.form:
            text_preprocessing = request.form.getlist('text_preprocessing')
            predictor.preprocess_data(text_preprocessing)
        elif 'train' in request.form:
            model_type = request.form['model_type']
            train_ratio = float(request.form['train_ratio'])
            vectorization = request.form.get('vectorization', request.form.get('distributional', 'bag_of_words'))
            predictor.train_model(model_type, train_ratio, vectorization)
        elif 'predict' in request.form:
            user_input = request.form['question']
            predictor.predict(user_input)

    data_rows = len(predictor.data) if predictor.data is not None else 0
    data_cols = len(predictor.columns) if predictor.columns else 0
    augment_random = 'random' if predictor.action == 'augment' and predictor.augment_random else ''

    return render_template('index3.html',
                          data_preview=predictor.data_preview(),
                          columns=predictor.columns,
                          text_column=predictor.text_column,
                          label_column=predictor.label_column,
                          labels=predictor.labels,
                          report_history=predictor.report_history,
                          user_input=predictor.user_input,
                          predicted_label=predictor.predicted_label,
                          error=predictor.error,
                          action=predictor.action,
                          text_augmentation=predictor.text_augmentation,
                          label_augmentation=predictor.label_augmentation,
                          text_preprocessing=predictor.text_preprocessing,
                          model_type=predictor.model_type,
                          train_ratio=predictor.train_ratio,
                          augment_preview=predictor.augment_preview,
                          preprocess_preview=predictor.preprocess_preview,
                          vectorization=predictor.vectorization_type,
                          data_rows=data_rows,
                          data_cols=data_cols,
                          augment_random=augment_random)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)