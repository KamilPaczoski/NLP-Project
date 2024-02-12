from flask import Flask, render_template, request
from EmotionEvaluator import classify_emotion, labels
from fineTune import update_model
import os

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['sentence']
        emotion = labels[classify_emotion(user_input)]
        return render_template('result.html', sentence=user_input, emotion=emotion)
    return render_template('index.html')


@app.route('/feedback', methods=['POST'])
def feedback():
    user_input = request.form['sentence']
    guessed_emotion = request.form['guessed_emotion']
    user_emotion = request.form['user_emotion']
    correct_emotion = request.form['correct_emotion']
    feedback_file = 'feedback.txt'
    with open(feedback_file, 'a') as f:
        if user_emotion == 'correct':
            f.write(f"{user_input};{guessed_emotion}\n")
        elif user_emotion == 'incorrect':
            f.write(f"{user_input};{correct_emotion}\n")
    return 'Feedback submitted'


@app.route('/update_model', methods=['POST'])
def trigger_update_model():
    update_model()
    return render_template('index.html', updated=True)


if __name__ == '__main__':
    app.run(debug=True)
