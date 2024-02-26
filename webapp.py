from flask import Flask, render_template, request
from intent_classifier import IntentClassifier

app = Flask(__name__)

# Create an instance of the IntentClassifier
intent_classifier = IntentClassifier()

# Load the model when the application starts
intent_classifier.load("xlm-roberta-large-custom-trained")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if not intent_classifier.is_ready():
        return "Model is not ready. Please check the model file path."

    if request.method == 'POST':
        text = request.form['text']
        intent = intent_classifier.classify_intent_function(text)
        return render_template('result.html', text=text, intent=intent)

if __name__ == '__main__':
    app.run(debug=True)
