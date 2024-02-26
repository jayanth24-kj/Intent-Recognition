# app.py

from flask import Flask, render_template, request, jsonify
from intent_classifier import IntentClassifier

app = Flask(__name__)
model = IntentClassifier()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/intent', methods=['POST'])
def classify_intent():
    try:
        # Check if the request has a JSON body
        if not request.json:
            return jsonify({"label": "BODY_MISSING", "message": "Request doesn't have a body."}), 400

        # Check if the 'text' field is present in the request JSON
        if 'text' not in request.json:
            return jsonify({"label": "TEXT_MISSING", "message": "\"text\" missing from request body."}), 400

        # Check if the 'text' field is a string
        if not isinstance(request.json['text'], str):
            return jsonify({"label": "INVALID_TYPE", "message": "\"text\" is not a string."}), 400

        # Check if the 'text' field is empty
        if not request.json['text'].strip():
            return jsonify({"label": "TEXT_EMPTY", "message": "\"text\" is empty."}), 400

        # Perform intent classification
        intents = model.classify_intent_function(request.json['text'])

        # Create a response with the top 3 intent predictions
        response = {"intents": intents[:3]}

        return render_template('index.html', result=response)

    except Exception as e:
        # Handle internal errors
        return jsonify({"label": "INTERNAL_ERROR", "message": str(e)}), 500


if __name__ == '__main__':
    # Load the model when the application starts
    model.load("xlm-roberta-large-custom-trained")
    app.run(debug=True)
