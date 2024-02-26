# -*- coding: utf-8 -*-
from roberta import Roberta

class IntentClassifier:
    def __init__(self, model_path="xlm-roberta-large-custom-trained"):
        self.model_loaded = False
        self.model = Roberta()
        self.load(model_path)

    def load(self, file_path):
        self.model_loaded = self.model.load(file_path)

    def is_ready(self):
        return self.model_loaded

    def classify_intent_function(self, text):
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model.inference(text)


if __name__ == '__main__':
    model = IntentClassifier()
    print(model.is_ready())
