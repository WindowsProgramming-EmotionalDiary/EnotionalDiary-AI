from flask import Flask, request, jsonify
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import numpy as np
from keras.applications.resnet import preprocess_input
from keras.layers import BatchNormalization
import tensorflow as tf
import os
import cv2

app = Flask(__name__)


# BERT Classifier model definition
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=7, dr_rate=0.5):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=dr_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# Set device to CPU
device = torch.device("cpu")

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bertmodel = BertModel.from_pretrained('bert-base-multilingual-cased')

# Initialize the BERT Classifier model
text_model = BERTClassifier(bertmodel).to(device)

# Load the trained model weights
try:
    text_model.load_state_dict(torch.load('bert_emotion_classification.pth', map_location=device))
    text_model.eval()
    print("BERT model loaded successfully.")
except Exception as e:
    print(f"Error loading BERT model: {e}")

# Emotion dictionary to decode numeric labels back to emotions
emotion_dict = {
    0: "불안",
    1: "놀람",
    2: "분노",
    3: "슬픔",
    4: "중립",
    5: "행복",
    6: "혐오"
}


# Function to predict emotion from a sentence
def predict_text_emotion(sentence):
    try:
        inputs = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = text_model(input_ids, attention_mask)
            logits = outputs[0]
            predicted_class = logits.argmax().item()
            print(f"Text emotion predicted: {predicted_class}")
            return predicted_class
    except Exception as e:
        print(f"Error predicting text emotion: {e}")
        raise


# Load the trained ResNet50 model
try:
    image_model = tf.keras.models.load_model("emotion_classification_model_resnet50.h5",
                                             custom_objects={'BatchNormalization': BatchNormalization}, compile=False)
    print("Image model loaded successfully.")
except Exception as e:
    print(f"Error loading image model: {e}")


# Function to predict emotion from an image
def predict_image_emotion(img_path, emotion1_value):
    try:
        img_size = 224

        # Load and preprocess the image using OpenCV
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        img_array = np.expand_dims(img, axis=0)
        img_array = preprocess_input(img_array.astype(np.float32)) / 255.0

        # Emotion1 input data
        emotion1_array = np.array([[emotion1_value]])

        # Predict
        prediction = image_model.predict([img_array, emotion1_array])
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_emotion = emotion_dict[predicted_class]

        return predicted_emotion
    except Exception as e:
        print(f"Error predicting image emotion: {e}")
        raise


@app.route('/', methods=['GET', 'POST'])
def test():
    return '감정일기'


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '이미지 파일이 포함되지 않았습니다.'}), 400

        sentence = request.form['sentence']
        img_file = request.files['image']

        # Create uploads directory if it does not exist
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        img_path = os.path.join('uploads', img_file.filename)
        img_file.save(img_path)

        text_emotion_value = predict_text_emotion(sentence)
        image_emotion = predict_image_emotion(img_path, text_emotion_value)

        os.remove(img_path)
        print("result "+image_emotion)
        print('Image file removed')

        return jsonify({'sentence': sentence, 'predicted_emotion': image_emotion})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
