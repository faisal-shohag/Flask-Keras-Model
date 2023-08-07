from flask import Flask
from flask import request
from flask import jsonify
import sys
from flask_cors import CORS

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model


from util import base64_to_pil

app = Flask(__name__)

CORS(app)
# print(sys.path)

image_class = ['Daisy', 'Dandelion', 'Rose', 'Sunflowers']

def get_model():
    global model
    model = load_model('flower.h5')
    print("* Model loaded!")
    

def preprocess_img(img):
    img = img.resize((224, 224))
    img_arr = tf.keras.utils.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr /= 255.0
    return img_arr


get_model()


@app.route("/")
def home():
    return "Hello Flask!"


@app.route('/predict', methods = ['POST'])
def hello():
    img = base64_to_pil(request.json)
    final_image = preprocess_img(img)
    prediction = model.predict(final_image).tolist()
    high_index = prediction[0].index(max(prediction[0]))
    class_name = image_class[high_index]
    print(prediction)
    print(high_index)
    response = {
        'Probabilities' : prediction,
        'Class': class_name 
    }

    return jsonify(response)
