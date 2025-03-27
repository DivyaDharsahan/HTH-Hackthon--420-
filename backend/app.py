print("om nama siva")


from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from classifiers import Meso4
from pipeline import preprocess_input



app = Flask(__name__)
CORS(app)

model_path = "model/deepfake_model.h5"
classifier = Meso4()
classifier.load(model_path)

print("✅ deepfake_model.h5 loaded successfully!")
