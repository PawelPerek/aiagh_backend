import json
from flask import Flask, request, redirect
from os.path import exists, isdir, join
from os import listdir
import numpy as np

import tensorflow as tf

MODEL_DIR = join(".", "models")
SAVED_DIR = join(".", "tmp")

arguments = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "spider", "squirrel"]

app = Flask(__name__,
    static_url_path="",
    static_folder="psi-frontend"    
)

@app.route("/", methods=["GET"])
def index():
    return redirect("/index.html")

@app.route("/health", methods=["GET"])
def healthcheck() -> str:
    return "Server OK"

@app.route("/list_models", methods=["GET"])
def list_models() -> list:
    if not exists(MODEL_DIR):
        print("There are no models folder on server")
        return []

    model_names = [
        directory for directory 
        in listdir(MODEL_DIR) 
        if isdir(join(MODEL_DIR, directory))
    ]

    return [
        {
            "id": name, 
            "name": name
        } 
        for name 
        in model_names
    ]

@app.route("/infere", methods=["POST"])
def infere():
    import uuid

    if not exists(MODEL_DIR):
        return []

    model_names: list[str] = request.headers.get("Models", default="").split(",")
    
    photo = request.files.get("animal")

    if photo is None or not photo.mimetype.startswith("image"):
        return [{"error": "Upload error"}]

    name = str(uuid.uuid4())
    path = join(SAVED_DIR, name)

    photo.save(path)

    result = []

    for model_name in model_names:
        model: tf.keras.Model = tf.keras.models.load_model(join(MODEL_DIR, model_name))

        model.summary()

        img = tf.keras.utils.load_img(path, target_size=(64, 64))
        img = tf.keras.utils.img_to_array(img)
        img = tf.expand_dims(img, 0)

        prediction = model.predict(img)

        result.append({
            "model": {
                "id": model_name,
                "name": model_name
            },
            "response": arguments[np.argmax(prediction)],
            "error": None
        })
    return result