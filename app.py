from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = tf.keras.models.load_model("model.h5")

with open("class_labels.pkl", "rb") as f:
    class_labels = pickle.load(f)

class_names = list(class_labels.keys())

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    filename = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            with open("last_input_image.pkl", "wb") as f:
                pickle.dump(img_array, f)

            preds = model.predict(img_array)[0]
            index = np.argmax(preds)

            prediction = class_names[index]
            confidence = round(float(preds[index]) * 100, 2)
            filename = file.filename

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
