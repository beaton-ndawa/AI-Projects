import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model/meat-model v2.h5")

# Define the allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


# Function to check if the file extension is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Route for the home page
@app.route("/")
def index():
    return render_template("index.html")


# Route for the home page
@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/evaluate")
def evaluate():
    return render_template("evaluate.html")


# Route to handle image classification
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file extension"})

    filename = secure_filename(file.filename)
    filepath = filename
    file.save(filepath)

    compress_image(filepath)

    img = Image.open(filepath)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    # Get the predicted class index and probability
    predicted_class = np.argmax(prediction)
    predicted_probability = float(prediction[0][predicted_class])  # Convert to Python float
    class_labels = {0: "Fresh", 1: "Half Fresh", 2: "Spoiled"}

    predicted_label = class_labels[predicted_class]
    

    # Return the predicted label and probability
    return jsonify(
        {
            "result": predicted_label,
            "probability": predicted_probability,
            "path": filepath,
        }
    )


def compress_image(image_path):
    # Open the image
    img = Image.open(image_path)

    # Compress the image
    compressed_image = img.copy()
    compressed_image.save(image_path, optimize=True, quality=20)

    return image_path


if __name__ == "__main__":
    app.run(debug=True)
