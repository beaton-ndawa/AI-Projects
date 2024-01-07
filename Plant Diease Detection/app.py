import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model/model.h5")

# Define the allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


# Function to check if the file extension is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Route for the home page
@app.route("/")
def index():
    return render_template("index.html")


# Route to handle image classification
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file extension"})

    filename = secure_filename(file.filename)
    filepath = os.path.join("static/uploads", filename)
    file.save(filepath)

    compress_image(filepath)

    img = Image.open(filepath)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    # Get the predicted class index and probability
    predicted_class = np.argmax(prediction)
    predicted_probability = float(
    prediction[0][predicted_class]
    )  # Convert to Python float

    original_dict = {
        "Maize___Cercospora_leaf_spot Gray_leaf_spot": 0,
        "Maize___Common_rust_": 1,
        "Maize___Northern_Leaf_Blight": 2,
        "Maize___healthy": 3,
        "Potato___Early_blight": 4,
        "Potato___Late_blight": 5,
        "Potato___healthy": 6,
        "Tomato___Bacterial_spot": 7,
        "Tomato___Early_blight": 8,
        "Tomato___Late_blight": 9,
        "Tomato___Leaf_Mold": 10,
        "Tomato___Septoria_leaf_spot": 11,
        "Tomato___Spider_mites Two-spotted_spider_mite": 12,
        "Tomato___Target_Spot": 13,
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 14,
        "Tomato___Tomato_mosaic_virus": 15,
        "Tomato___healthy": 16,
    }

    # Create dictionary for plant types
    plant_types_dict = {v: k.split("___")[0] for k, v in original_dict.items()}

    # Create dictionary for diseases
    diseases_dict = {v: k.split("___", 1)[1] for k, v in original_dict.items()}

    predicted_plant_label = plant_types_dict[predicted_class]
    predicted_disease_label = diseases_dict[predicted_class]

    # Return the predicted label and probability
    return jsonify(
        {
            "plant": predicted_plant_label,
            "disease": predicted_disease_label,
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
