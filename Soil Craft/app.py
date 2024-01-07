import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model/model_v3.h5")

# Define the allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


# Function to check if the file extension is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Route for the home page
@app.route("/")
def index():
    return render_template("index.html")


# Route for the predictions
@app.route("/predictions")
def predictions():
    return render_template("predictions.html")


# Route for the history
@app.route("/history")
def history():
    return render_template("history.html")


# Route to handle image classification
@app.route("/predict", methods=["POST"])
def predict():
    # Check if an image was uploaded
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    # Check if the file has an allowed extension
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file extension"})

    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join("static/uploads", filename)
    file.save(filepath)

    compress_image(filepath)

    # Open and preprocess the image
    img = Image.open(filepath)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Map the predicted class index to the actual class labels
    class_labels = {
        0: "Loamy Black Soil",
        1: "Clay Soil",
        2: "Peat Soil",
        3: "Loamy Red Soil",
        4: "Sandy Soil",
    }

    class_voices = {
        0: "black.mp3",
        1: "clay.mp3",
        2: "peat.mp3",
        3: "red.mp3",
        4: "sand.mp3",
    }

    class_characteristics = {
        1: "Clay soil is characterized by its fine particles (clay) that allow for poor drainage but excellent water retention. This can lead to waterlogging and root rot for plants.",
        0: "Loamy black soil typically contains 4-8% organic matter, which enhances soil structure, water retention, and nutrient availability.The dark color is due to the presence of humic substances, which are decomposed organic matter that contributes to soil fertility.Loamy black soil has a balanced texture of sand, silt, and clay, providing good drainage and aeration while retaining adequate moisture.",
        3: "The red color is due to the presence of iron oxides, which are formed during the weathering of iron-rich parent rocks under warm, humid conditions. Loamy red soil is generally low in nutrients, particularly nitrogen, phosphorus, and potassium, due to leaching and nutrient fixation.",
        2: "Peat soil typically contains 65-95% organic matter, which gives it a spongy texture and high water retention capacity. The dark color of peat soil is due to the presence of humic substances, which are decomposed organic matter. Peat soil is often poorly drained due to its low clay content and waterlogged conditions.",
        4: "Sandy soil is characterized by its large particles (sand) that allow for excellent drainage but poor water retention. This can lead to nutrient leaching and drought stress for plants.",
    }

    class_crops = {
        1: "Broccoli, Brussels sprouts, cabbage, celery, kale, lettuce, onions, spinach, Rice",
        0: "Wheat, corn, Soybeans, Potatoes, tomatoes, cucumbers, peppers, carrots, onions, Apples, grapes, strawberries",
        3: "Cassava, sweet potatoes, peanuts, sorghum, millet, upland rice, coffee and rubber",
        2: "Brussels sprouts, cabbage, celery, kale, lettuce, spinach, Blueberries, cranberries, raspberries",
        4: "Carrots, beets, cucumbers, melons, peppers, potatoes, radishes, sweet potatoes, Grapes, strawberries, blueberries, Corn, sorghum, millet, Beans, peas",
    }

    class_amendments = {
        1: [
            "Compost: Breaks down into humus, which improves soil structure and drainage.",
            "Sand: Increases pore space and facilitates water movement.",
            "Vermicompost: Enriches soil with nutrients and beneficial microorganisms.",
            "Green manure: Cover crops like rye or oats help loosen soil structure and add organic matter.",
        ],
        0: [
            "Crop Rotation: Rotating crops helps prevent nutrient depletion and soilborne diseases.",
            "Cover Crops: Planting cover crops during fallow periods adds organic matter and improves soil structure.",
            "Manure Application: Applying well-decomposed manure can supplement nutrients and enhance soil fertility.",
            "Integrated Pest Management: Practicing IPM helps control pests and diseases without harming beneficial soil organisms.",
        ],
        3: [
            "Lime Application: Applying lime can raise the soil pH, making it more suitable for a wider range of crops.",
            "Nutrient Supplementation: Adding fertilizers, particularly nitrogen, phosphorus, and potassium, can replenish nutrient deficiencies.",
            "Organic Matter Incorporation: Incorporating organic matter, such as compost or green manure, can improve soil structure, water retention, and nutrient availability.",
            "Erosion Control Practices: Implementing erosion control practices, such as cover cropping and terracing, can protect the soil from erosion, especially in sloping areas.",
        ],
        2: [
            "Liming: Applying lime can raise the soil pH, making it more suitable for a wider range of crops.",
            "Nutrient Supplementation: Adding fertilizers, particularly nitrogen, phosphorus, and potassium, can replenish nutrient deficiencies.",
            "Sand Incorporation: Mixing sand into peat soil can improve drainage and aeration.",
            "Organic Matter Incorporation: Adding organic matter, such as compost or green manure, can enhance soil structure, water retention, and nutrient availability.",
        ],
        4: [
            "Compost: Breaks down into humus, which improves soil structure and water retention.",
            "Peat moss: Highly absorbent material that enhances water retention.",
            "Vermicompost: Enriches soil with nutrients and beneficial microorganisms.",
            "Green manure: Cover crops like clover or rye add organic matter and fix nitrogen into the soil.",
        ],
    }

    class_fertilizers = {
        1: "clay.png",
        0: "loamy-black.png",
        3: "loamy-red.png",
        2: "peat.png",
        4: "sand.png",
    }

    class_irrigation = {
        1: "clay.png",
        0: "loamy-black.png",
        3: "loamy-red.png",
        2: "peat.png",
        4: "sand.png",
    }

    import requests

    # Replace 'your_api_url' with the actual URL of the API you want to access
    api_url = "http://pm.skilltainment.org/php/lackson-api.php"

    # Make a GET request
    response = requests.get(api_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # The API response is usually in JSON format
        data = response.json()
    else:
        print(f"Error: {response.status_code}")

    n = data["lackson"][-1]["n"]
    p = data["lackson"][-1]["p"]
    k = data["lackson"][-1]["k"]

    predicted_label = class_labels[predicted_class]
    predicted_voice = class_voices[predicted_class]
    predicted_characteristics = class_characteristics[predicted_class]
    predicted_crops = class_crops[predicted_class]
    predicted_amendments = class_amendments[predicted_class]
    predicted_fertilizers = class_fertilizers[predicted_class]
    predicted_irrigation = class_irrigation[predicted_class]

    predicted_probability = float(
        prediction[0][predicted_class]
    )  # Convert to Python float

    # Return the predicted labels and confidence scores
    return jsonify(
        {
            "result": predicted_label,
            "voice": predicted_voice,
            "probability": predicted_probability,
            "characteristics": predicted_characteristics,
            "crops": predicted_crops,
            "amendments": predicted_amendments,
            "fertilizer": predicted_fertilizers,
            "irrigation": predicted_irrigation,
            "n": n,
            "p": p,
            "k": k,
            "path": filename,
        }
    )

    # Return the predicted label
    # return jsonify({'result': predicted_label, 'path': filepath})


def compress_image(image_path):
    # Open the image
    img = Image.open(image_path)

    # Compress the image
    compressed_image = img.copy()
    compressed_image.save(image_path, optimize=True, quality=20)

    return image_path


if __name__ == "__main__":
    app.run(debug=True)
