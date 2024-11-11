import os
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify
import traceback

app = Flask(__name__)

# Paths to YOLO model configuration and weights files
model_cfg = "models/yolov3-tiny.cfg"
model_weights = "models/yolov3-tiny.weights"
labels_path = "models/coco.names"

# Load YOLO model
net = cv2.dnn.readNet(model_cfg, model_weights)

# Load class labels
with open(labels_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Output directory for JSON files
output_dir = "output/"
os.makedirs(output_dir, exist_ok=True)

@app.route('/predict', methods=['POST'])
def detect_objects():
    # Get the image from the request
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No file provided"}), 400

    # Decode the image
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Failed to decode image"}), 400

    # Perform object detection
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    results = []
    height, width = image.shape[:2]

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                results.append({
                    "class": classes[class_id],
                    "confidence": float(confidence),
                    "box": [x, y, int(w), int(h)]
                })

    # Create JSON response with detections
    response = {"detections": results}

    # Safe file naming with extension check
    file_name = os.path.splitext(file.filename)[0]  # Removes the extension from the filename
    output_filename = os.path.join(output_dir, f"{file_name}_detections.json")

    # Debug prints
    print(f"Output directory exists: {os.path.exists(output_dir)}")
    print(f"Saving detection results to: {output_filename}")

    try:
        # Log the output file path to confirm it's being generated correctly
        with open(output_filename, "w") as json_file:
            json.dump(response, json_file)
        print(f"Detection results saved to {output_filename}")
    except Exception as e:
        print(f"Failed to save JSON file: {e}")
        print(traceback.format_exc())  # Print the full traceback for debugging
        return jsonify({"error": "Failed to save JSON response", "details": str(e)}), 500

    # Return the response JSON
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
