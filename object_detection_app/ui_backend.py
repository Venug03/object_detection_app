from flask import Flask, request, jsonify, render_template, send_file
import requests
import os

app = Flask(__name__)

# Directories for uploads and outputs
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set the URL for the AI backend
AI_BACKEND_URL = 'http://ai_backend:5001/predict'

@app.route('/')
def upload_page():
    """Renders the upload page."""
    return render_template('upload.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Save the uploaded file
    uploaded_file = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
    uploaded_file.save(file_path)

    # Send the image to the AI backend
    with open(file_path, 'rb') as f:
        response = requests.post(AI_BACKEND_URL, files={'image': f})

    # Check the response from the AI backend
    if response.status_code == 200:
        detections = response.json().get("detections", [])
        return jsonify({"detections": detections})
    else:
        return jsonify({"error": "Object detection failed"}), response.status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
