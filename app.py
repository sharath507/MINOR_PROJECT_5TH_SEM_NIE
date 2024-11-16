import os
import base64
import cv2
import pytesseract
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Extract text from image
    extracted_text = extract_text(file_path)
    
    # Save extracted text to CSV
    save_text_to_csv(extracted_text)

    return jsonify({'extracted_text': extracted_text})

def extract_text(image_path):
    # Read image using OpenCV
    img = cv2.imread(image_path)
    # Convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(gray)
    return text

def save_text_to_csv(text):
    df = pd.DataFrame({'Extracted Text': [text]})
    df.to_csv('extracted_texts.csv', mode='a', index=False, header=False)

@app.route('/capture', methods=['POST'])
def capture_image():
    data = request.json['image']
    # Decode the image data
    img_data = data.split(',')[1]
    img_data = np.frombuffer(base64.b64decode(img_data), np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    # Extract text from the captured image
    extracted_text = extract_text(img)
    save_text_to_csv(extracted_text)

    # Render capture.html with the extracted text
    return render_template('capture.html', extracted_text=extracted_text)

if __name__ == '__main__':
    app.run(debug=True)