"""Small image upload page using the saved CNN."""
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
import torch
from predict import ROOT, load_model, predict

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
torch.set_num_threads(2)
model = None


@app.get('/')
def index():
    return send_from_directory(ROOT/'static', 'index.html')


@app.errorhandler(413)
def too_large(error):
    return jsonify(error='The image must be smaller than 5 MB.'), 413


@app.post('/predict')
def predict_image():
    global model
    if 'image' not in request.files:
        return jsonify(error='Select an image.'), 400
    try:
        with Image.open(request.files['image'].stream) as image:
            if image.width * image.height > 16000000:
                return jsonify(error='The image dimensions are too large.'), 400
            if model is None:
                if not (ROOT/'artifacts/cnn.pt').is_file():
                    return jsonify(error='Model not found. Run train.py first.'), 503
                model = load_model()
            result, _ = predict(model, image)
        return jsonify(result)
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
        return jsonify(error='The image could not be read, or no digit was found.'), 400


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
