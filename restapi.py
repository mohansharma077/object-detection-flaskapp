import argparse
import io
from PIL import Image

import torch
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5"


@app.route(DETECTION_URL, methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        if request.files.get("image"):
            image_file = request.files["image"]
            image_bytes = image_file.read()
            img = Image.open(io.BytesIO(image_bytes))
            results = model(img, size=640)  # reduce size=320 for faster inference

            cropped_images = []
            for result in results.pred:
                for *box, _, cls in result:
                    x1, y1, x2, y2 = map(int, box)
                    cropped_image = img.crop((x1, y1, x2, y2))
                    cropped_images.append(cropped_image)

            # Store cropped images and image URLs in session
            for i, cropped_image in enumerate(cropped_images):
                image_url = f"cropped_image_{i}.jpg"
                cropped_image.save(f"static/{image_url}")
                session[image_url] = cropped_image

            return render_template("index.html", image_urls=session.keys())

    elif request.method == "GET":
        return render_template("index.html", image_urls=session.keys())

    return


@app.route("/crop/<image_url>", methods=["GET"])
def crop(image_url):
    cropped_image = session.get(image_url)
    if cropped_image:
        cropped_image.save("static/cropped.jpg")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--model', default='yolov5s', help='model to run, i.e. --model yolov5s')
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', args.model)
    session = {}

    app.run(host="0.0.0.0", port=args.port)
