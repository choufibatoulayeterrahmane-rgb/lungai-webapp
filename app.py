"""
LungAI Diagnostics - Flask Web Application
===========================================
A clinical decision-support interface for chest X-ray (4-class) and
CT lung cancer (3-class) classification with Grad-CAM explainability.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000

Privacy: uploaded images are processed in memory and never written to
disk or persisted. The session history lives only in the browser.
"""

import os
import io
from flask import Flask, render_template, request, jsonify
from PIL import Image

from utils.inference import predict_cxr, predict_ct, load_models, is_demo_mode, prepare_models

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "dcm"}


def allowed_file(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Reports whether real models are loaded or running in demo mode."""
    load_models()
    return jsonify({
        "status": "ok",
        "cxr_demo": is_demo_mode("cxr"),
        "ct_demo": is_demo_mode("ct"),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    modality = request.form.get("modality", "CXR").upper()

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. "
                                 "Use PNG, JPG, BMP or DICOM."}), 400

    # Read into memory only - never written to disk (privacy posture)
    try:
        img_bytes = file.read()
        filename_lower = file.filename.lower()

        if filename_lower.endswith(".dcm"):
            # DICOM support — extract pixel array and convert to PIL
            import pydicom
            import numpy as np
            dcm = pydicom.dcmread(io.BytesIO(img_bytes))
            pixel_array = dcm.pixel_array

            # Normalize to 0-255
            pixel_array = pixel_array.astype(np.float32)
            pixel_array -= pixel_array.min()
            if pixel_array.max() > 0:
                pixel_array /= pixel_array.max()
            pixel_array = (pixel_array * 255).astype(np.uint8)

            # Convert grayscale to RGB if needed
            if pixel_array.ndim == 2:
                pil_img = Image.fromarray(pixel_array).convert("RGB")
            elif pixel_array.ndim == 3:
                pil_img = Image.fromarray(pixel_array).convert("RGB")
            else:
                return jsonify({"error": "Unsupported DICOM pixel format."}), 400
        else:
            pil_img = Image.open(io.BytesIO(img_bytes))

    except Exception as e:
        return jsonify({"error": f"Could not read the image file: {e}"}), 400

    try:
        if modality == "CT":
            result = predict_ct(pil_img)
        else:
            result = predict_cxr(pil_img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  LungAI Diagnostics  -  starting up")
    print("=" * 60)
    prepare_models()
    load_models()
    if is_demo_mode("cxr") or is_demo_mode("ct"):
        print("\n  NOTE: one or more models not found in models/.")
        print("  Running in DEMO mode for those modalities.")
        print("  Drop your .keras files into models/ to use real models:")
        print("    models/cxr_masked_model.keras")
        print("    models/ct_model.keras")
        print("    models/unet_lung_seg.hdf5  (optional)")
    print("\n  Open  http://127.0.0.1:5000  in your browser.\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
