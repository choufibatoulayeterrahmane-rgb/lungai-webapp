# LungAI Diagnostics — Web Application (v2)

A clinical decision-support interface for the LungAI Diagnostics system
described in the thesis. Implements the **deployed pipeline exactly as
in Chapter 3**:

- **CXR:** raw upload → U-Net lung segmentation → masked DenseNet-121 →
  dual-threshold decision rule (τ_COVID = 0.45, τ_LO = 0.40) → Grad-CAM
  computed on the masked input and overlaid on the original image.
- **CT:** raw upload → DenseNet-121 + SE attention → threshold-tuned
  rule (τ_Normal = 0.35) → Grad-CAM.

The four panels match the thesis spec: **Upload · Diagnosis · Explanation
· Session history**.

---

## What's new in v2

A second explainability view: **Detected Regions**. After computing
Grad-CAM++, the engine finds local maxima (peak detection), merges
nearby ones (DBSCAN clustering), and draws bounding boxes with
confidence labels on the original image. Colors follow a traffic-light
scheme: red > 75%, orange > 55%, yellow lower.

The doctor toggles between **Heatmap** (continuous attention) and
**Detected regions** (discrete bounded findings) in the Explanation
panel.

This is most useful for **CT lung cancer**, where pathology manifests
as discrete nodules rather than diffuse patterns. Bounded regions
with confidence scores are more clinically actionable for that case.
For CXR you can decide whether to keep it — both modalities support
the toggle so you can compare and choose.

---

## Quick start (demo mode — no models needed)

```bash
cd lungai_webapp
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>.

Demo mode runs the full interface (upload, predict, heatmap/regions
toggle, history) using deterministic synthetic predictions — enough to
**present the app live at your defense** even without the heavy model
files.

---

## Running with your real trained models

1. Install TensorFlow:

   ```bash
   pip install tensorflow==2.10.0
   ```

2. Copy your trained `.keras` files into the `models/` folder with
   these exact names:

   | File | What it is |
   |---|---|
   | `models/cxr_masked_model.keras` | your Phase-3 masked CXR model (`best_model_phase3_fixed.keras`) |
   | `models/ct_model.keras` | your CT DenseNet-121 + SE model |
   | `models/unet_lung_seg.hdf5` | pre-trained Montgomery+Shenzhen U-Net (512×512 grayscale, confirmed input shape) |

3. Restart `python app.py`. The status pill turns green ("models
   loaded") and real inference runs automatically — **no code changes
   needed**.

---

## Privacy

Uploaded images are read into memory and **never written to disk or
persisted**. Session history lives only in the browser tab and is
cleared on refresh — matching the privacy posture described in Chapter
3.

---

## Project structure

```
lungai_webapp/
├── app.py                  # Flask entry point
├── requirements.txt
├── models/                 # drop your .keras files here
├── utils/
│   └── inference.py        # the full thesis pipeline (with regions)
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## Defense talking points

- **Heatmap view**: "Continuous Grad-CAM showing where the model
  attends across the entire lung field. Useful for diffuse pathology
  like COVID ground-glass opacity."
- **Detected regions view**: "Grad-CAM++ followed by peak detection
  and DBSCAN clustering. This converts a continuous heatmap into
  discrete bounded findings with confidence scores — more clinically
  actionable for nodular pathology like CT lung cancer."
- **The toggle**: "The doctor chooses the visualization that fits the
  case — diffuse patterns or focal lesions."

This is a research prototype and a screening aid only — not a
substitute for radiologist review.
