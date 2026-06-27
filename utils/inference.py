"""
LungAI Diagnostics - Inference Engine (v2)
===========================================
Implements the deployed pipeline from Chapter 3 of the thesis,
with TWO complementary explainability views per prediction:

  1. HEATMAP view  : standard Grad-CAM overlay  (continuous attention)
  2. REGIONS view  : Grad-CAM++ -> peak detection -> DBSCAN clustering
                     -> bounding boxes with confidence labels
                     (discrete, clinician-friendly localisation)

The doctor chooses between the two in the UI. Both views are produced
on every prediction so switching is instant.

  CXR:  raw upload -> U-Net lung segmentation -> masked DenseNet-121
        -> dual-threshold decision rule -> Heatmap + Regions

  CT:   raw upload -> DenseNet-121 + SE attention -> threshold-tuned
        decision rule -> Heatmap + Regions (with geometric lung mask
        since no CT segmentation model is available)

Demo mode is preserved: if .keras files are missing the engine returns
synthetic placeholders so the UI is always usable.
"""

import os
import io
import base64
import numpy as np
from PIL import Image

# TensorFlow loaded lazily (demo mode works without it)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# scipy + skimage power the peak-detection / clustering pipeline.
# Loaded lazily so the app still starts without them (heatmap-only mode).
try:
    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max
    from sklearn.cluster import DBSCAN
    REGIONS_AVAILABLE = True
except Exception:
    REGIONS_AVAILABLE = False
# --------------------------------------------------------------------------
# Model encryption / decryption  (optional security layer)
# --------------------------------------------------------------------------
def prepare_models():
    """
    If encrypted .enc model files exist AND a LUNGAI_SECRET_KEY environment
    variable is set, decrypt them into the models/ folder before loading.
    Safe to call even if encryption is not set up — it simply does nothing.
    """
    key_str = os.environ.get("LUNGAI_SECRET_KEY")
    if not key_str:
        return  # no key set → skip decryption, load normally

    try:
        from cryptography.fernet import Fernet
        fernet = Fernet(key_str.encode())
    except Exception as e:
        print(f"[LungAI] Encryption library error: {e}")
        return

    enc_files = [
        ("models/cxr_masked_model.keras.enc",  "models/cxr_masked_model.keras"),
        ("models/ct_model.keras.enc",           "models/ct_model.keras"),
        ("models/unet_lung_seg.hdf5.enc",       "models/unet_lung_seg.hdf5"),
    ]

    for enc_path, out_path in enc_files:
        if not os.path.exists(enc_path):
            continue   # encrypted file not present → skip
        if os.path.exists(out_path):
            continue   # already decrypted this session → skip
        try:
            with open(enc_path, "rb") as f:
                decrypted = fernet.decrypt(f.read())
            with open(out_path, "wb") as f:
                f.write(decrypted)
            print(f"[LungAI] Decrypted: {out_path}")
        except Exception as e:
            print(f"[LungAI] Failed to decrypt {enc_path}: {e}")

# --------------------------------------------------------------------------
# Configuration  (matches thesis exactly)
# --------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

CXR_MODEL_PATH  = os.path.join(MODELS_DIR, "cxr_masked_model.keras")
CT_MODEL_PATH   = os.path.join(MODELS_DIR, "ct_model.keras")
UNET_MODEL_PATH = os.path.join(MODELS_DIR, "unet_lung_seg.hdf5")

CXR_CLASSES = ["COVID", "Lung_Opacity", "NORMAL", "Viral Pneumonia"]
CT_CLASSES  = ["Benign", "Malignant", "Normal"]

TAU_COVID     = 0.45            # Chapter 3 §3.4.3
TAU_LO        = 0.40
TAU_NORMAL_CT = 0.35            # Chapter 3 §3.5.4

# Grad-CAM target layer — matches friend's CT script.
GRADCAM_LAYER = "conv5_block16_concat"

IMG_SIZE     = (224, 224)
UNET_INPUT   = (512, 512)        # confirmed by model.input_shape inspection

# Region-detection tuning
PEAK_MIN_DISTANCE   = 6
PEAK_THRESHOLD_REL  = 0.60
CLUSTER_EPS         = 12
MAX_REGIONS         = 5
SMOOTHING_SIGMA     = 0.6
SHARPNESS_FACTOR    = 2.0
BOX_HALF_SIZE_PX    = 12


# --------------------------------------------------------------------------
# Model loading (lazy, cached)
# --------------------------------------------------------------------------
_models = {"cxr": None, "ct": None, "unet": None, "loaded": False}


def _focal_loss(gamma=2.0, alpha=0.25):
    """Custom loss for deserialising the trained .keras files."""
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(weight * ce)
    return loss


def load_models():
    if _models["loaded"]:
        return _models
    _models["loaded"] = True

    if not TF_AVAILABLE:
        print("[LungAI] TensorFlow not available -> DEMO mode")
        return _models

    custom = {"loss": _focal_loss()}
    for key, path in [("cxr",  CXR_MODEL_PATH),
                      ("ct",   CT_MODEL_PATH),
                      ("unet", UNET_MODEL_PATH)]:
        if not os.path.exists(path):
            continue
        try:
            _models[key] = tf.keras.models.load_model(
                path,
                custom_objects=custom if key != "unet" else None,
                compile=False,
            )

            print(f"[LungAI] Loaded {key}: {path}")
        except Exception as e:
            print(f"[LungAI] Failed to load {key} ({e})")

    return _models


def is_demo_mode(modality):
    load_models()
    return _models.get(modality) is None


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def preprocess_image(pil_img):
    """224x224 RGB float in [0,1]."""
    return np.asarray(
        pil_img.convert("RGB").resize(IMG_SIZE)
    ).astype(np.float32) / 255.0


def preprocess_for_unet(pil_img):
    """512x512 grayscale, single channel, float in [0,1]."""
    arr = np.asarray(
        pil_img.convert("L").resize(UNET_INPUT)
    ).astype(np.float32) / 255.0
    return arr[..., np.newaxis]


# --------------------------------------------------------------------------
# Lung segmentation
# --------------------------------------------------------------------------
def segment_lungs_cxr(pil_img):
    """U-Net mask if available; geometric ellipse fallback otherwise.
    Returns a (224, 224, 1) float mask."""
    load_models()
    if _models["unet"] is not None:
        inp = np.expand_dims(preprocess_for_unet(pil_img), axis=0)
        pred = _models["unet"].predict(inp, verbose=0)[0]
        mask512 = (pred > 0.5).astype(np.float32)
        if mask512.ndim == 2:
            mask512 = mask512[..., np.newaxis]
        # Resize 512 -> 224 with TF for bilinear
        mask = tf.image.resize(mask512, IMG_SIZE).numpy()
        mask = (mask > 0.5).astype(np.float32)
        return mask
    return _geometric_lung_mask()


def _geometric_lung_mask():
    """Two-ellipse fallback / CT mask. Not anatomically precise; only
    constrains attention to a plausible thoracic region."""
    h, w = IMG_SIZE
    yy, xx = np.ogrid[:h, :w]
    cy = h / 2
    left = (((xx - w * 0.32) / (w * 0.18)) ** 2 +
            ((yy - cy)        / (h * 0.34)) ** 2) <= 1
    right = (((xx - w * 0.68) / (w * 0.18)) ** 2 +
             ((yy - cy)        / (h * 0.34)) ** 2) <= 1
    mask = (left | right).astype(np.float32)
    return mask[..., np.newaxis]


# --------------------------------------------------------------------------
# Grad-CAM (standard)  --  Chapter 3 §3.6
# --------------------------------------------------------------------------
def compute_gradcam(model, img_array, class_idx, layer_name=GRADCAM_LAYER):
    """Vanilla Grad-CAM. Used for the HEATMAP view."""
    # Disable softmax safely for Functional models
    if hasattr(model.layers[-1], "activation"):
        model.layers[-1].activation = tf.keras.activations.linear
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output])

    inp = np.expand_dims(img_array, axis=0)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp)
        class_channel = preds[:, class_idx]

    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    heatmap = tf.cond(max_val > 0,
                      lambda: heatmap / max_val,
                      lambda: heatmap)
    return heatmap.numpy()


# --------------------------------------------------------------------------
# Grad-CAM++  --  feeds the REGIONS view
# --------------------------------------------------------------------------
def compute_gradcam_pp(model, img_array, class_idx, layer_name=GRADCAM_LAYER):
    """Grad-CAM++ matching the friend's CT script.
    Temporarily swaps softmax -> linear on the last layer so gradients
    don't saturate, then restores it. The probabilities returned by
    model.predict() elsewhere are untouched."""

    # Save and strip softmax just for this Grad-CAM++ pass
    last_layer = model.layers[-1]

    original_activation = None

    if hasattr(last_layer, "activation"):
        original_activation = last_layer.activation
        last_layer.activation = tf.keras.activations.linear

    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )

        inp = np.expand_dims(img_array, axis=0)
        img_tf = tf.convert_to_tensor(inp, tf.float32)

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_tf)
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return np.zeros((conv_out.shape[1], conv_out.shape[2]), dtype=np.float32)

        first = grads
        second = grads ** 2
        third = grads ** 3

        global_sum = tf.reduce_sum(conv_out, axis=(1, 2), keepdims=True)
        alpha_den = 2 * second + third * global_sum + 1e-7
        alpha = second / alpha_den

        weights = tf.reduce_sum(alpha * tf.nn.relu(first), axis=(1, 2))
        cam = tf.reduce_sum(weights[:, None, None, :] * conv_out, axis=-1)[0]
        cam = tf.nn.relu(cam).numpy()
        return cam
    finally:
        # Restore only if the layer actually has an activation
        if hasattr(last_layer, "activation") and original_activation is not None:
            last_layer.activation = original_activation


# --------------------------------------------------------------------------
# Peak detection + clustering + splatter generation
# (adapted from friend's CT analysis script)
# --------------------------------------------------------------------------
def _norm01(x):
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    rng = x.max() - x.min()
    return (x - x.min()) / (rng + 1e-8) if rng > 1e-6 else x


def detect_regions(heatmap, lung_mask=None):
    """Returns (peak_coords, peak_values) after smoothing, lung-masking,
    local-maxima detection, and DBSCAN clustering. Matches friend's
    detect_precise_peaks + adaptive_clustering."""
    if not REGIONS_AVAILABLE:
        return np.empty((0, 2), int), np.empty((0,), float)

    if lung_mask is not None:
        m2 = lung_mask[..., 0] if lung_mask.ndim == 3 else lung_mask
        hm = heatmap * m2
    else:
        hm = heatmap

    hm_smooth = gaussian_filter(hm, sigma=1.5)

    peaks = peak_local_max(
        hm_smooth,
        min_distance=PEAK_MIN_DISTANCE,
        threshold_rel=PEAK_THRESHOLD_REL,
        exclude_border=5,
        num_peaks=10,
    )

    if len(peaks) == 0:
        # Friend's fallback: take top-5 brightest pixels
        flat_idx = np.argsort(hm_smooth.flatten())[-5:]
        peaks = np.array(np.unravel_index(flat_idx, hm_smooth.shape)).T

    if len(peaks) == 0:
        return np.empty((0, 2), int), np.empty((0,), float)

    values = np.array([hm_smooth[y, x] for y, x in peaks])
    order = np.argsort(values)[::-1]
    peaks = peaks[order]
    values = values[order]

    if len(peaks) > 5:
        peaks = peaks[:5]
        values = values[:5]

    # DBSCAN to merge nearby peaks
    if len(peaks) > 1:
        clustering = DBSCAN(eps=CLUSTER_EPS, min_samples=1).fit(peaks)
        merged_peaks, merged_vals = [], []
        for lbl in set(clustering.labels_):
            mask = clustering.labels_ == lbl
            cluster_peaks = peaks[mask]
            cluster_vals = values[mask]
            best = np.argmax(cluster_vals)
            merged_peaks.append(cluster_peaks[best])
            merged_vals.append(cluster_vals[best])
        peaks = np.array(merged_peaks)
        values = np.array(merged_vals)
        order = np.argsort(values)[::-1]
        peaks = peaks[order]
        values = values[order]

    if len(peaks) > MAX_REGIONS:
        peaks = peaks[:MAX_REGIONS]
        values = values[:MAX_REGIONS]
    return peaks, values


def generate_splatters(shape, peak_coords, peak_values,
                       lung_mask=None, min_size=6, max_size=12):
    """Sparse Gaussian-blob heatmap at peak locations. Matches friend's
    generate_precise_splatters."""
    splatter = np.zeros(shape, dtype=np.float32)
    if len(peak_coords) == 0:
        return splatter

    if len(peak_values) > 1:
        v_norm = ((peak_values - peak_values.min()) /
                  (peak_values.max() - peak_values.min() + 1e-8))
    else:
        v_norm = np.array([1.0])

    for (y, x), v_n, v_raw in zip(peak_coords, v_norm, peak_values):
        y_int, x_int = int(round(y)), int(round(x))
        size = int(min_size + (max_size - min_size) * v_n)
        size = max(min_size, min(max_size, size))
        sigma = size / 2.5
        yy, xx = np.ogrid[-size:size + 1, -size:size + 1]
        blob = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        blob = (blob ** SHARPNESS_FACTOR) * v_raw

        y0, y1 = max(0, y_int - size), min(shape[0], y_int + size + 1)
        x0, x1 = max(0, x_int - size), min(shape[1], x_int + size + 1)
        b_y0 = size - (y_int - y0) if y_int - size < 0 else 0
        b_y1 = b_y0 + (y1 - y0)
        b_x0 = size - (x_int - x0) if x_int - size < 0 else 0
        b_x1 = b_x0 + (x1 - x0)
        if b_y1 > b_y0 and b_x1 > b_x0:
            splatter[y0:y1, x0:x1] = np.maximum(
                splatter[y0:y1, x0:x1], blob[b_y0:b_y1, b_x0:b_x1])

    if lung_mask is not None:
        m2 = lung_mask[..., 0] if lung_mask.ndim == 3 else lung_mask
        splatter = splatter * m2
    return _norm01(splatter)


# --------------------------------------------------------------------------
# Image rendering
# --------------------------------------------------------------------------
def _jet(v):
    """Vector jet colormap, v in [0,1] -> (..., 3) uint8."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype("uint8")


def overlay_heatmap(original_pil, heatmap, alpha=0.45):
    """Smooth heatmap blended onto the ORIGINAL image."""
    base = np.asarray(
        original_pil.convert("RGB").resize(IMG_SIZE)).astype(np.float32)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    hm_pil = Image.fromarray(
        (heatmap * 255).astype("uint8")).resize(IMG_SIZE, Image.BILINEAR)
    hm_arr = np.asarray(hm_pil).astype(np.float32) / 255.0

    colored = _jet(hm_arr).astype(np.float32)
    blended = (colored * alpha + base * (1 - alpha)).astype("uint8")
    return Image.fromarray(blended)


def render_boxes(original_pil, peak_coords, peak_values):
    """Draw labelled bounding boxes on the original image. Pure PIL."""
    print(f"[LungAI DEBUG] render_boxes: {len(peak_coords)} peaks, "
          f"values = {peak_values}")
    from PIL import ImageDraw, ImageFont
    base = original_pil.convert("RGB").resize(IMG_SIZE).copy()
    draw = ImageDraw.Draw(base)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    if len(peak_values) > 0:
        v_norm = ((peak_values - peak_values.min()) /
                  (peak_values.max() - peak_values.min() + 1e-8)) \
            if len(peak_values) > 1 else np.array([1.0])
    else:
        v_norm = []

    used_label_areas = []

    for (y, x), v_raw, v_n in zip(peak_coords, peak_values, v_norm):
        print(f"[LungAI DEBUG] box: x={int(x)}, y={int(y)}, "
              f"conf={int(v_raw * 100)}%")
        conf = int(np.clip(v_raw * 100, 0, 100))
        if   conf > 75: outline = "#e63946"   # red
        elif conf > 55: outline = "#f1a208"   # orange
        else:           outline = "#f4d35e"   # yellow

        h = BOX_HALF_SIZE_PX
        x0, y0 = max(0, int(x) - h), max(0, int(y) - h)
        x1, y1 = min(IMG_SIZE[0], int(x) + h), min(IMG_SIZE[1], int(y) + h)

        # Box (drawn twice for thicker outline)
        for off in (0, 1):
            draw.rectangle(
                [x0 - off, y0 - off, x1 + off, y1 + off], outline=outline)

        # Label — try above the box first, fall back to below/right
        label = f"{conf}%"
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]

        candidates = [
            (x0,            y0 - th - 4, x0 + tw + 6, y0),              # above
            (x0,            y1,          x0 + tw + 6, y1 + th + 4),     # below
            (x1 + 2,        y0,          x1 + tw + 8, y0 + th + 4),     # right
        ]
        chosen = None
        for cand in candidates:
            cx0, cy0, cx1, cy1 = cand
            if cy0 < 0 or cy1 > IMG_SIZE[1] or cx1 > IMG_SIZE[0]:
                continue
            overlap = any(
                not (cx1 < u[0] or cx0 > u[2] or cy1 < u[1] or cy0 > u[3])
                for u in used_label_areas
            )
            if not overlap:
                chosen = cand
                break
        if chosen is None:
            chosen = candidates[0]

        cx0, cy0, cx1, cy1 = chosen
        draw.rectangle([cx0, cy0, cx1, cy1], fill=outline)
        draw.text((cx0 + 3, cy0 + 1), label, fill="black", font=font)
        used_label_areas.append(chosen)

    return base


# --------------------------------------------------------------------------
# Decision rules
# --------------------------------------------------------------------------
def apply_cxr_threshold_rule(probs):
    p_covid = probs[CXR_CLASSES.index("COVID")]
    p_lo    = probs[CXR_CLASSES.index("Lung_Opacity")]
    if p_covid >= TAU_COVID:
        return CXR_CLASSES.index("COVID"), \
               f"COVID threshold (tau={TAU_COVID})"
    if p_lo >= TAU_LO:
        return CXR_CLASSES.index("Lung_Opacity"), \
               f"Lung Opacity threshold (tau={TAU_LO})"
    return int(np.argmax(probs)), None


def apply_ct_threshold_rule(probs):
    p_normal = probs[CT_CLASSES.index("Normal")]
    if p_normal >= TAU_NORMAL_CT:
        return CT_CLASSES.index("Normal"), \
               f"Normal threshold (tau={TAU_NORMAL_CT})"
    return int(np.argmax(probs)), None


# --------------------------------------------------------------------------
# Demo helpers (only used when real models are absent)
# --------------------------------------------------------------------------
def _demo_probs(img_array, n_classes):
    seed = int((img_array.mean() * 1e6)) % (2 ** 31)
    rng = np.random.RandomState(seed)
    logits = rng.rand(n_classes) * 2.0
    logits[rng.randint(n_classes)] += 3.0
    e = np.exp(logits - logits.max())
    return (e / e.sum()).astype(np.float32)


def _demo_heatmap(class_idx, n_classes):
    h = w = 7
    rng = np.random.RandomState(class_idx + 1)
    base = rng.rand(h, w) * 0.3
    cy, cx = (2 + class_idx) % h, (3 + 2 * class_idx) % w
    yy, xx = np.ogrid[:h, :w]
    blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / 2.0))
    hm = base + blob
    return (hm / hm.max()).astype(np.float32)


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------
def _pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --------------------------------------------------------------------------
# Top-level prediction
# --------------------------------------------------------------------------
def _build_views(original_pil, target_pil_for_overlay,
                 heatmap_lowres, gradcampp_lowres, lung_mask_224):
    """Produce both the HEATMAP overlay and the REGIONS overlay."""
    print(f"[LungAI DEBUG] _build_views CALLED. "
          f"gradcampp is None? {gradcampp_lowres is None}. "
          f"REGIONS_AVAILABLE? {REGIONS_AVAILABLE}")
    if gradcampp_lowres is not None:
        print(f"[LungAI DEBUG] gradcampp_lowres shape={gradcampp_lowres.shape}, "
              f"min={gradcampp_lowres.min():.4f}, max={gradcampp_lowres.max():.4f}")
    if gradcampp_lowres is not None and REGIONS_AVAILABLE:
        # Upsample Grad-CAM++ to 224 (friend uses cv2.resize, we use PIL)
        gpp_pil = Image.fromarray(
            (_norm01(gradcampp_lowres) * 255).astype("uint8")
        ).resize(IMG_SIZE, Image.BILINEAR)
        gpp = np.asarray(gpp_pil).astype(np.float32) / 255.0
        gpp = gaussian_filter(gpp, sigma=SMOOTHING_SIGMA)
        gpp = _norm01(gpp)

        print(f"[LungAI DEBUG] Grad-CAM++ map: min={gpp.min():.3f}, "
              f"max={gpp.max():.3f}, mean={gpp.mean():.3f}")

        peaks, values = detect_regions(gpp, lung_mask=lung_mask_224)
        splatter = generate_splatters(
            IMG_SIZE, peaks, values, lung_mask=lung_mask_224)
        boxes_img = render_boxes(target_pil_for_overlay, peaks, values)
        regions_heat_img = overlay_heatmap(
            target_pil_for_overlay, splatter, alpha=0.45)
        n_regions = int(len(peaks))
    else:
        boxes_img = target_pil_for_overlay.convert("RGB").resize(IMG_SIZE)
        regions_heat_img = boxes_img
        n_regions = 0

    heatmap_img = overlay_heatmap(
        target_pil_for_overlay, heatmap_lowres, alpha=0.45)

    return {
        "heatmap_b64":      _pil_to_b64(heatmap_img),
        "boxes_b64":        _pil_to_b64(boxes_img),
        "regions_heat_b64": _pil_to_b64(regions_heat_img),
        "n_regions":        n_regions,
    }


def predict_cxr(pil_img):
    load_models()
    original = pil_img.convert("RGB")
    img_array = preprocess_image(original)

    lung_mask = segment_lungs_cxr(original)        # (224,224,1)
    masked    = img_array * lung_mask

    demo = _models["cxr"] is None
    if demo:
        probs = _demo_probs(masked, len(CXR_CLASSES))
    else:
        probs = _models["cxr"].predict(
            np.expand_dims(masked, 0), verbose=0)[0]

    pred_idx, rule = apply_cxr_threshold_rule(probs)

    if demo:
        gc  = _demo_heatmap(pred_idx, len(CXR_CLASSES))
        gpp = gc
    else:
        CXR_GRADCAM_LAYER = "conv5_block16_2_conv"  # thesis §3.6.1

        gc = compute_gradcam(_models["cxr"], masked, pred_idx, layer_name=CXR_GRADCAM_LAYER)
        gpp = compute_gradcam_pp(_models["cxr"], masked, pred_idx, layer_name=CXR_GRADCAM_LAYER)

    views = _build_views(
        original_pil=original,
        target_pil_for_overlay=original,
        heatmap_lowres=gc,
        gradcampp_lowres=gpp,
        lung_mask_224=lung_mask,
    )

    return {
        "modality": "CXR",
        "demo": demo,
        "prediction": CXR_CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities":
            {c: float(p) for c, p in zip(CXR_CLASSES, probs)},
        "rule_triggered": rule,
        "original_b64": _pil_to_b64(original.resize(IMG_SIZE)),
        **views,
    }


def predict_ct(pil_img):
    load_models()
    original = pil_img.convert("RGB")
    img_array = preprocess_image(original)

    # CT has no segmentation model; use the geometric mask just to
    # constrain peak detection to a thoracic region.
    lung_mask = _geometric_lung_mask()

    demo = _models["ct"] is None
    if demo:
        probs = _demo_probs(img_array, len(CT_CLASSES))
    else:
        probs = _models["ct"].predict(
            np.expand_dims(img_array, 0), verbose=0)[0]

    pred_idx, rule = apply_ct_threshold_rule(probs)

    if demo:
        gc  = _demo_heatmap(pred_idx, len(CT_CLASSES))
        gpp = gc
    else:
        gc  = compute_gradcam   (_models["ct"], img_array, pred_idx)
        gpp = compute_gradcam_pp(_models["ct"], img_array, pred_idx)

    views = _build_views(
        original_pil=original,
        target_pil_for_overlay=original,
        heatmap_lowres=gc,
        gradcampp_lowres=gpp,
        lung_mask_224=lung_mask,
    )

    return {
        "modality": "CT",
        "demo": demo,
        "prediction": CT_CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities":
            {c: float(p) for c, p in zip(CT_CLASSES, probs)},
        "rule_triggered": rule,
        "original_b64": _pil_to_b64(original.resize(IMG_SIZE)),
        **views,
    }