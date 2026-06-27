# LungAI Diagnostics — Step-by-Step Setup Guide (Windows)

This guide assumes you have **never run a web app before**. Follow it
exactly. Every command goes into a black window called **Command
Prompt**. You will keep that black window OPEN the whole time the app
runs.

---

## Part 1 — Get the files onto your computer

1. Unzip `lungai_webapp.zip`.
2. Put the resulting `lungai_webapp` folder somewhere simple, e.g.
   `C:\lungai_webapp`.

---

## Part 2 — Open Command Prompt and go to the folder

1. Press **Windows key**, type `cmd`, press Enter.
2. Type:
   ```
   cd C:\lungai_webapp
   ```
3. The prompt should now say `C:\lungai_webapp>`.

---

## Part 3 — Install the requirements (ONE TIME)

```
pip install -r requirements.txt
```

This installs Flask, Pillow, numpy, scipy, scikit-image, and
scikit-learn. Takes 2–4 minutes the first time. You only do this
ONCE per computer.

If you see "pip is not recognized", try:
```
python -m pip install -r requirements.txt
```

---

## Part 4 — Run the app

```
python app.py
```

You'll see output like:
```
LungAI Diagnostics - starting up
Open  http://127.0.0.1:5000  in your browser.
```

**Leave this black window OPEN.** The app is running.

---

## Part 5 — Use the app

Open Chrome or Edge. Type in the address bar:
```
http://127.0.0.1:5000
```

The interface appears. Upload an X-ray or CT, click Analyze, and
review the result. Use the **Heatmap | Detected regions** toggle in
the Explanation panel to switch between the two visualization styles.

To STOP the app: go back to the black window and press **Ctrl + C**,
or just close the window.

---

## Part 6 — REAL MODE (your trained models)

Do this at the lab where your `.keras` files are.

1. Install TensorFlow (one time, big download):
   ```
   pip install tensorflow==2.10.0
   ```

2. Copy your trained files into `C:\lungai_webapp\models\` and
   rename them to exactly these names:

   - `cxr_masked_model.keras`     ← `best_model_phase3_fixed.keras`
   - `ct_model.keras`             ← your CT model
   - `unet_lung_seg.hdf5`         ← pre-trained Montgomery+Shenzhen U-Net

3. Run again:
   ```
   python app.py
   ```

You should see:
```
[LungAI] Loaded cxr: ...
[LungAI] Loaded ct: ...
[LungAI] Loaded unet: ...
```

The status pill in the top-right turns GREEN. Predictions are now real.

---

## Troubleshooting

**"python is not recognized"** — use the full path to your Python 3.10:
`C:\Users\YOU\PycharmProjects\PFE\venv_tf\Scripts\python.exe app.py`

**"pip is not recognized"** — use `python -m pip install ...`.

**Browser says "can't reach this page"** — the app isn't running. Check
the black terminal window for errors.

**Predictions look weird (Grad-CAM noisy or wrong)** — check the black
window when starting up. You should see three "Loaded" lines. If a
model didn't load, the app falls back to a placeholder.

**Port 5000 in use** — edit `app.py`, change `port=5000` to `port=5050`,
save, restart, use `http://127.0.0.1:5050`.

---

## Cheat sheet (after first-time setup)

```
1. Open Command Prompt (Windows key, type cmd)
2. cd C:\lungai_webapp
3. python app.py
4. Open browser at http://127.0.0.1:5000
5. To stop: Ctrl + C
```
