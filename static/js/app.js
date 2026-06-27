/* ============================================================
   LungAI Diagnostics — frontend logic (v2)
   Now supports two explanation views:
     - HEATMAP  : continuous Grad-CAM overlay (existing)
     - REGIONS  : peak-detected bounding boxes (new, from CT pipeline)
   Session history is in-memory only (privacy posture).
   ============================================================ */

const CLASS_COLORS = {
  "COVID":            "#e8635f",
  "Lung_Opacity":     "#5b8fd1",
  "NORMAL":           "#46b98f",
  "Viral Pneumonia":  "#d6a94e",
  "Benign":           "#5b8fd1",
  "Malignant":        "#e8635f",
  "Normal":           "#46b98f",
};

let currentModality = "CXR";
let currentView     = "heatmap";          // or "regions"
let currentResult   = null;
let selectedFile    = null;
const history       = [];

// ---------- Element refs ----------
const fileInput   = document.getElementById("fileInput");
const dropzone    = document.getElementById("dropzone");
const dzInner     = document.getElementById("dropzoneInner");
const previewImg  = document.getElementById("previewImg");
const analyzeBtn  = document.getElementById("analyzeBtn");
const btnLabel    = analyzeBtn.querySelector(".btn-label");
const btnSpinner  = analyzeBtn.querySelector(".btn-spinner");

const resultEmpty = document.getElementById("resultEmpty");
const resultBody  = document.getElementById("resultBody");
const verdictClass= document.getElementById("verdictClass");
const confFill    = document.getElementById("confFill");
const confPct     = document.getElementById("confPct");
const ruleFlag    = document.getElementById("ruleFlag");
const ruleText    = document.getElementById("ruleText");
const probList    = document.getElementById("probList");

const camEmpty    = document.getElementById("camEmpty");
const camBody     = document.getElementById("camBody");
const camOriginal = document.getElementById("camOriginal");
const camOverlay  = document.getElementById("camOverlay");
const opacitySlider = document.getElementById("opacitySlider");
const opacityLabel  = document.getElementById("opacityLabel");
const opacityRow    = document.getElementById("opacityRow");
const camLegend     = document.getElementById("camLegend");
const regionLegend  = document.getElementById("regionLegend");
const regionCount   = document.getElementById("regionCount");

const historyTrack = document.getElementById("historyTrack");
const historyEmpty = document.getElementById("historyEmpty");
const clearHistory = document.getElementById("clearHistory");

const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");

// ---------- Health check ----------
fetch("/health").then(r => r.json()).then(d => {
  const demo = (currentModality === "CT") ? d.ct_demo : d.cxr_demo;
  updateStatus(demo);
  window._health = d;
}).catch(() => {
  statusText.textContent = "offline";
});

function updateStatus(demo) {
  statusPill.classList.remove("live", "demo");
  if (demo) {
    statusPill.classList.add("demo");
    statusText.textContent = "demo mode";
  } else {
    statusPill.classList.add("live");
    statusText.textContent = "models loaded";
  }
}

// ---------- Modality toggle ----------
document.querySelectorAll(".mod-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mod-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentModality = btn.dataset.modality;
    if (window._health) {
      updateStatus(currentModality === "CT"
        ? window._health.ct_demo : window._health.cxr_demo);
    }
  });
});

// ---------- View toggle (heatmap | regions) ----------
document.querySelectorAll(".view-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".view-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentView = btn.dataset.view;
    if (currentResult) applyView(currentResult);
  });
});

function applyView(d) {
  // The original (un-overlayed) is the same in both views.
  camOriginal.src = "data:image/png;base64," + d.original_b64;

  if (currentView === "regions") {
    // Show bounding boxes (already drawn on the original by the backend).
    camOverlay.src = "data:image/png;base64," + (d.boxes_b64 || d.heatmap_b64);
    // Boxes ARE the original image with rectangles painted on, so the
    // opacity slider would just fade them out — hide it in this view.
    opacityRow.hidden    = true;
    camLegend.hidden     = true;
    regionLegend.hidden  = false;
    if (typeof d.n_regions === "number") {
      regionCount.textContent = d.n_regions;
      regionCount.hidden = false;
    }
  } else {
    // Heatmap view
    camOverlay.src = "data:image/png;base64," + d.heatmap_b64;
    opacityRow.hidden    = false;
    camLegend.hidden     = false;
    regionLegend.hidden  = true;
    regionCount.hidden   = (typeof d.n_regions !== "number");
    if (typeof d.n_regions === "number") {
      regionCount.textContent = d.n_regions;
    }
    opacitySlider.value = 100;
    camOverlay.style.opacity = 1;
  }
}

// ---------- File selection ----------
fileInput.addEventListener("change", e => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

["dragover", "dragenter"].forEach(ev =>
  dropzone.addEventListener(ev, e => {
    e.preventDefault(); dropzone.classList.add("drag");
  }));
["dragleave", "drop"].forEach(ev =>
  dropzone.addEventListener(ev, e => {
    e.preventDefault(); dropzone.classList.remove("drag");
  }));
dropzone.addEventListener("drop", e => {
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    previewImg.src = ev.target.result;
    previewImg.hidden = false;
    dzInner.style.display = "none";
  };
  reader.readAsDataURL(file);
  analyzeBtn.disabled = false;
}

// ---------- Analyze ----------
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  setLoading(true);

  const fd = new FormData();
  fd.append("image", selectedFile);
  fd.append("modality", currentModality);

  try {
    const res = await fetch("/predict", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    renderResult(data);
    updateStatus(data.demo);
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    setLoading(false);
  }
});

function setLoading(on) {
  analyzeBtn.disabled = on;
  btnLabel.textContent = on ? "Analyzing…" : "Analyze";
  btnSpinner.hidden = !on;
}

// ---------- Render result ----------
function renderResult(d) {
  currentResult = d;

  resultEmpty.hidden = true;
  resultBody.hidden = false;
  resultBody.classList.remove("fade-in");
  void resultBody.offsetWidth;
  resultBody.classList.add("fade-in");

  const color = CLASS_COLORS[d.prediction] || "#3fd0c9";
  verdictClass.textContent = d.prediction.replace(/_/g, " ");
  verdictClass.style.color = color;

  const pct = Math.round(d.confidence * 100);
  confPct.textContent = pct + "%";
  confFill.style.width = pct + "%";

  if (d.rule_triggered) {
    ruleFlag.hidden = false;
    ruleText.textContent = "Decision via " + d.rule_triggered;
  } else {
    ruleFlag.hidden = true;
  }

  // Probability bars
  probList.innerHTML = "";
  const entries = Object.entries(d.probabilities).sort((a, b) => b[1] - a[1]);
  for (const [cls, p] of entries) {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-name">${cls.replace(/_/g, " ")}</span>
      <div class="prob-track">
        <div class="prob-bar" style="background:${CLASS_COLORS[cls] || '#3fd0c9'}"></div>
      </div>
      <span class="prob-val">${(p * 100).toFixed(1)}%</span>`;
    probList.appendChild(row);
    requestAnimationFrame(() => {
      row.querySelector(".prob-bar").style.width = (p * 100) + "%";
    });
  }

  // Explanation panel
  camEmpty.hidden = true;
  camBody.hidden = false;
  applyView(d);

  addHistory(d);
}

// ---------- Grad-CAM opacity slider ----------
opacitySlider.addEventListener("input", e => {
  camOverlay.style.opacity = e.target.value / 100;
});

// ---------- Session history ----------
function addHistory(d) {
  history.unshift({
    cls: d.prediction,
    conf: d.confidence,
    modality: d.modality,
    thumb: d.boxes_b64 || d.heatmap_b64,
    time: new Date().toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}),
    full: d,
  });
  renderHistory();
}

function renderHistory() {
  historyEmpty.hidden = history.length > 0;
  clearHistory.hidden = history.length === 0;

  historyTrack.querySelectorAll(".hist-card").forEach(c => c.remove());

  history.forEach(h => {
    const card = document.createElement("div");
    card.className = "hist-card fade-in";
    card.innerHTML = `
      <img class="hist-thumb" src="data:image/png;base64,${h.thumb}" alt="">
      <div class="hist-meta">
        <span class="hist-class" style="color:${CLASS_COLORS[h.cls] || '#3fd0c9'}">
          ${h.cls.replace(/_/g, " ")}</span>
        <span class="hist-sub">${h.modality} · ${(h.conf*100).toFixed(0)}% · ${h.time}</span>
      </div>`;
    card.addEventListener("click", () => renderResult(h.full));
    historyTrack.appendChild(card);
  });
}

clearHistory.addEventListener("click", () => {
  history.length = 0;
  renderHistory();
});
