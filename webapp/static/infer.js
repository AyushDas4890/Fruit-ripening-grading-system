// In-browser inference for the static (serverless) build. Returns the SAME JSON shape as
// POST /api/predict on webapp/server.py, so app.js does not care which one answers.
// Models: fold-4 ordinal (+ CAM computed inside the graph) and fold-4 softmax, ONNX format.
(function () {
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  ort.env.wasm.numThreads = 1; // static hosts may not send cross-origin isolation headers

  let sessions = null;
  async function fetchWithProgress(url, label, onp) {
    const res = await fetch(url);
    const total = +res.headers.get("content-length") || 0;
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      onp(`${label} ${total ? Math.round((got / total) * 100) + "%" : (got / 1e6).toFixed(0) + " MB"}`);
    }
    const buf = new Uint8Array(got);
    let o = 0;
    for (const c of chunks) { buf.set(c, o); o += c.length; }
    return buf;
  }
  async function load(onp) {
    if (sessions) return sessions;
    const a = await fetchWithProgress("models/ordinal.onnx", "loading ordinal model", onp);
    const b = await fetchWithProgress("models/softmax.onnx", "loading softmax model", onp);
    onp("starting models…");
    sessions = {
      ord: await ort.InferenceSession.create(a, { executionProviders: ["wasm"] }),
      soft: await ort.InferenceSession.create(b, { executionProviders: ["wasm"] }),
    };
    return sessions;
  }

  // like server.py: shrink to <= 512 px, then stretch to 224 x 224 (torchvision Resize((224,224)))
  async function toTensor(blob) {
    const bmp = await createImageBitmap(blob, { imageOrientation: "from-image" });
    let w = bmp.width, h = bmp.height, src = bmp;
    const s = Math.min(1, 512 / Math.max(w, h));
    if (s < 1) {
      // step down in halves for a smooth (antialiased) result on big phone photos
      let cw = w, ch = h, cur = bmp;
      while (cw * 0.5 > w * s) {
        const c = new OffscreenCanvas(Math.round(cw * 0.5), Math.round(ch * 0.5));
        const g = c.getContext("2d"); g.imageSmoothingQuality = "high"; g.drawImage(cur, 0, 0, c.width, c.height);
        cur = c; cw = c.width; ch = c.height;
      }
      const c = new OffscreenCanvas(Math.round(w * s), Math.round(h * s));
      const g = c.getContext("2d"); g.imageSmoothingQuality = "high"; g.drawImage(cur, 0, 0, c.width, c.height);
      src = c; w = c.width; h = c.height;
    }
    const c = new OffscreenCanvas(224, 224);
    const g = c.getContext("2d");
    g.imageSmoothingQuality = "high";
    g.drawImage(src, 0, 0, 224, 224);
    const px = g.getImageData(0, 0, 224, 224).data;
    const t = new Float32Array(3 * 224 * 224);
    for (let i = 0; i < 224 * 224; i++) {
      for (let k = 0; k < 3; k++) t[k * 50176 + i] = (px[i * 4 + k] / 255 - MEAN[k]) / STD[k];
    }
    return { tensor: new ort.Tensor("float32", t, [1, 3, 224, 224]), rgba: px };
  }

  // same colour blend as server.py heat_png: periwinkle -> magenta, 55% at full heat
  function heatPng(rgba, cam) {
    const c = document.createElement("canvas");
    c.width = c.height = 224;
    const g = c.getContext("2d");
    const img = g.createImageData(224, 224);
    const warm = [0.93, 0.08, 0.60], cool = [0.64, 0.73, 0.98];
    for (let i = 0; i < 224 * 224; i++) {
      const h = cam[i];
      for (let k = 0; k < 3; k++) {
        const base = rgba[i * 4 + k] / 255, col = cool[k] * (1 - h) + warm[k] * h;
        img.data[i * 4 + k] = Math.round(Math.min(1, Math.max(0, base * (1 - 0.55 * h) + col * 0.55 * h)) * 255);
      }
      img.data[i * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    return c.toDataURL("image/png");
  }

  const sig = (z) => 1 / (1 + Math.exp(-z));
  window.localPredict = async function (blob, sample, onp) {
    const S = await load(onp);
    onp("grading…");
    const { tensor, rgba } = await toTensor(blob);
    const o = await S.ord.run({ input: tensor });
    const p_gt = Array.from(o.logits.data).map(sig);
    const cum = [1, ...p_gt, 0];
    const probs = cum.slice(0, 4).map((v, k) => Math.max(0, v - cum[k + 1]));
    const sl = Array.from((await S.soft.run({ input: tensor })).logits.data);
    const mx = Math.max(...sl), ex = sl.map((v) => Math.exp(v - mx)), z = ex.reduce((a, b) => a + b, 0);
    const sp = ex.map((v) => v / z);
    const r4 = (v) => Math.round(v * 1e4) / 1e4;
    return {
      used: sample ? "fold-4 model (never saw this photo) · runs in your browser" : "fold-4 model · runs in your browser",
      tl_ordinal: { probs: probs.map(r4), p_gt: p_gt.map(r4), grade: p_gt.filter((p) => p > 0.5).length,
        expected: Math.round(p_gt.reduce((a, b) => a + b, 0) * 1e3) / 1e3 },
      tl_softmax: { probs: sp.map(r4), grade: sp.indexOf(Math.max(...sp)) },
      gradcam: heatPng(rgba, o.cam.data),
      grades: ["unripe", "partially ripe", "ripe", "rotten"],
    };
  };
})();
