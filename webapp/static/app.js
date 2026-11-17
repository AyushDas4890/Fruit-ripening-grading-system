// Ripe or Not – front end. Motion is used to explain: the pointer slides to the grade,
// bars fill to their probability, dashed arrows draw themselves as you reach them.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const GRADES = ["unripe", "partially ripe", "ripe", "rotten"];
const NAMES = {
  cnn: "Small CNN, from scratch",
  tl_softmax: "ResNet18 + softmax",
  tl_ordinal: "ResNet18 + ordinal",
};
const NOTE = { cnn: "note-mint", tl_softmax: "note-lilac", tl_ordinal: "note-pink" };
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches || new URLSearchParams(location.search).has("sample");
const hasGsap = typeof gsap !== "undefined" && !reduce;
if (hasGsap) gsap.registerPlugin(ScrollTrigger);

/* ---------------------------------------------------------------- dashed lines draw in */
function prepDraw(path) {
  const len = path.getTotalLength();
  path.style.strokeDasharray = path.getAttribute("stroke-dasharray") || getComputedStyle(path).strokeDasharray;
  // reveal with a mask-like clip: animate a second dash pattern on the dashoffset of a solid copy
  path.dataset.len = len;
}
function drawOnView() {
  $$(".draw").forEach((p) => {
    prepDraw(p);
    if (!hasGsap) return;
    const len = +p.dataset.len;
    // clip-path reveal keeps the dashed look while the line grows
    // a solid copy of the path inside a mask grows along the line, so the dashed line
    // keeps its dashes while it is revealed
    const svg = p.ownerSVGElement;
    const id = "reveal" + Math.random().toString(36).slice(2);
    const solid = p.cloneNode();
    solid.removeAttribute("class");
    solid.removeAttribute("marker-end");
    solid.setAttribute("stroke-width", "14");
    solid.style.strokeDasharray = `${len} ${len}`;
    solid.style.strokeDashoffset = len;
    solid.setAttribute("fill", "none");
    const mask = document.createElementNS("http://www.w3.org/2000/svg", "mask");
    mask.id = id;
    mask.setAttribute("maskUnits", "userSpaceOnUse");
    solid.setAttribute("stroke", "#fff");
    mask.appendChild(solid);
    (svg.querySelector("defs") || svg.insertBefore(document.createElementNS(svg.namespaceURI, "defs"), svg.firstChild)).appendChild(mask);
    p.setAttribute("mask", `url(#${id})`);
    gsap.to(solid, {
      strokeDashoffset: 0, duration: 1.1, ease: "power2.out",
      scrollTrigger: { trigger: svg, start: "top 85%", once: true },
    });
  });
}

/* ---------------------------------------------------------------- entrances */
function entrances() {
  if (!hasGsap) return;
  gsap.from(".hero-copy > *", { y: 16, opacity: 0, duration: .7, stagger: .08, ease: "power3.out" });
  gsap.from(".board", { y: 24, opacity: 0, duration: .8, delay: .2, ease: "power3.out" });
  // sticky notes drop onto the page once, with a little settle
  $$(".sec .paper, .node, .unit-note, .formula").forEach((el) => {
    gsap.from(el, {
      y: 28, opacity: 0, rotate: "+=3", duration: .7, ease: "back.out(1.4)",
      scrollTrigger: { trigger: el, start: "top 88%", once: true },
    });
  });
  gsap.from(".boxes span", {
    y: 12, opacity: 0, stagger: .08, duration: .4,
    scrollTrigger: { trigger: ".boxes", start: "top 85%", once: true },
  });
  gsap.from(".qtable .y", {
    backgroundColor: "rgba(195,235,226,0)", stagger: .07, duration: .35,
    scrollTrigger: { trigger: ".qtable", start: "top 80%", once: true },
  });
}

/* ---------------------------------------------------------------- demo */
async function loadSamples(samples) {
  const row = $("#samples");
  row.innerHTML = "";
  samples.slice(0, 8).forEach((s) => {
    const b = document.createElement("button");
    b.className = "sample";
    b.title = `${s.fruit}, true grade: ${GRADES[s.grade]}`;
    b.innerHTML = `<img src="/crops/${s.crop}" alt="${s.fruit}, ${GRADES[s.grade]}">`;
    b.onclick = async () => {
      const blob = await (await fetch(`/crops/${s.crop}`)).blob();
      run(blob, s);
    };
    row.appendChild(b);
  });
}

function setupDrop() {
  const drop = $("#drop"), file = $("#file");
  file.onchange = () => file.files[0] && run(file.files[0]);
  ["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (ev) => ev.dataTransfer.files[0] && run(ev.dataTransfer.files[0]));
}

async function run(blob, sample = null) {
  const photo = $("#photo");
  photo.src = URL.createObjectURL(blob);
  $("#drop").classList.add("has");
  $("#status").textContent = "grading…";
  const fd = new FormData();
  fd.append("file", blob, "img.jpg");
  const q = sample ? `?crop=${encodeURIComponent(sample.crop)}` : "";
  let r;
  try {
    // static build (no server): infer.js defines localPredict and runs the model in the browser
    r = window.localPredict
      ? await window.localPredict(blob, sample, (msg) => ($("#status").textContent = msg))
      : await (await fetch("/api/predict" + q, { method: "POST", body: fd })).json();
  } catch (e) {
    console.error(e);
    $("#status").textContent = window.localPredict ? "could not run the model in this browser" : "server not reachable";
    return;
  }
  show(r, sample);
}

function show(r, sample) {
  const res = $("#result");
  const first = res.hidden;
  res.hidden = false;
  const o = r.tl_ordinal;
  $("#status").textContent = r.used;
  // pointer slides to the continuous ripeness value (sum of the 3 yes-probabilities)
  const pos = (o.expected / 3) * 100;
  $("#score").textContent = o.expected.toFixed(2);
  if (hasGsap) gsap.to("#pointer", { left: pos + "%", duration: first ? 1.2 : .8, ease: "power3.inOut" });
  else $("#pointer").style.left = pos + "%";

  const truth = sample ? `<small>true grade: ${GRADES[sample.grade]} ${sample.grade === o.grade ? "– correct" : `– off by ${Math.abs(sample.grade - o.grade)}`}</small>` : "<small>grade = number of yes answers below</small>";
  $("#verdict").innerHTML = `${GRADES[o.grade]}${truth}`;

  $$(".q").forEach((q, k) => {
    const p = o.p_gt[k];
    $(".pct", q).textContent = `${(p * 100).toFixed(1)}% ${p > .5 ? "yes" : "no"}`;
    const bar = $(".bar i", q);
    if (hasGsap) gsap.to(bar, { width: p * 100 + "%", duration: .9, delay: .15 * k, ease: "power2.out" });
    else bar.style.width = p * 100 + "%";
  });

  $("#camImg").src = r.gradcam;
  const s = r.tl_softmax;
  $("#softmaxSays").textContent = GRADES[s.grade];
  $("#softmaxProbs").textContent = s.probs.map((p, i) => `${i}: ${(p * 100).toFixed(0)}%`).join("  ");

  if (first && hasGsap) {
    gsap.from("#result > *", { y: 16, opacity: 0, stagger: .1, duration: .6, ease: "power3.out" });
  }
}

/* ---------------------------------------------------------------- results */
function fmt(m, s, pct = false, d = 3) {
  return pct ? `${(m * 100).toFixed(1)}% ± ${(s * 100).toFixed(1)}` : `${m.toFixed(d)} ± ${s.toFixed(d)}`;
}

function renderResults(R) {
  if (!R.summary) {
    $("#resultsNote").textContent = "results not computed yet – run src/evaluate.py";
    return;
  }
  $("#resultsNote").textContent =
    `${R.n} fruit crops, each tested once by a model that never saw it · mean ± spread over ${R.seeds} random seeds`;
  const stats = $("#stats");
  stats.innerHTML = "";
  Object.keys(NAMES).forEach((k) => {
    const m = R.summary[k];
    if (!m) return;
    const d = document.createElement("div");
    d.className = `stat note ${NOTE[k]}`;
    d.innerHTML = `<span class="holes"></span><h4>${NAMES[k]}</h4>
      <div class="mae mono" data-v="${m.mae[0]}">0.000</div><div class="lbl">mean grade error (MAE), lower is better</div>
      <dl class="mono"><dt>accuracy</dt><dd>${fmt(m.acc[0], m.acc[1], true)}</dd>
      <dt>QWK</dt><dd>${fmt(m.qwk[0], m.qwk[1])}</dd>
      <dt>2+ grades off</dt><dd>${fmt(m.severe[0], m.severe[1], true)}</dd></dl>`;
    stats.appendChild(d);
  });
  // numbers count up when the cards come into view
  $$(".mae").forEach((el) => {
    const v = +el.dataset.v;
    if (!hasGsap) { el.textContent = v.toFixed(3); return; }
    const o = { v: 0 };
    gsap.to(o, {
      v, duration: 1.2, ease: "power2.out", onUpdate: () => (el.textContent = o.v.toFixed(3)),
      scrollTrigger: { trigger: el, start: "top 90%", once: true },
    });
  });

  // error distance bars
  const dist = $("#dist");
  dist.innerHTML = "";
  const colors = { cnn: "var(--mint)", tl_softmax: "var(--lilac)", tl_ordinal: "var(--pink)" };
  const maxV = Math.max(...Object.values(R.error_dist).flat());
  ["1 grade off", "2 grades off", "3 grades off"].forEach((lab, i) => {
    const row = document.createElement("div");
    row.className = "dist-row";
    row.innerHTML = `<span class="lab">${lab}</span><div class="dist-bars"></div>`;
    Object.keys(NAMES).forEach((k) => {
      const v = R.error_dist[k][i];
      const b = document.createElement("div");
      b.className = "dist-bar mono";
      b.innerHTML = `<i style="background:${colors[k]}" data-w="${(v / maxV) * 82}"></i><span>${v.toFixed(1)}%</span>`;
      $(".dist-bars", row).appendChild(b);
    });
    dist.appendChild(row);
  });
  const legend = document.createElement("div");
  legend.className = "mono small";
  legend.innerHTML = Object.keys(NAMES).map((k) => `<span style="display:inline-flex;gap:6px;align-items:center;margin-right:14px"><i style="width:14px;height:14px;border:2px solid #1a1a1a;border-radius:3px;background:${colors[k]}"></i>${NAMES[k]}</span>`).join("");
  dist.appendChild(legend);
  $$(".dist-bar i").forEach((el) => {
    if (!hasGsap) { el.style.width = el.dataset.w + "%"; return; }
    gsap.to(el, { width: el.dataset.w + "%", duration: .9, ease: "power2.out", scrollTrigger: { trigger: dist, start: "top 85%", once: true } });
  });

  // confusion matrix with a picker
  const pick = $("#cmPick");
  pick.innerHTML = Object.keys(NAMES).map((k) => `<option value="${k}">${NAMES[k]}</option>`).join("");
  pick.value = "tl_ordinal";
  pick.onchange = () => drawCM(R.confusion[pick.value]);
  drawCM(R.confusion.tl_ordinal);

  // per-fruit table
  const rows = R.per_fruit.map((r) => `<tr><td>${r.fruit}</td><td>${NAMES[r.model]}</td><td>${r.mae.toFixed(3)}</td><td>${(r.acc * 100).toFixed(1)}%</td><td>${r.qwk.toFixed(3)}</td><td>${(r.severe * 100).toFixed(1)}%</td></tr>`).join("");
  $("#perFruit").innerHTML = `<h3>Per fruit</h3><table class="mono"><thead><tr><th>fruit</th><th>model</th><th>MAE</th><th>accuracy</th><th>QWK</th><th>2+ off</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function drawCM(cm) {
  const el = $("#cm");
  const max = Math.max(...cm.flat());
  let h = `<span class="hd"></span>` + GRADES.map((_, j) => `<span class="hd">${j}</span>`).join("");
  cm.forEach((row, i) => {
    h += `<span class="hd">${i}</span>`;
    row.forEach((v, j) => {
      const a = v / max;
      const bg = i === j ? `rgba(240,124,160,${0.15 + 0.85 * a})` : `rgba(163,184,250,${a ? 0.15 + 0.85 * a : 0})`;
      h += `<span style="background:${bg};${a > .55 ? "color:#fff" : ""}">${v}</span>`;
    });
  });
  el.innerHTML = h;
}

/* ---------------------------------------------------------------- boot */
(async function boot() {
  setupDrop();
  drawOnView();
  entrances();
  let R = {};
  try { R = await (await fetch(window.RESULTS_URL || "/api/results")).json(); } catch (e) { /* offline */ }
  renderResults(R);
  if (R.samples) loadSamples(R.samples);
  if (hasGsap) ScrollTrigger.refresh();
  // ?sample=N grades gallery sample N on load (used for README screenshots)
  const auto = new URLSearchParams(location.search).get("sample");
  if (auto !== null && R.samples) $$(".sample")[+auto]?.click();
})();
