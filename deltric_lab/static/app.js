"use strict";
const $ = (id) => document.getElementById(id);
const palette = [
  "#6477d4",
  "#dca269",
  "#63ad9c",
  "#ad89cb",
  "#de8298",
  "#6cacc8",
  "#aeb96b",
  "#bc9380",
  "#7f93af",
  "#a59dd7",
  "#ceab57",
  "#94b7aa",
];
const S = {
  catalog: [],
  name: null,
  run: "default",
  data: null,
  selected: null,
  evidence: null,
  region: [],
  frame: 0,
  split: false,
  playing: false,
  cameras: {},
  tab: "explore",
  workspace: {},
  experimentRows: [],
  notes: [],
  loadToken: 0,
  pointToken: 0,
  busy: false,
};
let saveTimer, toastTimer, pointTimer, tickTimer;
function esc(text) {
  return String(text ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
}
function num(v, d = 3) {
  return v == null || !Number.isFinite(Number(v)) ? "—" : Number(v).toFixed(d);
}
function compact(v) {
  return v == null ? "?" : Number(v).toLocaleString();
}
function short(name) {
  return name
    .replace(/^testing\/(real_world|additional_anomaly)\//, "")
    .replace(/^gaia_/, "Gaia · ")
    .replace(/_/g, " ");
}
function delta(v) {
  return v == null ? "—" : `${v >= 0 ? "+" : ""}${num(v)}`;
}
function color(label) {
  return label < 0
    ? "#b4bdc9"
    : palette[((label % palette.length) + palette.length) % palette.length];
}
function notify(text, error = false) {
  $("toast").textContent = text;
  $("toast").classList.remove("hidden");
  $("toast").classList.toggle("error", error);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(
    () => $("toast").classList.add("hidden"),
    error ? 9000 : 4000,
  );
}
function status(text) {
  $("status").textContent = text;
}
async function api(path, body) {
  const response = await fetch(
    path,
    body
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : {},
  );
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}
function params(extra = {}) {
  return new URLSearchParams({ dataset: S.name, run: S.run, ...extra });
}
function safe(fn) {
  return (...args) =>
    Promise.resolve()
      .then(() => fn(...args))
      .catch((e) => {
        notify(e.message, true);
        status(e.message);
      });
}
function hasData() {
  if (!S.data) {
    notify("Open a recorded case first.");
    return false;
  }
  return true;
}
function stages() {
  return S.data && $("pipeline-mode").value === "full"
    ? S.data.stage_labels.length
    : 0;
}
function claimCount() {
  return Math.max(0, S.frame - stages());
}
function total() {
  return S.data ? stages() + S.data.claims.length : 0;
}
function currentLabels() {
  if (!S.data) return [];
  const count = claimCount(),
    offset = stages();
  if (offset && S.frame < offset) return S.data.stage_labels[S.frame];
  const labels = S.data.base.slice();
  for (let i = 0; i < count; i++) {
    const c = S.data.claims[i];
    labels[c[0]] = c[1];
  }
  return labels;
}
function labelsFor(view) {
  if (!S.data) return [];
  return {
    current: S.visible,
    before: S.data.base,
    truth: S.data.y,
    default: S.data.recorded_final,
    final: S.data.final,
  }[view];
}
function claimsFor(view) {
  if (view === "before" || view === "truth") return [];
  const claims = view === "default" ? S.data.recorded_claims : S.data.claims;
  return view === "current" ? claims.slice(0, claimCount()) : claims;
}
function winningPath(node, claims) {
  const parents = new Map(claims.map((c) => [c[0], c[4]]));
  const path = [node], seen = new Set(path);
  while (parents.has(path[0])) {
    const parent = parents.get(path[0]);
    if (seen.has(parent)) break;
    path.unshift(parent);
    seen.add(parent);
  }
  return path;
}
function pointOutcome(truth, owner) {
  if (truth < 0)
    return owner < 0
      ? { tone: "green", text: "CORRECTLY UNASSIGNED" }
      : { tone: "red", text: "ANOMALY ASSIGNED" };
  return owner < 0
    ? { tone: "amber", text: "NORMAL UNASSIGNED" }
    : { tone: "neutral", text: "NORMAL ASSIGNED" };
}
function choose2(n) {
  return (n * (n - 1)) / 2;
}
function ari(truth, pred, clean = false) {
  let n = 0;
  const rows = new Map(),
    cols = new Map(),
    cells = new Map();
  for (let i = 0; i < truth.length; i++) {
    if (clean && truth[i] < 0) continue;
    n++;
    const a = truth[i],
      b = pred[i],
      key = `${a}:${b}`;
    rows.set(a, (rows.get(a) || 0) + 1);
    cols.set(b, (cols.get(b) || 0) + 1);
    cells.set(key, (cells.get(key) || 0) + 1);
  }
  if (n < 2) return 1;
  const sum = (m) => [...m.values()].reduce((a, v) => a + choose2(v), 0),
    a = sum(rows),
    b = sum(cols),
    c = sum(cells),
    expected = (a * b) / choose2(n),
    den = (a + b) / 2 - expected;
  return Math.abs(den) < 1e-12 ? 1 : (c - expected) / den;
}
function metrics(labels) {
  let tp = 0,
    fp = 0,
    fn = 0,
    tn = 0;
  for (let i = 0; i < labels.length; i++) {
    const trueAnomaly = S.data.y[i] < 0,
      flagged = labels[i] < 0;
    if (trueAnomaly && flagged) tp++;
    else if (!trueAnomaly && flagged) fp++;
    else if (trueAnomaly) fn++;
    else tn++;
  }
  return {
    ari_all: ari(S.data.y, labels),
    ari_clean: ari(S.data.y, labels, true),
    f1: tp + fn ? (2 * tp) / (2 * tp + fp + fn) : null,
    recall: tp + fn ? tp / (tp + fn) : null,
    precision: tp + fp ? tp / (tp + fp) : 0,
    TP: tp,
    FP: fp,
    FN: fn,
    TN: tn,
  };
}
function setFrame(frame, { persist = true, inspect = true } = {}) {
  if (!S.data) return;
  S.frame = Math.max(0, Math.min(total(), Math.round(frame)));
  $("timeline").value = S.frame;
  S.visible = currentLabels();
  const count = claimCount(),
    offset = stages();
  $("frame-label").textContent =
    offset && S.frame < offset
      ? S.data.stage_names[S.frame]
      : `Claim ${compact(count)} / ${compact(S.data.claims.length)}`;
  renderMetrics();
  renderInspector();
  regionSummary(false);
  renderPlots();
  drawQuality();
  noteContext();
  if (inspect && S.selected !== null) {
    clearTimeout(pointTimer);
    pointTimer = setTimeout(
      () =>
        inspectPoint(
          S.selected,
          false,
          S.evidence?.route?.cluster ?? null,
          true,
        ).catch((e) => status(e.message)),
      140,
    );
  }
  if (persist) scheduleSave();
}
function renderMetrics() {
  if (!S.data) return;
  const before = S.data.meta.before,
    now = metrics(S.visible);
  S.scores = now;
  const count = claimCount(),
    nodes = S.data.claims.slice(0, count).map((c) => c[0]),
    absorbed = nodes.filter((n) => S.data.y[n] < 0).length,
    recovered = nodes.length - absorbed;
  const cards = [
    [
      "ANOMALY F1",
      before.f1,
      now.f1,
      `TP ${now.TP} · FP ${now.FP} · FN ${now.FN}`,
    ],
    [
      "CLUSTER QUALITY · ARI",
      before.ari_all,
      now.ari_all,
      "All points · permutation invariant",
    ],
    ["ANOMALIES ABSORBED", 0, absorbed, "New true anomalies assigned"],
    ["NORMALS CLAIMED", 0, recovered, "Destination still needs checking"],
  ];
  $("metrics").innerHTML = cards
    .map(
      ([title, a, b, foot], i) =>
        `<div class="metric-card"><div class="metric-title">${title}</div><div class="metric-value ${i === 2 && b ? "negative" : ""}">${i < 2 ? `<span class="metric-before">${num(a)}<b>→</b></span>${num(b)}` : compact(b)}</div><div class="metric-foot"><span>${foot}</span>${i < 2 ? `<span class="${b >= a ? "positive" : "negative"}">${delta(b == null || a == null ? null : b - a)}</span>` : ""}</div></div>`,
    )
    .join("");
}
function renderCatalog() {
  const term = $("search").value.toLowerCase(),
    filter = $("dataset-filter").value,
    sort = $("sort").value;
  let rows = S.catalog.filter(
    (r) =>
      (r.dataset.toLowerCase().includes(term) ||
        short(r.dataset).toLowerCase().includes(term)) &&
      (filter === "all" ||
        (filter === "ready" && r.ready) ||
        (filter === "gaia" && r.dataset.includes("gaia")) ||
        (filter === "harm" && r.delta?.f1 < 0) ||
        (filter === "clean" &&
          r.ready &&
          (r.delta?.f1 > 0 ||
            (r.swallowed_anomalies === 0 && r.recovered_normals > 0)))),
  );
  rows.sort((a, b) =>
    sort === "f1"
      ? (a.delta?.f1 ?? 10) - (b.delta?.f1 ?? 10)
      : sort === "f1-best"
        ? (b.delta?.f1 ?? -10) - (a.delta?.f1 ?? -10)
      : sort === "size"
        ? (a.n ?? a.bytes) - (b.n ?? b.bytes)
        : sort === "gaia"
          ? Number(b.dataset.includes("gaia")) -
              Number(a.dataset.includes("gaia")) ||
            a.dataset.localeCompare(b.dataset)
          : a.dataset.localeCompare(b.dataset),
  );
  $("dataset-total").textContent =
    `${S.catalog.filter((r) => r.ready).length} ready`;
  $("dataset-list").innerHTML = rows.length
    ? rows
        .map(
          (r) =>
            `<button class="dataset-card ${r.dataset === S.name ? "active" : ""}" data-dataset="${esc(r.dataset)}"><div class="dataset-group"><span>${r.dataset.includes("gaia") ? "ASTRONOMY" : r.dataset.includes("text") ? "TEXT EMBEDDINGS" : r.dataset.includes("adbench") ? "ANOMALY BENCHMARK" : "DEVELOPMENT DATA"}</span><span>${r.ready ? "●" : "○"}</span></div><div class="dataset-name">${esc(short(r.dataset))}</div><div class="dataset-details"><span>${r.n ? compact(r.n) + " pts" : num(r.bytes / 1048576, 1) + " MB"}${r.d ? " · " + r.d + "D" : ""}</span><span class="${r.delta?.f1 < 0 ? "negative" : r.delta?.f1 > 0 ? "positive" : ""}">${r.ready ? "Δ F1 " + delta(r.delta?.f1) : "Not recorded"}</span></div></button>`,
        )
        .join("")
    : '<div class="empty-state">No matching datasets. Try “All development data”.</div>';
  $("dataset-list")
    .querySelectorAll("[data-dataset]")
    .forEach((b) => (b.onclick = safe(() => openDataset(b.dataset.dataset))));
}
async function refreshCatalog() {
  S.catalog = await api("/api/catalog");
  renderCatalog();
}
function stateSnapshot() {
  const ids = [
    "projection",
    "pipeline-mode",
    "view-a",
    "view-b",
    "color-a",
    "color-b",
    "point-filter",
    "cluster-focus",
    "speed",
  ];
  const controls = Object.fromEntries(ids.map((id) => [id, $(id).value]));
  const checks = Object.fromEntries(
    ["linked", "graph", "tree", "frontier", "harm", "paths"].map((id) => [
      id,
      $(id).checked,
    ]),
  );
  return {
    run: S.run,
    selected: S.selected,
    region: S.region,
    frame: S.frame,
    split: S.split,
    cameras: S.cameras,
    controls,
    checks,
    tab: S.tab,
    canvasFocus: document.body.classList.contains("canvas-focus"),
    noteDraft: $("note-text").value,
    sidebarHidden: document.body.classList.contains("no-sidebar"),
    inspectorHidden: document.body.classList.contains("no-inspector"),
  };
}
function scheduleSave() {
  if (!S.data) return;
  $("save-state").textContent = "Saving view…";
  clearTimeout(saveTimer);
  saveTimer = setTimeout(safe(saveWorkspace), 650);
}
async function saveWorkspace() {
  if (!S.data) return;
  const name = S.name,
    state = stateSnapshot();
  await api("/api/workspace", { dataset: name, state });
  localStorage.setItem("deltric-last-dataset", name);
  $("save-state").textContent = "Workspace saved locally";
}
async function openDataset(name, { run = null, restore = true } = {}) {
  const previousDraft = name === S.name ? $("note-text").value : "";
  if (S.data) await saveWorkspace();
  pause();
  clearTimeout(pointTimer);
  document
    .querySelectorAll(
      ".workspace button,.workspace select,.workspace input,.workspace textarea",
    )
    .forEach((b) => (b.disabled = true));
  document.querySelector(".workspace").classList.add("loading");
  const token = ++S.loadToken;
  S.pointToken++;
  S.name = name;
  S.selected = null;
  S.evidence = null;
  S.region = [];
  S.data = null;
  S.cameras = {};
  renderCatalog();
  status(`Opening ${name}…`);
  $("case-title").textContent = short(name);
  const row = S.catalog.find((r) => r.dataset === name);
  if (!row?.ready) {
    showUnrecorded(row);
    finishLoading();
    return;
  }
  try {
    const workspace = restore
      ? await api("/api/workspace?" + new URLSearchParams({ dataset: name }))
      : {};
    if (token !== S.loadToken) return;
    S.run = run || workspace.run || "default";
    let data;
    try {
      data = await api("/api/case?" + params());
    } catch (e) {
      if (S.run === "default") throw e;
      S.run = "default";
      data = await api("/api/case?" + params());
      notify(
        "Saved experiment cannot be restored; opened recorded default.",
        true,
      );
    }
    if (token !== S.loadToken) return;
    S.data = data;
    S.workspace = workspace;
    $("note-text").value = workspace.noteDraft ?? previousDraft;
    document.body.classList.toggle("canvas-focus", !!workspace.canvasFocus);
    S.selected =
      restore && Number.isInteger(workspace.selected)
        ? workspace.selected
        : null;
    S.region = restore ? workspace.region || [] : [];
    S.evidence = null;
    S.cameras = restore ? workspace.cameras || {} : {};
    S.split = restore ? !!workspace.split : true;
    configureClusters(workspace.controls?.["cluster-focus"] || "all");
    for (const [id, value] of Object.entries(workspace.controls || {})) {
      if ($(id) && [...$(id).options].some((o) => o.value === value))
        $(id).value = value;
    }
    for (const [id, value] of Object.entries(workspace.checks || {}))
      if ($(id)) $(id).checked = value;
    const projectionOptions = Object.keys(data.projections);
    $("projection").innerHTML = projectionOptions
      .map(
        (p) =>
          `<option value="${esc(p)}">${p === "graph" ? "Graph projection" : p === "pca" ? "PCA" : esc(p)}</option>`,
      )
      .join("");
    $("projection").value = projectionOptions.includes(
      workspace.controls?.projection,
    )
      ? workspace.controls.projection
      : "graph";
    $("case-subtitle").textContent =
      `${compact(data.meta.n)} points · ${data.meta.d} original features · ${compact(data.y.filter((y) => y < 0).length)} true anomalies · ${Object.keys(data.clusters).length} source clusters`;
    $("run-badge").textContent = data.meta.name;
    $("stale-banner").classList.toggle("hidden", !data.meta.stale);
    $("stale-banner").textContent =
      "This recorded case predates changes to the dataset or pipeline. The reassignment replay matches its saved labels; record a fresh case to assess current pipeline behavior.";
    for (const key of ["reach", "penalty", "density"])
      $(key).value = data.meta.settings[key];
    $("density-clip").checked = !!data.meta.settings.density_clip;
    $("timeline").max = total();
    $("tab-explore").classList.remove("hidden");
    $("unrecorded-panel")?.remove();
    applyLayout();
    document.body.classList.toggle("no-sidebar", !!workspace.sidebarHidden);
    document.body.classList.toggle("no-inspector", !!workspace.inspectorHidden);
    setFrame(
      restore && Number.isFinite(workspace.frame) ? workspace.frame : total(),
      { persist: false, inspect: false },
    );
    setTab(restore ? workspace.tab || "explore" : "explore");
    await Promise.all([refreshExperiments(), refreshNotes()]);
    if (token !== S.loadToken) return;
    if (S.selected !== null && S.selected < data.y.length)
      await inspectPoint(S.selected, false);
    else renderInspector();
    status(
      `Verified production replay · ${compact(data.claims.length)} accepted claims · ${compact(data.event_summary.budget)} over-budget attempts`,
    );
    localStorage.setItem("deltric-last-dataset", name);
    scheduleSave();
  } catch (error) {
    if (token !== S.loadToken) return;
    notify(error.message, true);
    status(error.message);
    showUnrecorded(row, error.message);
  } finally {
    if (token === S.loadToken) finishLoading();
  }
}
function finishLoading() {
  document
    .querySelectorAll(
      ".workspace button,.workspace select,.workspace input,.workspace textarea",
    )
    .forEach((b) => (b.disabled = false));
  document.querySelector(".workspace").classList.remove("loading");
}
function showUnrecorded(row, error = "") {
  S.run = "default";
  $("case-subtitle").textContent =
    "Record a laptop-sized run to inspect its graph and reassignment queue.";
  $("metrics").innerHTML = "";
  $("stale-banner").classList.add("hidden");
  $("tab-explore").classList.add("hidden");
  $("unrecorded-panel")?.remove();
  const panel = document.createElement("div");
  panel.id = "unrecorded-panel";
  panel.className = "run-panel";
  panel.innerHTML = `<div class="eyebrow">LOCAL DATASET RUNNER</div><h2 style="margin-top:10px">${error ? "Record a fresh case" : "Ready to investigate this dataset"}</h2><p>${error ? esc(error) : "This development dataset has no saved diagnostic case yet. The runner executes current DelTriC defaults, saves upstream stages, and verifies the final reassignment trace."}</p><p>Choose a time limit for your laptop. Large or high-dimensional datasets can take longer; timed-out runs leave a log.</p><label>Time limit <input id="run-timeout" type="number" min="10" max="900" value="180"> seconds</label><button class="primary" id="run-new">Record case →</button>`;
  $("tab-explore").before(panel);
  $("run-new").onclick = safe(() => evaluateDataset());
  renderInspector();
  setTab("explore");
}
function applyLayout() {
  $("plot-grid").classList.toggle("split", S.split);
  $("panel-b").classList.toggle("hidden", !S.split);
  $("single").classList.toggle("active", !S.split);
  $("split").classList.toggle("active", S.split);
  requestAnimationFrame(renderPlots);
}
function setTab(tab) {
  S.tab = tab;
  document
    .querySelectorAll(".work-tabs>button")
    .forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document
    .querySelectorAll(".tab-content")
    .forEach((el) => el.classList.toggle("active", el.id === "tab-" + tab));
  $("unrecorded-panel")?.classList.toggle("hidden", tab !== "explore");
  if (tab === "explore")
    requestAnimationFrame(() => {
      renderPlots();
      drawQuality();
    });
  if (tab === "notes") noteContext();
  if (tab === "costs") requestAnimationFrame(drawAnatomy);
  scheduleSave();
}
function pause() {
  S.playing = false;
  $("play").textContent = "▶";
  $("play").setAttribute("aria-label", "Play replay");
  clearTimeout(tickTimer);
}
function play() {
  if (!hasData()) return;
  if (S.playing) {
    pause();
    return;
  }
  if (S.frame >= total()) setFrame(0);
  S.playing = true;
  $("play").textContent = "Ⅱ";
  $("play").setAttribute("aria-label", "Pause replay");
  function tick() {
    if (!S.playing) return;
    const step = S.frame < stages() ? 1 : Number($("speed").value);
    setFrame(S.frame + step, { inspect: true });
    if (S.frame >= total()) {
      pause();
      if (S.selected !== null) inspectPoint(S.selected, false);
      return;
    }
    tickTimer = setTimeout(tick, 180);
  }
  tickTimer = setTimeout(tick, 180);
}
function firstHarm(inspect = false) {
  if (!hasData()) return;
  const i = S.data.claims.findIndex((c) => S.data.y[c[0]] < 0);
  if (i < 0) {
    notify("This run absorbed no true anomalies.");
    return;
  }
  pause();
  setFrame(stages() + i + 1);
  if (inspect) inspectPoint(S.data.claims[i][0]);
}
const majorities = new WeakMap();
function sourceMajority() {
  // Most common truth label among each cluster's fixed normal sources.
  if (majorities.has(S.data)) return majorities.get(S.data);
  const counts = new Map();
  S.data.base.forEach((c, i) => {
    const y = S.data.y[i];
    if (c < 0 || y < 0) return;
    const row = counts.get(c) || new Map();
    row.set(y, (row.get(y) || 0) + 1);
    counts.set(c, row);
  });
  const result = new Map(
    [...counts].map(([c, row]) => [
      c,
      [...row].sort((a, b) => b[1] - a[1])[0][0],
    ]),
  );
  majorities.set(S.data, result);
  return result;
}
function normalMatch(node, cluster) {
  return (
    S.data.y[node] >= 0 && sourceMajority().get(cluster) === S.data.y[node]
  );
}
function majorityNote(node, cluster) {
  const truth = S.data.y[node],
    majority = sourceMajority().get(cluster);
  if (truth < 0 || majority === undefined) return "";
  return ` Most of cluster ${cluster}'s fixed normal sources have truth ${majority}${majority === truth ? ", matching this point." : `, not ${truth}.`}`;
}
function nextNormalClaim(inspect = false) {
  if (!hasData()) return;
  const claims = S.data.claims,
    matches = (c) => normalMatch(c[0], c[1]);
  let i = claims.findIndex((c, k) => k >= claimCount() && matches(c));
  if (i < 0) i = claims.findIndex(matches);
  if (i < 0) {
    notify("No normal claim matches its source cluster's majority truth label.");
    return;
  }
  pause();
  setFrame(stages() + i + 1);
  if (inspect) inspectPoint(claims[i][0]);
}
class Plot {
  constructor(id, suffix) {
    this.canvas = $(id);
    this.suffix = suffix;
    this.drag = null;
    this.selectionBox = null;
    this.hover = null;
    this.bind();
  }
  get projection() {
    return $("projection").value;
  }
  get xy() {
    return S.data?.projections[this.projection] || [];
  }
  get key() {
    return this.projection + ($("linked").checked ? "" : ":" + this.suffix);
  }
  get camera() {
    if (!S.cameras[this.key]) this.fit(false);
    return S.cameras[this.key];
  }
  dimensions() {
    const box = this.canvas.getBoundingClientRect();
    return { w: Math.max(1, box.width), h: Math.max(1, box.height) };
  }
  fit(persist = true, nodes = null) {
    if (!S.data) return;
    const points = nodes?.length ? nodes.map((n) => this.xy[n]) : this.xy;
    let xmin = Infinity,
      xmax = -Infinity,
      ymin = Infinity,
      ymax = -Infinity;
    for (const p of points) {
      xmin = Math.min(xmin, p[0]);
      xmax = Math.max(xmax, p[0]);
      ymin = Math.min(ymin, p[1]);
      ymax = Math.max(ymax, p[1]);
    }
    if (!Number.isFinite(xmin)) return;
    const size = Math.max(xmax - xmin, ymax - ymin, 1e-5);
    S.cameras[this.key] = {
      cx: (xmin + xmax) / 2,
      cy: (ymin + ymax) / 2,
      span: size * 1.17,
    };
    renderPlots();
    if (persist) scheduleSave();
  }
  transform() {
    const { w, h } = this.dimensions(),
      cam = this.camera,
      scale = Math.min(w - 64, h - 54) / cam.span;
    return {
      w,
      h,
      scale,
      px: w / 2 - cam.cx * scale,
      py: h / 2 + cam.cy * scale,
    };
  }
  screen(point, t = this.transform()) {
    return [point[0] * t.scale + t.px, -point[1] * t.scale + t.py];
  }
  world(x, y, t = this.transform()) {
    return [(x - t.px) / t.scale, (t.py - y) / t.scale];
  }
  zoom(factor, x, y) {
    if (!S.data) return;
    const t = this.transform();
    x = x ?? t.w / 2;
    y = y ?? t.h / 2;
    const before = this.world(x, y, t),
      cam = this.camera;
    cam.span = Math.max(1e-9, Math.min(1e12, cam.span / factor));
    const after = this.world(x, y);
    cam.cx += before[0] - after[0];
    cam.cy += before[1] - after[1];
    renderPlots();
    scheduleSave();
  }
  nearest(x, y) {
    if (!S.data) return null;
    const t = this.transform(),
      labels = labelsFor($("view-" + this.suffix).value);
    let best = null,
      dist = 130;
    for (let i = 0; i < this.xy.length; i++) {
      if (!pointAllowed(i, labels)) continue;
      const p = this.screen(this.xy[i], t),
        d = (p[0] - x) ** 2 + (p[1] - y) ** 2;
      if (d < dist) {
        dist = d;
        best = i;
      }
    }
    return best;
  }
  bind() {
    const c = this.canvas;
    const coords = (e) => {
      const r = c.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    c.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const [x, y] = coords(e);
        this.zoom(Math.exp(-e.deltaY * 0.0015), x, y);
      },
      { passive: false },
    );
    c.addEventListener("pointerdown", (e) => {
      if (!S.data || e.button !== 0) return;
      const [x, y] = coords(e);
      c.setPointerCapture(e.pointerId);
      this.drag = { x, y, lastX: x, lastY: y, shift: e.shiftKey, moved: false };
      this.selectionBox = e.shiftKey ? [x, y, x, y] : null;
      $("tooltip").classList.add("hidden");
    });
    c.addEventListener("pointermove", (e) => {
      if (!S.data) return;
      const [x, y] = coords(e);
      if (this.drag) {
        const d = this.drag;
        if (Math.hypot(x - d.x, y - d.y) > 3) d.moved = true;
        if (d.shift) this.selectionBox = [d.x, d.y, x, y];
        else {
          const t = this.transform();
          this.camera.cx -= (x - d.lastX) / t.scale;
          this.camera.cy += (y - d.lastY) / t.scale;
          d.lastX = x;
          d.lastY = y;
        }
        renderPlots();
        return;
      }
      const node = this.nearest(x, y);
      if (node === null) {
        $("tooltip").classList.add("hidden");
        return;
      }
      const labels = labelsFor($("view-" + this.suffix).value),
        claim = S.data.claims.findIndex((cl) => cl[0] === node);
      $("tooltip").innerHTML =
        `<strong>Point ${node}</strong> · ${S.data.y[node] < 0 ? "true anomaly" : "truth " + S.data.y[node]}<br><small>Visible cluster ${labels[node]} · before ${S.data.base[node]}<br>${claim >= 0 ? "Claim " + (claim + 1) + " · cost " + num(S.data.claims[claim][2]) : "No successful reassignment claim"}<br>Click for path and proposals</small>`;
      $("tooltip").classList.remove("hidden");
      $("tooltip").style.left =
        Math.min(window.innerWidth - 245, e.clientX + 14) + "px";
      $("tooltip").style.top =
        Math.min(window.innerHeight - 110, e.clientY + 12) + "px";
    });
    c.addEventListener(
      "pointerup",
      safe(async (e) => {
        if (!this.drag) return;
        const [x, y] = coords(e),
          d = this.drag;
        this.drag = null;
        if (d.shift && d.moved) {
          const t = this.transform(),
            labels = labelsFor($("view-" + this.suffix).value);
          S.region = [];
          for (let i = 0; i < this.xy.length; i++) {
            const p = this.screen(this.xy[i], t);
            if (
              pointAllowed(i, labels) &&
              p[0] >= Math.min(d.x, x) &&
              p[0] <= Math.max(d.x, x) &&
              p[1] >= Math.min(d.y, y) &&
              p[1] <= Math.max(d.y, y)
            )
              S.region.push(i);
          }
          this.selectionBox = null;
          regionSummary();
          renderPlots();
          scheduleSave();
        } else if (!d.moved) {
          const node = this.nearest(x, y);
          if (node !== null) await inspectPoint(node);
        } else scheduleSave();
      }),
    );
    c.addEventListener("pointercancel", () => {
      this.drag = null;
      this.selectionBox = null;
      renderPlots();
    });
    c.addEventListener("pointerleave", () => {
      if (!this.drag) $("tooltip").classList.add("hidden");
    });
    c.addEventListener("dblclick", () => {
      if (S.selected !== null)
        this.fit(true, [S.selected, ...(S.evidence?.path || [])]);
    });
  }
  draw() {
    if (!S.data || (this.suffix === "b" && !S.split) || S.tab !== "explore")
      return;
    const { w, h } = this.dimensions();
    if (w < 5 || h < 5) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    const ctx = this.canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, w, h);
    const t = this.transform(),
      positions = this.xy.map((p) => this.screen(p, t)),
      view = $("view-" + this.suffix).value,
      labels = labelsFor(view),
      colors = $("color-" + this.suffix).value,
      count = claimCount(),
      changed = new Set(S.data.claims.slice(0, count).map((c) => c[0]));
    const visible = (p) =>
      p[0] > -15 && p[0] < w + 15 && p[1] > -15 && p[1] < h + 15;
    // Subtle coordinate grid, aligned with the persistent camera.
    const ideal = 70 / t.scale,
      power = 10 ** Math.floor(Math.log10(ideal)),
      step = power * ([1, 2, 5, 10].find((k) => k * power >= ideal) || 10);
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#f1f3f8";
    ctx.fillStyle = "#b0bac8";
    ctx.font = "10px ui-sans-serif, sans-serif";
    const min = this.world(25, h - 25, t),
      max = this.world(w - 25, 15, t);
    for (let x = Math.ceil(min[0] / step) * step; x <= max[0]; x += step) {
      const sx = x * t.scale + t.px;
      ctx.beginPath();
      ctx.moveTo(sx, 12);
      ctx.lineTo(sx, h - 20);
      ctx.stroke();
      ctx.fillText(num(x, Math.abs(step) < 1 ? 2 : 0), sx + 4, h - 12);
    }
    for (let y = Math.ceil(min[1] / step) * step; y <= max[1]; y += step) {
      const sy = -y * t.scale + t.py;
      ctx.beginPath();
      ctx.moveTo(25, sy);
      ctx.lineTo(w - 12, sy);
      ctx.stroke();
      ctx.fillText(num(y, Math.abs(step) < 1 ? 2 : 0), 7, sy - 4);
    }
    const segment = (u, v, stroke, width = 1, dash = []) => {
      const a = positions[u],
        b = positions[v];
      if (!a || !b) return;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(...a);
      ctx.lineTo(...b);
      ctx.stroke();
      ctx.setLineDash([]);
    };
    if ($("graph").checked) {
      ctx.globalAlpha = 0.45;
      for (const [u, v] of S.data.edges)
        if (visible(positions[u]) || visible(positions[v]))
          segment(u, v, "#c5ccdc", 0.6);
      ctx.globalAlpha = 1;
    }
    const viewClaims = claimsFor(view);
    $("stage-" + this.suffix).textContent =
      view === "current" && stages() && S.frame < stages()
        ? S.data.stage_names[S.frame]
        : view === "truth"
          ? "Truth labels"
          : `${compact(viewClaims.length)} claims`;
    const arrow = (u, v, stroke, width = 1.8) => {
      segment(u, v, stroke, width);
      const a = positions[u], b = positions[v];
      if (!a || !b) return;
      const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (length < 12) return;
      const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
      const x = b[0] - Math.cos(angle) * 5, y = b[1] - Math.sin(angle) * 5;
      ctx.beginPath();
      ctx.moveTo(x - Math.cos(angle - 0.5) * 5, y - Math.sin(angle - 0.5) * 5);
      ctx.lineTo(x, y);
      ctx.lineTo(x - Math.cos(angle + 0.5) * 5, y - Math.sin(angle + 0.5) * 5);
      ctx.stroke();
    };
    if ($("tree").checked) {
      for (const c of viewClaims)
        if (pointAllowed(c[0], labels) &&
          ($("cluster-focus").value === "all" || c[1] === Number($("cluster-focus").value)))
          arrow(c[4], c[0], color(c[1]));
    }
    // Frontier uses production hop costs and current ownership, not projected lengths.
    const frontier = new Set();
    if ($("frontier").checked && view === "current" && !(stages() && S.frame < stages())) {
      const settings = S.data.meta.settings;
      for (let i = 0; i < S.data.edges.length; i++) {
        const [u, v] = S.data.edges[i];
        for (const [parent, node] of [
          [u, v],
          [v, u],
        ]) {
          if (S.visible[parent] < 0 || S.visible[node] >= 0) continue;
          const cluster = S.visible[parent],
            ref = S.data.clusters[cluster];
          if (!ref) continue;
          let rho = S.data.core[node] / ref.core_reference;
          if (settings.density_clip) rho = Math.max(1, rho);
          const hop =
              ((S.data.lengths[i] / ref.mean_edge) * rho ** settings.density) **
              (1 + settings.penalty),
            parentClaim = S.claimMap.get(parent),
            parentCost = parentClaim ? parentClaim.claim[2] : 0;
          if (parentCost + hop <= settings.reach) frontier.add(node);
        }
      }
    }
    const radius =
      S.data.y.length > 10000 ? 1.9 : S.data.y.length > 2000 ? 2.4 : 3.2;
    for (let i = 0; i < positions.length; i++) {
      const p = positions[i];
      if (!visible(p) || !pointAllowed(i, labels)) continue;
      let fill = color(labels[i]);
      if (colors === "truth") fill = color(S.data.y[i]);
      if (colors === "outcome") {
        fill =
          S.data.y[i] < 0
            ? labels[i] < 0
              ? "#279b7b"
              : "#d86972"
            : labels[i] < 0
              ? "#d2a564"
              : "#6fab9b";
      }
      ctx.fillStyle = fill;
      ctx.globalAlpha =
        S.region.length && !S.regionSet.has(i)
          ? 0.18
          : labels[i] < 0
            ? 0.65
            : 0.9;
      ctx.beginPath();
      ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      if (frontier.has(i)) {
        ctx.strokeStyle = "#d8ae54";
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.arc(p[0], p[1], radius + 2.5, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (
        $("harm").checked &&
        changed.has(i) &&
        !(stages() && S.frame < stages()) &&
        view === "current"
      ) {
        if (S.data.y[i] < 0) {
          ctx.strokeStyle = "#d45965";
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.moveTo(p[0] - 4, p[1] - 4);
          ctx.lineTo(p[0] + 4, p[1] + 4);
          ctx.moveTo(p[0] - 4, p[1] + 4);
          ctx.lineTo(p[0] + 4, p[1] - 4);
          ctx.stroke();
        } else {
          ctx.strokeStyle = "#279b7b";
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.arc(p[0], p[1], radius + 2, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
    }
    if ($("paths").checked && S.selected !== null) {
      const path = winningPath(S.selected, viewClaims);
      for (let i = 1; i < path.length; i++)
        arrow(path[i - 1], path[i], "#344764", 2.8);
      if (path.length > 1) {
        for (const node of path) {
          ctx.fillStyle = "#fff";
          ctx.strokeStyle = "#344764";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(...positions[node], 4.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
      }
      // Queue proposals and hypothetical routes belong to the active run only.
      const activeView = view === "final" ||
        (view === "current" && !(stages() && S.frame < stages()));
      const evidence = S.evidence?.node === S.selected ? S.evidence : null;
      const route = activeView ? evidence?.route?.path || [] : [];
      for (let i = 1; i < route.length; i++)
        segment(route[i - 1], route[i], "#d3a144", 2, [5, 4]);
      for (const proposal of (activeView ? evidence?.proposals || [] : [])
        .filter((e) => e.at <= (view === "final" ? S.data.claims.length : count) && e.cluster !== evidence.final)
        .slice(0, 40))
        segment(proposal.parent, proposal.node, "#aa89c580", 1, [3, 4]);
    }
    if (S.selected !== null) {
      const p = positions[S.selected];
      if (p) {
        ctx.strokeStyle = "#293c60";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(...p, 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = "#fff";
        ctx.fillRect(p[0] + 10, p[1] - 23, 55, 18);
        ctx.fillStyle = "#3a4c6b";
        ctx.font = "600 10px ui-sans-serif, sans-serif";
        ctx.fillText("#" + S.selected, p[0] + 14, p[1] - 10);
      }
    }
    if (this.selectionBox) {
      const [x, y, x2, y2] = this.selectionBox;
      ctx.fillStyle = "#5264cf15";
      ctx.fillRect(x, y, x2 - x, y2 - y);
      ctx.strokeStyle = "#5264cf";
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(x, y, x2 - x, y2 - y);
      ctx.setLineDash([]);
    }
    const label =
      $("projection").value === "graph"
        ? "Graph topology projection · costs in original standardized features"
        : $("projection").value === "pca"
          ? "PCA display only · costs in original standardized features"
          : $("projection").value + " · raw feature display";
    $("panel-" + this.suffix).querySelector(".axis-label").textContent = label;
  }
}
const plots = [new Plot("canvas-a", "a"), new Plot("canvas-b", "b")];
function pointAllowed(node, labels) {
  const cluster = $("cluster-focus").value;
  if (
    cluster !== "all" &&
    S.data.base[node] !== Number(cluster) &&
    labels[node] !== Number(cluster) &&
    !(labels === S.visible && S.data.final[node] === Number(cluster))
  )
    return false;
  const filter = $("point-filter").value;
  return (
    filter === "all" ||
    (filter === "anomalies" && S.data.y[node] < 0) ||
    (filter === "normals" && S.data.y[node] >= 0) ||
    (filter === "unassigned" && labels[node] < 0) ||
    (filter === "absorbed" &&
      S.data.y[node] < 0 &&
      S.data.base[node] < 0 &&
      labels[node] >= 0) ||
    (filter === "changed" && S.data.base[node] < 0 && labels[node] >= 0) ||
    (filter.startsWith("normal-") &&
      S.data.y[node] >= 0 &&
      S.data.base[node] < 0 &&
      labels[node] >= 0 &&
      normalMatch(node, labels[node]) === (filter === "normal-match"))
  );
}
function renderPlots() {
  if (!S.data) return;
  S.claimMap = new Map(
    S.data.claims.map((c, i) => [c[0], { index: i + 1, claim: c }]),
  );
  S.regionSet = new Set(S.region);
  for (const plot of plots) plot.draw();
}
function regionSummary(announce = true) {
  if (!S.data) return;
  const anomalies = S.region.filter((n) => S.data.y[n] < 0).length;
  $("selection-summary").textContent = S.region.length
    ? `${S.region.length} selected · ${anomalies} true anomalies · shift-drag to replace`
    : S.selected !== null
      ? `Point #${S.selected} selected`
      : "";
  if (S.region.length && announce)
    notify(
      `Selected ${S.region.length} points (${anomalies} true anomalies). Use Focus selection to inspect this region.`,
    );
}
function drawQuality() {
  if (!S.data || S.tab !== "explore") return;
  const canvas = $("quality"),
    rect = canvas.getBoundingClientRect(),
    w = rect.width,
    h = rect.height;
  if (!w) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const pad = 10,
    offset = stages(),
    max = Math.max(1, total()),
    x = (v) => pad + ((w - 2 * pad) * v) / max,
    y = (v) => h - 9 - (h - 20) * v;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#edf0f6";
  for (const value of [0, 0.5, 1]) {
    ctx.beginPath();
    ctx.moveTo(pad, y(value));
    ctx.lineTo(w - pad, y(value));
    ctx.stroke();
  }
  for (const [col, label, stroke] of [
    [1, "ARI", "#6d7bd7"],
    [3, "Anomaly F1", "#d57a88"],
  ]) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    let started = false;
    const early = S.data.stage_scores
      .slice(0, offset)
      .map((s, i) => [i, s[col === 1 ? "ari_all" : "f1"]]);
    const points = [
      ...early,
      ...S.data.timeline.map((t) => [t[0] + offset, t[col]]),
    ];
    for (const [at, value] of points) {
      if (value == null) {
        started = false;
        continue;
      }
      if (!started) ctx.moveTo(x(at), y(value));
      else ctx.lineTo(x(at), y(value));
      started = true;
    }
    ctx.stroke();
    ctx.fillStyle = stroke;
    ctx.font = "9px sans-serif";
    ctx.fillText(label, w - (col === 1 ? 155 : 100), 9);
  }
  ctx.strokeStyle = "#8d99b5";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(x(S.frame), 2);
  ctx.lineTo(x(S.frame), h);
  ctx.stroke();
  ctx.setLineDash([]);
}
async function inspectPoint(
  node,
  focus = false,
  candidate = null,
  keepPlaying = false,
) {
  if (!hasData()) return;
  node = Number(node);
  if (!Number.isInteger(node) || node < 0 || node >= S.data.y.length)
    throw new Error(`Point ID must be between 0 and ${S.data.y.length - 1}`);
  const token = ++S.pointToken,
    load = S.loadToken;
  S.evidence = null;
  S.selected = node;
  $("point-id").value = node;
  if (!keepPlaying) pause();
  renderPlots();
  regionSummary();
  if (!keepPlaying)
    $("inspector-body").innerHTML =
      '<div class="empty-state">Reading path and queue evidence…</div>';
  const extra = { node, count: claimCount() };
  if (candidate !== null) extra.cluster = candidate;
  const evidence = await api("/api/point?" + params(extra));
  if (token !== S.pointToken || load !== S.loadToken) return;
  S.evidence = evidence;
  if (stages() && S.frame < stages()) {
    evidence.visible = S.visible[node];
    evidence.visible_path = [];
  }
  renderInspector();
  renderPlots();
  noteContext();
  if (focus) plots[0].fit(true, evidence.path);
  scheduleSave();
  if (window.innerWidth < 750) document.body.classList.add("mobile-inspector");
}
function renderInspector() {
  const el = $("inspector-body");
  if (!S.evidence || !S.data) {
    el.innerHTML =
      '<div class="empty-inspector"><div class="empty-orbit">◎</div><h3>Start with a point</h3><p>Click an absorbed anomaly, an unassigned point, or any place that makes you curious.</p><p>Its winning path, hop costs, and competing proposals will appear here.</p><button id="inspect-first-harm">Inspect first absorbed anomaly →</button><button id="inspect-normal-claim">Inspect a normal claim →</button></div>';
    $("inspect-first-harm").onclick = () => firstHarm(true);
    $("inspect-normal-claim").onclick = () => nextNormalClaim(true);
    return;
  }
  const p = S.evidence,
    upstream = stages() && S.frame < stages(),
    count = claimCount(),
    hopVisible = (h) => !upstream && h.claim <= count,
    proposalVisible = (e) => !upstream && (e.status === "accepted" ? e.at < count : e.at <= count),
    explanation = upstream
      ? `Visible upstream stage: ${S.data.stage_names[S.frame]}. This point currently has label ${S.visible[p.node]}. The evidence below describes its later reassignment.`
      : p.base < 0 && p.claim !== null
        ? p.claim <= count
          ? `Claim ${p.claim} assigned this point to cluster ${p.final}. ${p.truth < 0 ? "A true anomaly was absorbed." : "A true normal was assigned; check its destination cluster against ground truth."}${majorityNote(p.node, p.final)}`
          : `Still unassigned at this frame. It will be claimed at claim ${p.claim} by cluster ${p.final}.`
        : p.explanation;
  const ref = p.source;
  const owner = S.visible[p.node], outcome = pointOutcome(p.truth, owner);
  let html = `<div class="point-heading"><h3>Point #${p.node}</h3><span class="tag">${p.truth < 0 ? "TRUE ANOMALY" : "TRUE NORMAL"}</span></div><div class="point-outcome"><span class="tag ${outcome.tone}">${outcome.text}</span><small>At the current frame</small></div><p class="explanation">${esc(explanation)}</p><div class="point-stats">${[
    ["Truth", p.truth < 0 ? "Anomaly" : p.truth],
    ["Visible owner", owner < 0 ? "Unassigned" : owner],
    ["Before owner", p.base < 0 ? "Unassigned" : p.base],
    ["Final owner", p.final < 0 ? "Unassigned" : p.final],
    ["Claim number", p.claim ?? "No claim"],
    ["Core distance", num(p.core, 5)],
  ]
    .map(
      ([title, value]) =>
        `<div class="point-stat"><small>${title}</small><strong>${value}</strong></div>`,
    )
    .join(
      "",
    )}</div><button id="note-point" style="width:100%;margin-top:12px;font-size:11px">✎ Write an observation</button>`;
  if (ref) {
    html += `<div class="inspector-section"><h3>Destination source cluster <span>${p.final}</span></h3><p>${ref.size} fixed source points · ${ref.anomalies} true anomalies (${num((100 * ref.anomalies) / ref.size, 0)}%)</p>${ref.anomalies === ref.size ? '<div class="notice">Every fixed source point is a true anomaly. This error exists before reassignment and can propagate.</div>' : ""}<p style="margin-top:8px">Mean internal edge ${num(ref.mean_edge, 5)}<br>Median core distance ${num(ref.core_reference, 5)}</p></div>`;
  }
  html += `<div class="inspector-section"><h3>Winning path <span>${p.hops.length} hops</span></h3>${p.hops.length ? `<div class="path-chain">${p.path.map((n, i) => `<button data-node="${n}" class="${i && !hopVisible(p.hops[i - 1]) ? "future" : ""}">${n}</button>`).join("<span>→</span>")}</div>` : "<p>No successful reassignment path for this point.</p>"}${p.hops.map((h) => `<div class="hop-card ${!hopVisible(h) ? "future" : ""}"><div class="hop-top"><strong>${h.parent} → ${h.node}</strong><span>${!hopVisible(h) ? "Future · " : ""}claim ${h.claim}</span></div><div class="hop-details"><span>Edge / mean<b>${num(h.edge_ratio)}</b></span><span>Core / median<b>${num(h.density_ratio)}</b></span><span>Hop cost<b>${num(h.hop_cost, 4)}</b></span><span>Cumulative<b>${num(h.cumulative, 4)}</b></span></div><div class="hop-formula">edge ${num(h.length, 5)} / mean ${num(h.mean_edge, 5)}<br>core ${num(h.core, 5)} / median ${num(h.core_reference, 5)}</div></div>`).join("")}</div>`;
  html += `<div class="inspector-section"><h3>Competing proposals <span>${p.proposals.length} attempts</span></h3><p>Production queue evidence. Faded attempts occur later than the visible reassignment frame.</p><div style="margin-top:9px">${p.proposals.length ? p.proposals.map((e) => `<div class="proposal ${e.status} ${!proposalVisible(e) ? "future" : ""}"><div class="proposal-top"><button data-candidate="${e.cluster}">Cluster ${e.cluster}</button><strong>${num(e.cost, 4)}</strong></div><div class="proposal-detail">${e.status === "accepted" ? "Accepted → permanent owner" : e.status === "owned" ? `Skipped: already owned by cluster ${e.owner}` : "Not queued: exceeds reach budget"}<br>via point ${e.parent} · after ${e.at} claims${!proposalVisible(e) ? " · future" : ""}</div></div>`).join("") : '<div class="empty-state">No recorded queue or over-budget attempt reached this point.</div>'}</div><div class="candidate-row"><select id="candidate-cluster" aria-label="Hypothetical source cluster">${Object.entries(
    S.data.clusters,
  )
    .map(
      ([c, r]) =>
        `<option value="${c}" ${p.route?.cluster === Number(c) ? "selected" : ""}>Cluster ${c} · ${r.size} sources</option>`,
    )
    .join(
      "",
    )}</select><button id="compare-route">Compare route</button></div>${p.route ? `<div class="route-explanation">${esc(p.route.reason)}${p.route.cost !== undefined ? `<br><strong>Cost ${num(p.route.cost, 4)} · ${p.route.within_budget ? "within" : "outside"} budget</strong><br>${p.route.blockers.length ? "Competing ownership along route: " + p.route.blockers.join(", ") : "No competing-owned points on this route."}` : ""}</div>` : ""}</div>`;
  html += `<div class="inspector-section"><h3>Graph neighbors <span>${p.neighbors.length}</span></h3><div class="neighbor-list">${p.neighbors.map((n) => `<button data-node="${n.node}" title="Original edge length ${num(n.length, 5)} · visible owner ${n.visible}">${n.node} <span style="color:${color(n.visible)}">●</span></button>`).join("")}</div></div><details class="inspector-section"><summary style="font-size:12px;cursor:pointer">Original features · ${p.features.length}</summary><div class="feature-table">${p.features.map((f) => `<div class="feature-row"><span>${esc(f.name)}</span><code>${num(f.value, 6)}</code></div>`).join("")}</div></details>`;
  el.innerHTML = html;
  el.querySelectorAll("[data-node]").forEach(
    (b) => (b.onclick = safe(() => inspectPoint(Number(b.dataset.node)))),
  );
  el.querySelectorAll("[data-candidate]").forEach(
    (b) =>
      (b.onclick = safe(() =>
        inspectPoint(p.node, false, Number(b.dataset.candidate)),
      )),
  );
  $("compare-route").onclick = safe(() =>
    inspectPoint(p.node, false, Number($("candidate-cluster").value)),
  );
  $("note-point").onclick = () => {
    setTab("notes");
    $("note-text").focus();
  };
}
async function refreshExperiments() {
  if (!S.name) return;
  const name = S.name,
    rows = await api(
      "/api/experiments?" + new URLSearchParams({ dataset: name }),
    );
  if (name !== S.name) return;
  S.experimentRows = rows;
  $("experiment-count").textContent = rows.length;
  renderExperiments();
}
function renderExperiments() {
  if (!S.data) {
    $("experiment-history").innerHTML =
      '<div class="empty-state">Record a case first.</div>';
    return;
  }
  const rows = [S.data.recorded_meta, ...S.experimentRows];
  $("experiment-history").innerHTML =
    `<table><thead><tr><th>Run</th><th>Reach / penalty / density</th><th>Anomaly F1</th><th>ARI</th><th>Absorbed</th><th></th></tr></thead><tbody>${rows.map((r) => `<tr class="${r.run_id === S.run ? "active-row" : ""}"><td>${esc(r.name)}<small>${r.created ? new Date(r.created).toLocaleString() : "Frozen pipeline baseline"}</small></td><td>${r.settings ? `${r.settings.reach} / ${r.settings.penalty} / ${r.settings.density}${r.settings.density_clip ? " · floor" : ""}` : "Recorded settings"}</td><td>${num(r.after.f1)}</td><td>${num(r.after.ari_all)}</td><td>${r.run_id === "default" ? S.data.y.filter((y, i) => y < 0 && S.data.base[i] < 0 && S.data.recorded_final[i] >= 0).length : r.swallowed_anomalies}</td><td><button data-run="${r.run_id}">${r.run_id === S.run ? "Active" : "Open"}</button></td></tr>`).join("")}</tbody></table>`;
  $("experiment-history")
    .querySelectorAll("[data-run]")
    .forEach((b) => (b.onclick = safe(() => activateRun(b.dataset.run))));
}
async function activateRun(run) {
  if (!hasData()) return;
  await saveWorkspace();
  pause();
  clearTimeout(pointTimer);
  status("Opening saved run…");
  const token = ++S.loadToken;
  S.pointToken++;
  const data = await api(
    "/api/case?" + new URLSearchParams({ dataset: S.name, run }),
  );
  if (token !== S.loadToken) return;
  S.data = data;
  S.run = run;
  S.evidence = null;
  S.region = [];
  S.selected = null;
  configureClusters("all");
  for (const key of ["reach", "penalty", "density"])
    $(key).value = data.meta.settings[key];
  $("density-clip").checked = !!data.meta.settings.density_clip;
  $("timeline").max = total();
  $("run-badge").textContent = data.meta.name;
  setFrame(total());
  if (run !== "default") {
    S.split = true;
    $("view-a").value = "current";
    $("view-b").value = "default";
    applyLayout();
  }
  renderInspector();
  renderExperiments();
  noteContext();
  setTab("explore");
  status("Saved run restored. Graph and original source clusters held fixed.");
  scheduleSave();
}
function experimentSettings() {
  return {
    reach: Number($("reach").value),
    penalty: Number($("penalty").value),
    density: Number($("density").value),
    density_clip: $("density-clip").checked,
  };
}
async function runJob(endpoint, body) {
  if (S.busy) {
    notify("A local job is already running.");
    return null;
  }
  S.busy = true;
  document
    .querySelectorAll("#experiment-form button,#sweep,#rerun,#run-new")
    .forEach((b) => (b.disabled = true));
  try {
    const started = await api(endpoint, { dataset: S.name, ...body });
    let result = started;
    while (result.status === "running") {
      status(result.progress || "Running locally…");
      await new Promise((r) => setTimeout(r, 800));
      result = await api("/api/job?" + new URLSearchParams({ id: started.id }));
    }
    if (result.status === "error") throw new Error(result.error);
    return result.result;
  } finally {
    S.busy = false;
    document
      .querySelectorAll("#experiment-form button,#sweep,#rerun,#run-new")
      .forEach((b) => (b.disabled = false));
  }
}
async function evaluateDataset() {
  const name = S.name;
  const timeout = Number($("run-timeout")?.value || 180);
  const result = await runJob("/api/evaluate", { timeout });
  if (!result) return;
  await refreshCatalog();
  if (S.name === name)
    await openDataset(name, { run: "default", restore: false });
  notify("Fresh case saved. Production replay verified.");
}
async function refreshNotes() {
  if (!S.name) return;
  const name = S.name,
    rows = await api("/api/notes?" + new URLSearchParams({ dataset: name }));
  if (name !== S.name) return;
  S.notes = rows;
  $("note-count").textContent = rows.length;
  renderNotes();
}
function noteContext() {
  $("note-context").textContent = S.data
    ? `${S.data.meta.name} · frame ${S.frame}${S.selected !== null ? " · point #" + S.selected : ""}`
    : "Open a dataset to attach an observation.";
}
function renderNotes() {
  $("notes-list").innerHTML = S.notes.length
    ? S.notes
        .map(
          (n) =>
            `<article class="note-card"><div class="note-meta"><span>${new Date(n.created).toLocaleString()}</span><span>${n.node !== null && n.node !== undefined ? "Point #" + n.node + " · " : ""}frame ${n.frame ?? "—"}</span></div><p>${esc(n.text)}</p><button data-note-open="${n.id}">Return to evidence ↗</button><button data-note-edit="${n.id}">Edit note</button></article>`,
        )
        .join("")
    : '<div class="empty-state">No observations yet. Start with a question, a surprising path, or an idea to test.</div>';
  $("notes-list")
    .querySelectorAll("[data-note-open]")
    .forEach(
      (b) =>
        (b.onclick = safe(async () => {
          const note = S.notes.find((n) => n.id === b.dataset.noteOpen);
          if (note.run !== S.run) await activateRun(note.run);
          setTab("explore");
          setFrame(note.frame || 0);
          if (note.node !== null && note.node !== undefined)
            await inspectPoint(note.node, true);
        })),
    );
  $("notes-list")
    .querySelectorAll("[data-note-edit]")
    .forEach(
      (b) =>
        (b.onclick = () => {
          const note = S.notes.find((n) => n.id === b.dataset.noteEdit);
          $("note-text").value = note.text;
          $("note-form").dataset.editId = note.id;
          $("note-text").focus();
          $("note-context").textContent =
            "Editing existing observation · original evidence link retained";
        }),
    );
}
function download(blob, name) {
  const url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
function bind() {
  ["search", "dataset-filter", "sort"].forEach((id) =>
    $(id).addEventListener(id === "search" ? "input" : "change", renderCatalog),
  );
  document
    .querySelectorAll("[data-tab]")
    .forEach((b) => (b.onclick = () => setTab(b.dataset.tab)));
  $("single").onclick = () => {
    S.split = false;
    applyLayout();
    scheduleSave();
  };
  $("split").onclick = () => {
    S.split = true;
    applyLayout();
    scheduleSave();
  };
  $("linked").onchange = () => {
    if (!$("linked").checked) {
      const cam = S.cameras[$("projection").value];
      if (cam) {
        S.cameras[$("projection").value + ":a"] = { ...cam };
        S.cameras[$("projection").value + ":b"] = { ...cam };
      }
    } else {
      const cam = S.cameras[$("projection").value + ":a"];
      if (cam) S.cameras[$("projection").value] = { ...cam };
    }
    renderPlots();
    scheduleSave();
  };
  [
    "projection",
    "view-a",
    "view-b",
    "color-a",
    "color-b",
    "point-filter",
    "graph",
    "tree",
    "frontier",
    "harm",
    "paths",
  ].forEach(
    (id) =>
      ($(id).onchange = () => {
        renderPlots();
        scheduleSave();
      }),
  );
  $("zoom-in").onclick = () => plots[0].zoom(1.4);
  $("zoom-out").onclick = () => plots[0].zoom(1 / 1.4);
  $("fit").onclick = () => plots.forEach((p) => p.fit());
  $("focus").onclick = () => {
    if (!hasData()) return;
    const nodes = S.region.length ? S.region : S.evidence?.path || [];
    if (!nodes.length) {
      notify("Select a point or shift-drag a region first.");
      return;
    }
    plots.forEach((p) => p.fit(true, nodes));
  };
  $("timeline").oninput = () => {
    pause();
    setFrame(Number($("timeline").value));
  };
  $("pipeline-mode").onchange = () => {
    pause();
    $("timeline").max = total();
    setFrame(total());
  };
  $("speed").onchange = scheduleSave;
  $("play").onclick = play;
  $("prev").onclick = () => {
    pause();
    setFrame(S.frame - 1);
  };
  $("next").onclick = () => {
    pause();
    setFrame(S.frame + 1);
  };
  $("start").onclick = () => {
    pause();
    setFrame(0);
  };
  $("end").onclick = () => {
    pause();
    setFrame(total());
  };
  $("first-harm").onclick = () => firstHarm(true);
  $("inspect-first-harm").onclick = () => firstHarm(true);
  $("normal-claim").onclick = () => nextNormalClaim(true);
  $("inspect-normal-claim").onclick = () => nextNormalClaim(true);
  $("find-point").onclick = safe(() =>
    inspectPoint(Number($("point-id").value), true),
  );
  $("point-id").onkeydown = safe((e) => {
    if (e.key === "Enter")
      return inspectPoint(Number($("point-id").value), true);
  });
  $("quality").onclick = (e) => {
    if (!hasData()) return;
    pause();
    const rect = $("quality").getBoundingClientRect();
    setFrame(((e.clientX - rect.left - 10) / (rect.width - 20)) * total());
  };
  $("toggle-sidebar").onclick = () => {
    if (window.innerWidth <= 1100)
      document.body.classList.toggle("mobile-sidebar");
    else document.body.classList.toggle("no-sidebar");
    requestAnimationFrame(renderPlots);
    scheduleSave();
  };
  $("toggle-inspector").onclick = () => {
    if (window.innerWidth < 750)
      document.body.classList.toggle("mobile-inspector");
    else document.body.classList.toggle("no-inspector");
    requestAnimationFrame(renderPlots);
    scheduleSave();
  };
  $("canvas-focus").onclick = () => {
    document.body.classList.toggle("canvas-focus");
    requestAnimationFrame(() => {
      renderPlots();
      drawQuality();
    });
    scheduleSave();
  };
  $("note-text").oninput = scheduleSave;
  $("help").onclick = () => $("help-dialog").showModal();
  $("close-help").onclick = () => $("help-dialog").close();
  $("png").onclick = () => {
    if (hasData())
      $("canvas-a").toBlob((blob) =>
        download(
          blob,
          S.name.replaceAll("/", "__") + "_frame_" + S.frame + ".png",
        ),
      );
  };
  $("export").onclick = safe(async () => {
    if (!hasData()) return;
    const extra = { count: claimCount() };
    if (S.selected !== null) extra.node = S.selected;
    const evidence = await api("/api/export?" + params(extra));
    evidence.workspace = stateSnapshot();
    download(
      new Blob([JSON.stringify(evidence, null, 2)], {
        type: "application/json",
      }),
      S.name.replaceAll("/", "__") + "_evidence.json",
    );
    notify("Evidence exported; a copy is saved in data/exports/.");
  });
  $("rerun").onclick = safe(() => {
    if (S.name) return evaluateDataset();
  });
  $("restore-default").onclick = safe(() => activateRun("default"));
  $("experiment-form").onsubmit = safe(async (e) => {
    e.preventDefault();
    if (!hasData()) return;
    const name = S.name;
    const result = await runJob("/api/experiment", {
      settings: experimentSettings(),
      name: $("experiment-name").value,
    });
    if (!result) return;
    if (S.name === name) {
      await refreshExperiments();
      await activateRun(result.run_id);
    }
    notify("Experiment saved and opened beside the recorded default.");
  });
  $("sweep").onclick = safe(async () => {
    if (!hasData()) return;
    const values = $("sweep-values")
      .value.split(",")
      .map((v) => Number(v.trim()));
    if (
      $("sweep-values").value.trim() === "" ||
      values.some((v) => !Number.isFinite(v))
    )
      throw new Error("Enter comma-separated reach values.");
    const name = S.name;
    const result = await runJob("/api/sweep", {
      settings: experimentSettings(),
      reaches: values,
    });
    if (!result) return;
    if (S.name === name) await refreshExperiments();
    status(`Sweep complete · ${result.length} experiments saved`);
    notify(`${result.length} experiments saved. Open any run to inspect it.`);
  });
  $("note-form").onsubmit = safe(async (e) => {
    e.preventDefault();
    if (!hasData()) return;
    const id = $("note-form").dataset.editId,
      existing = id ? S.notes.find((n) => n.id === id) : null;
    await api("/api/notes", {
      dataset: S.name,
      run: existing?.run ?? S.run,
      node: existing ? existing.node : S.selected,
      frame: existing ? existing.frame : S.frame,
      text: $("note-text").value,
      ...(id ? { id } : {}),
    });
    $("note-text").value = "";
    delete $("note-form").dataset.editId;
    noteContext();
    await refreshNotes();
    notify("Observation saved with its evidence link.");
  });
  document.addEventListener("keydown", (e) => {
    if (
      ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName) ||
      $("help-dialog").open
    )
      return;
    if (e.code === "Space") {
      e.preventDefault();
      play();
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      pause();
      setFrame(S.frame + 1);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      pause();
      setFrame(S.frame - 1);
    } else if (e.key.toLowerCase() === "f") plots.forEach((p) => p.fit());
    else if (e.key.toLowerCase() === "s") {
      $(S.split ? "single" : "split").click();
    } else if (e.key === "Escape") {
      S.region = [];
      S.selected = null;
      S.evidence = null;
      document.body.classList.remove("mobile-inspector", "mobile-sidebar");
      renderInspector();
      regionSummary();
      renderPlots();
      scheduleSave();
    }
  });
  window.addEventListener("resize", () => {
    renderPlots();
    drawQuality();
  });
  const observer = new ResizeObserver(() => {
    requestAnimationFrame(() => {
      renderPlots();
      drawQuality();
    });
  });
  plots.forEach((p) => observer.observe(p.canvas));
  window.addEventListener("pagehide", () => {
    if (S.data) {
      const blob = new Blob(
        [JSON.stringify({ dataset: S.name, state: stateSnapshot() })],
        { type: "application/json" },
      );
      navigator.sendBeacon("/api/workspace", blob);
    }
  });
}
async function initialize() {
  bind();
  await refreshCatalog();
  const remembered = localStorage.getItem("deltric-last-dataset"),
    first =
      S.catalog.find((r) => r.dataset === remembered && r.ready) ||
      S.catalog.find((r) => r.dataset === "gaia_ngc_2632" && r.ready) ||
      S.catalog.find((r) => r.ready);
  if (first) await openDataset(first.dataset);
  else status("Choose a development dataset to record your first case.");
}
initialize().catch((e) => {
  notify(e.message, true);
  status(e.message);
});

function configureClusters(selected = "all") {
  if (!S.data) return;
  $("cluster-focus").innerHTML =
    '<option value="all">All source clusters</option>' +
    Object.entries(S.data.clusters)
      .map(
        ([id, ref]) =>
          `<option value="${id}">#${id} · ${ref.size} sources · ${ref.anomalies} true anomalies</option>`,
      )
      .join("");
  $("cluster-focus").value = Object.hasOwn(S.data.clusters, selected)
    ? selected
    : "all";
  $("cluster-summary").textContent =
    "Select a cluster to focus its sources and claimed points.";
}
function focusCluster() {
  if (!hasData()) return;
  const id = $("cluster-focus").value;
  if (id === "all") {
    $("cluster-summary").textContent =
      "All source clusters and unassigned points.";
    plots.forEach((p) => p.fit());
  } else {
    const ref = S.data.clusters[id],
      nodes = S.data.final.flatMap((owner, i) =>
        owner === Number(id) ? [i] : [],
      );
    $("cluster-summary").textContent =
      `${ref.size} fixed sources · ${ref.anomalies} true anomalies · ${nodes.length - ref.size} claims`;
    plots.forEach((p) => p.fit(true, nodes));
  }
  renderPlots();
  scheduleSave();
}
function anatomyCanvas(id) {
  const canvas = $(id),
    rect = canvas.getBoundingClientRect(),
    w = rect.width,
    h = rect.height;
  if (!w || !h) return null;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = w * ratio;
  canvas.height = h * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "10px ui-sans-serif, sans-serif";
  return { canvas, ctx, w, h };
}
function drawAnatomy() {
  if (!S.data || S.tab !== "costs") return;
  const claims = S.data.claims,
    refs = S.data.clusters,
    reach = S.data.meta.settings.reach;
  const ratios = claims.map((c) => [
    S.data.lengths[c[3]] / refs[c[1]].mean_edge,
    S.data.core[c[0]] / refs[c[1]].core_reference,
  ]);
  const histogram = anatomyCanvas("cost-histogram"),
    scatter = anatomyCanvas("cost-ratios");
  if (!histogram || !scatter) return;
  for (const chart of [histogram, scatter]) {
    chart.ctx.strokeStyle = "#edf0f6";
    for (let i = 0; i < 5; i++) {
      const y = chart.h - 35 - ((chart.h - 70) * i) / 4;
      chart.ctx.beginPath();
      chart.ctx.moveTo(45, y);
      chart.ctx.lineTo(chart.w - 18, y);
      chart.ctx.stroke();
    }
    chart.ctx.fillStyle = "#9ba5b6";
  }
  const bins = 24,
    maxCost = Math.max(reach, ...claims.map((c) => c[2]), 0.001),
    counts = [Array(bins).fill(0), Array(bins).fill(0)];
  claims.forEach(
    (c) =>
      counts[S.data.y[c[0]] < 0 ? 1 : 0][
        Math.min(bins - 1, Math.floor((c[2] / maxCost) * bins))
      ]++,
  );
  const top = Math.max(1, ...counts.flat()),
    histX = (v) => 45 + ((histogram.w - 65) * v) / maxCost,
    histY = (v) => histogram.h - 35 - ((histogram.h - 65) * v) / top;
  counts.forEach((values, kind) => {
    histogram.ctx.strokeStyle = kind ? "#d65b63" : "#16836c";
    histogram.ctx.lineWidth = 2;
    histogram.ctx.beginPath();
    values.forEach((v, i) => {
      const a = histX((i / bins) * maxCost),
        b = histX(((i + 1) / bins) * maxCost),
        y = histY(v);
      if (i === 0) histogram.ctx.moveTo(a, histY(0));
      histogram.ctx.lineTo(a, y);
      histogram.ctx.lineTo(b, y);
    });
    histogram.ctx.stroke();
  });
  histogram.ctx.strokeStyle = "#a2aec4";
  histogram.ctx.setLineDash([4, 3]);
  histogram.ctx.beginPath();
  histogram.ctx.moveTo(histX(reach), 20);
  histogram.ctx.lineTo(histX(reach), histogram.h - 30);
  histogram.ctx.stroke();
  histogram.ctx.setLineDash([]);
  for (let i = 0; i < 5; i++) {
    histogram.ctx.fillText(
      num((maxCost * i) / 4, 1),
      histX((maxCost * i) / 4) - 6,
      histogram.h - 16,
    );
    histogram.ctx.fillText(
      compact(Math.round((top * i) / 4)),
      12,
      histY((top * i) / 4) + 3,
    );
  }
  histogram.ctx.fillText(
    "Cumulative original-space cost",
    Math.max(45, histogram.w / 2 - 80),
    histogram.h - 1,
  );
  const maxX = Math.max(1, ...ratios.map((r) => r[0])) * 1.07,
    maxY = Math.max(1, ...ratios.map((r) => r[1])) * 1.07;
  const x = (v) => 45 + ((scatter.w - 65) * v) / maxX,
    y = (v) => scatter.h - 35 - ((scatter.h - 65) * v) / maxY;
  scatter.ctx.strokeStyle = "#bdc7d9";
  scatter.ctx.setLineDash([4, 3]);
  scatter.ctx.beginPath();
  scatter.ctx.moveTo(x(1), 20);
  scatter.ctx.lineTo(x(1), scatter.h - 35);
  scatter.ctx.moveTo(45, y(1));
  scatter.ctx.lineTo(scatter.w - 18, y(1));
  scatter.ctx.stroke();
  scatter.ctx.setLineDash([]);
  ratios.forEach((r, i) => {
    scatter.ctx.fillStyle =
      S.data.y[claims[i][0]] < 0 ? "#d65b6380" : "#16836c80";
    scatter.ctx.beginPath();
    scatter.ctx.arc(x(r[0]), y(r[1]), 3.3, 0, Math.PI * 2);
    scatter.ctx.fill();
  });
  scatter.ctx.fillStyle = "#9ba5b6";
  for (let i = 0; i < 5; i++) {
    scatter.ctx.fillText(
      num((maxX * i) / 4, 1),
      x((maxX * i) / 4) - 6,
      scatter.h - 16,
    );
    scatter.ctx.fillText(num((maxY * i) / 4, 1), 9, y((maxY * i) / 4) + 3);
  }
  scatter.ctx.fillText(
    "Edge length / cluster mean",
    Math.max(45, scatter.w / 2 - 72),
    scatter.h - 1,
  );
  scatter.ctx.fillText("Core / source median", 45, 12);
  if (!claims.length)
    for (const chart of [histogram, scatter])
      chart.ctx.fillText("No successful claims in this run.", 55, chart.h / 2);
  $("cost-ratios").onclick = safe(async (e) => {
    const rect = scatter.canvas.getBoundingClientRect(),
      px = e.clientX - rect.left,
      py = e.clientY - rect.top;
    let index = -1,
      distance = 120;
    ratios.forEach((r, i) => {
      const d = (x(r[0]) - px) ** 2 + (y(r[1]) - py) ** 2;
      if (d < distance) {
        index = i;
        distance = d;
      }
    });
    if (index >= 0) {
      setTab("explore");
      setFrame(stages() + index + 1);
      await inspectPoint(claims[index][0], true);
    }
  });
  const clusters = Object.entries(refs)
    .map(([id, ref]) => {
      const mine = claims.filter((c) => c[1] === Number(id)),
        anomalies = mine.filter((c) => S.data.y[c[0]] < 0).length;
      return { id, ref, normal: mine.length - anomalies, anomalies };
    })
    .sort((a, b) => b.anomalies - a.anomalies);
  $("cluster-cost-table").innerHTML =
    `<table><thead><tr><th>Source cluster</th><th>Fixed sources</th><th>Source anomalies</th><th>Normals claimed</th><th>Anomalies absorbed</th><th>Mean edge</th><th>Median core</th></tr></thead><tbody>${clusters.map((c) => `<tr><td><button data-focus-cluster="${c.id}">Cluster ${c.id} ↗</button></td><td>${c.ref.size}</td><td class="${c.ref.anomalies ? "negative" : ""}">${c.ref.anomalies} / ${c.ref.size}</td><td>${c.normal}</td><td class="${c.anomalies ? "negative" : ""}">${c.anomalies}</td><td>${num(c.ref.mean_edge, 4)}</td><td>${num(c.ref.core_reference, 4)}</td></tr>`).join("")}</tbody></table>`;
  $("cluster-cost-table")
    .querySelectorAll("[data-focus-cluster]")
    .forEach(
      (b) =>
        (b.onclick = () => {
          setTab("explore");
          $("cluster-focus").value = b.dataset.focusCluster;
          focusCluster();
        }),
    );
}
$("cluster-focus").onchange = focusCluster;
window.addEventListener("resize", drawAnatomy);
