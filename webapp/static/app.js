/* VOLTARA · EV Intelligence front end.
 * Every number shown comes from POST /api/assess (src/inference/ev_assess.py); this file
 * only collects the form, calls the API and animates the results. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
const MAX_UPLOAD = 5_000_000;
// Form start values, as in the design mock-up.
const START = { capacity: 60, soc: 80, age: 3, odometer: 45000, consumption: 16 };
const LIMITS = { capacity: [0.5, 1000, "Battery capacity"], age: [0, 40, "Vehicle age"],
                 odometer: [0, 3_000_000, "Odometer"], consumption: [1, 400, "Energy consumption"] };
const CAPTIONS = {
  sample: "A predefined dataset will be used to demonstrate model predictions.",
  upload: "The full models score the latest row of your file; earlier rows give the trend and the anomaly checks.",
  manual: "Health is estimated from age and equivalent full cycles (odometer × consumption ÷ capacity).",
};
const CLIMATE_LABELS = { Unknown: "Unknown", temperate: "Temperate", mixed: "Mixed", cold: "Cold",
                         hot_arid: "Hot & dry", tropical_humid: "Tropical" };

const S = {
  meta: null, source: "sample", dataset: "ev_degradation", unit: null,
  csv: null, fileName: null, profiles: { sample: null, upload: null },
  manual: { ...START, vehicle: "Electric car", chemistry: "Unknown", climate: "Unknown" },
  vehicle: "Electric car", chemistry: "Unknown", climate: "Unknown",
  socTouched: false, consTouched: false, lastKey: null, busy: false, first: true,
};
const F = {};   // form inputs

/* ---------------------------------------------------------------- helpers */
const fmt = (n, d = 0) => n == null || !isFinite(n) ? "—"
  : Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const icon = (id) => `<svg><use href="#i-${id}"/></svg>`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw new Error((data && data.error) || `The server answered ${res.status}.`);
  return data;
}

function toast(msg, ms = 4200) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), ms);
}

function countTo(el, to, decimals = 0, ms = 1100) {
  const from = parseFloat(el.dataset.v ?? "0") || 0;
  el.dataset.v = to;
  if (REDUCE) { el.textContent = fmt(to, decimals); return; }
  const t0 = performance.now();
  const step = (t) => {
    const k = clamp((t - t0) / ms, 0, 1), e = 1 - Math.pow(1 - k, 4);
    el.textContent = fmt(from + (to - from) * e, decimals);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* Bold the numbers in a rule-generated sentence (text is escaped first). */
const emphasise = (t) => esc(t).replace(/(-?\d[\d,]*(?:\.\d+)?\s?(?:%|km|years?|points?|cycles?|°C|kWh\/100 km)?)/g, "<b>$1</b>");

/* ---------------------------------------------------------------- build form */
function radio(name, value, label, checked) {
  return `<label class="radio"><input type="radio" name="${name}" value="${esc(value)}" ${checked ? "checked" : ""}>` +
         `<span class="dot"></span><span>${esc(label)}</span></label>`;
}

function chips(el, values, labels, current, onPick) {
  el.innerHTML = values.map((v) =>
    `<button type="button" data-v="${esc(v)}" aria-pressed="${v === current}">${esc(labels[v] || v)}</button>`).join("");
  el.onclick = (e) => {
    const b = e.target.closest("button");
    if (!b || b.disabled) return;
    $$("button", el).forEach((x) => x.setAttribute("aria-pressed", x === b));
    onPick(b.dataset.v);
  };
}
const pressChip = (el, v) => $$("button", el).forEach((x) => x.setAttribute("aria-pressed", x.dataset.v === v));

function buildForm(meta) {
  $("#vehicle-radios").innerHTML = Object.keys(meta.vehicle_types)
    .map((v) => radio("vehicle", v, v, v === S.vehicle)).join("");
  chips($("#chem-chips"), meta.chemistries, {}, S.chemistry, (v) => { S.chemistry = v; changed(); });
  chips($("#climate-chips"), meta.climates, CLIMATE_LABELS, S.climate, (v) => { S.climate = v; changed(); });

  const seg = $("#dataset-seg");
  seg.innerHTML = Object.entries(meta.datasets).map(([k, d], i) =>
    `<button type="button" role="tab" data-k="${k}" aria-selected="${k === S.dataset}" title="${esc(d.detail)}">${esc(d.label)}</button>`
  ).join("") + `<span class="thumb"></span>`;
  seg.onclick = (e) => {
    const b = e.target.closest("button");
    if (!b || b.dataset.k === S.dataset) return;
    S.dataset = b.dataset.k;
    $$("button", seg).forEach((x, i) => { x.setAttribute("aria-selected", x === b); if (x === b) seg.dataset.i = i; });
    fillUnits(true);
  };
}

function fillUnits(load) {
  const units = S.meta.datasets[S.dataset].units;
  const noun = S.dataset === "ev_degradation" ? "Battery" : "Vehicle";
  F.unit.innerHTML = units.map((u) => `<option value="${u}">${noun} ${u}</option>`).join("");
  S.unit = units[0];
  if (load) loadSample();
}

/* ---------------------------------------------------------------- SOC slider */
function paintSoc() {
  const v = Number(F.soc.value);
  $("#soc-val").textContent = v;
  F.soc.style.setProperty("--p", `${v}%`);
  const fill = $("#cell-fill");
  fill.style.width = `${Math.max(v, 3)}%`;
  fill.style.backgroundColor = v <= 15 ? "var(--bad)" : v <= 35 ? "var(--watch)" : "var(--good)";
}

/* ---------------------------------------------------------------- modes */
function saveManual() {
  if (S.source !== "manual") return;
  Object.assign(S.manual, { capacity: F.capacity.value, soc: F.soc.value, age: F.age.value,
    odometer: F.odometer.value, consumption: F.consumption.value,
    vehicle: S.vehicle, chemistry: S.chemistry, climate: S.climate });
}

function setSource(src) {
  saveManual();
  S.source = src;
  $$(".panel").forEach((p) => (p.hidden = p.dataset.for !== src));
  const cap = $("#source-caption");
  cap.style.opacity = 0;
  setTimeout(() => { cap.textContent = CAPTIONS[src]; cap.style.opacity = 1; }, 140);
  applyMode();
}

function lock(key, on, note) {
  const field = $(`.field[data-key="${key}"]`);
  field.classList.toggle("locked", on);
  F[key].disabled = on;
  const tag = $(".from-data", field);
  if (tag) tag.title = note || "";
}

function applyMode() {
  const manual = S.source === "manual";
  const p = S.profiles[S.source];
  $$("input[name=vehicle]").forEach((r) => (r.disabled = !manual));
  $(".lock-note").hidden = manual;
  $$("#chem-chips button").forEach((b) => (b.disabled = !manual));

  if (manual) {
    const m = S.manual;
    Object.assign(S, { vehicle: m.vehicle, chemistry: m.chemistry, climate: m.climate });
    F.capacity.value = m.capacity; F.age.value = m.age; F.odometer.value = m.odometer;
    F.consumption.value = m.consumption; F.soc.value = m.soc;
    ["capacity", "age", "odometer"].forEach((k) => lock(k, false));
    $$("input[name=vehicle]").forEach((r) => (r.checked = r.value === S.vehicle));
    pressChip($("#chem-chips"), S.chemistry);
    pressChip($("#climate-chips"), S.climate);
    $$("#climate-chips button").forEach((b) => (b.disabled = false));
    $("#soc-hint").textContent = ""; $("#cons-hint").textContent = "";
    $("#soc-reset").hidden = true;
  } else {
    $$("input[name=vehicle]").forEach((r) => (r.checked = r.value === "Electric car"));
    $$("#climate-chips button").forEach((b) => (b.disabled = true));
    if (p) {
      F.capacity.value = fmt(p.capacity_kwh, 1).replace(/,/g, "");
      F.age.value = fmt(p.age_years, 1).replace(/,/g, "");
      F.odometer.value = Math.round(p.odometer_km);
      lock("capacity", true, "Pack capacity in the data");
      lock("age", true, "Age at the latest reading");
      lock("odometer", true, `Estimate: ${p.odometer_note}. The models use the data's cycle count, not this number.`);
      pressChip($("#chem-chips"), p.chemistry);
      pressChip($("#climate-chips"), "");
      if (!S.socTouched) F.soc.value = Math.round(p.soc_pct ?? 80);
      if (!S.consTouched) F.consumption.value = fmt(p.consumption_kwh_100km ?? p.default_consumption_kwh_100km, 1).replace(/,/g, "");
      $("#soc-hint").textContent = S.socTouched ? "Your value is used for the range." : p.soc_note;
      $("#cons-hint").innerHTML = S.consTouched
        ? `Your value is used for the range. <button type="button" class="reset" id="cons-reset">use data value</button>`
        : esc(p.consumption_note);
      $("#soc-reset").hidden = !S.socTouched;
    } else {
      ["capacity", "age", "odometer"].forEach((k) => lock(k, true, ""));
      F.capacity.value = F.age.value = F.odometer.value = "";
      $("#soc-hint").textContent = "Upload a file to fill these fields from the data.";
      $("#cons-hint").textContent = "";
      $("#soc-reset").hidden = true;
    }
  }
  paintSoc();
  changed();
}

function setProfile(src, p) {
  S.profiles[src] = p;
  S.socTouched = S.consTouched = false;
  const el = $(`#profile-${src}`);
  const noun = p.dataset === "ev_degradation" ? "battery" : "vehicle";
  const parts = [];
  if (src === "sample") parts.push(`<span class="chip-i held" title="This ${noun} was held out before training and never seen by the models."><i></i>Held-out test ${noun}</span>`);
  else parts.push(`<span class="chip-i held"><i></i>${esc(S.fileName || "Uploaded file")}</span>`);
  parts.push(`<span class="chip-i"><b>${esc(p.vehicle)}</b></span>`,
             `<span class="chip-i">${esc(p.chemistry)}</span>`,
             `<span class="chip-i">${fmt(p.rows)} readings</span>`,
             `<span class="chip-i">${fmt(p.cycles)} cycles</span>`);
  el.innerHTML = parts.join("");
  $$(".chip-i", el).forEach((c, i) => (c.style.animationDelay = `${i * 50}ms`));
  if (S.source === src) applyMode();
}

async function loadSample() {
  const unit = S.unit, dataset = S.dataset;
  try {
    const p = await api(`/api/sample?dataset=${encodeURIComponent(dataset)}&unit=${encodeURIComponent(unit)}`);
    if (unit === S.unit && dataset === S.dataset) setProfile("sample", p);
  } catch (e) { toast(e.message); }
}

async function readFile(file) {
  if (!file) return;
  if (file.size > MAX_UPLOAD) { toast("The file is larger than 5 MB."); return; }
  const text = await file.text();
  const drop = $("#drop");
  $("#drop-title").textContent = `Reading ${file.name}…`;
  try {
    const p = await api("/api/profile", { method: "POST", body: text, headers: { "Content-Type": "text/csv" } });
    S.csv = text; S.fileName = file.name;
    drop.classList.add("has");
    $("#drop-title").innerHTML = `${esc(file.name)} · <u>replace</u>`;
    setProfile("upload", p);
  } catch (e) {
    S.csv = null; S.fileName = null; S.profiles.upload = null;
    drop.classList.remove("has");
    $("#drop-title").innerHTML = "Drop a CSV here or <u>browse</u>";
    $("#profile-upload").innerHTML = "";
    applyMode();
    toast(e.message, 7000);
  }
}

/* ---------------------------------------------------------------- request */
function body() {
  const num = (k) => (F[k].value === "" ? null : Number(F[k].value));
  if (S.source === "manual") {
    return { source: "manual", vehicle_type: S.vehicle, capacity_kwh: num("capacity"), soc_pct: Number(F.soc.value),
             age_years: num("age"), odometer_km: num("odometer"), consumption_kwh_100km: num("consumption"),
             chemistry: S.chemistry, climate: S.climate };
  }
  const b = { source: S.source, soc_pct: S.socTouched ? Number(F.soc.value) : null,
              consumption_kwh_100km: S.consTouched ? num("consumption") : null };
  if (S.source === "sample") Object.assign(b, { dataset: S.dataset, unit: S.unit });
  else b.csv = S.csv;
  return b;
}
const keyOf = (b) => JSON.stringify({ ...b, csv: b.csv ? `${S.fileName}:${b.csv.length}` : undefined });

function validate() {
  $$(".pill").forEach((x) => x.classList.remove("invalid"));
  if (S.source === "upload" && !S.csv) return "Upload a CSV first, or pick a sample.";
  const keys = S.source === "manual" ? Object.keys(LIMITS) : (S.consTouched ? ["consumption"] : []);
  const errs = [];
  for (const k of keys) {
    const [lo, hi, name] = LIMITS[k];
    const v = F[k].value === "" ? NaN : Number(F[k].value);
    if (!(v >= lo && v <= hi)) { errs.push(`${name} must be between ${fmt(lo, lo % 1 ? 1 : 0)} and ${fmt(hi)}.`); F[k].classList.add("invalid"); }
  }
  return errs.join(" ");
}

function changed() {
  const has = S.lastKey !== null;
  const stale = has && keyOf(body()) !== S.lastKey;
  $("#stale").hidden = !stale;
  $("#go").classList.toggle("nudge", stale);
}

async function submit(e) {
  e.preventDefault();
  if (S.busy) return;
  const err = validate();
  const fe = $("#form-error");
  fe.hidden = !err; fe.textContent = err;
  if (err) return;
  const b = body();
  S.busy = true;
  const go = $("#go");
  go.classList.add("loading"); go.classList.remove("nudge");
  $(".cta-text", go).textContent = "Running the models…";
  $$(".tile").forEach((t) => { t.classList.remove("enter"); t.classList.add("loading"); });
  try {
    const [res] = await Promise.all([
      api("/api/assess", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) }),
      sleep(REDUCE ? 0 : 700),
    ]);
    S.lastKey = keyOf(b);
    render(res, b);
    if (S.first && innerWidth < 980) $("#out").scrollIntoView({ behavior: REDUCE ? "auto" : "smooth", block: "start" });
    S.first = false;
  } catch (e2) {
    fe.hidden = false; fe.textContent = e2.message;
    $$(".tile").forEach((t) => t.classList.remove("loading"));
  } finally {
    S.busy = false;
    go.classList.remove("loading");
    $(".cta-text", go).textContent = "Preview expected output";
    changed();
  }
}

/* ---------------------------------------------------------------- render */
function setTile(id, { value, unit = "", decimals = 0, word = false, cap }) {
  const t = $(id);
  const num = $(".num", t);
  t.classList.remove("loading", "placeholder-state");
  num.classList.toggle("word", word);
  if (typeof value === "number" && isFinite(value)) countTo(num, value, decimals);
  else { num.dataset.v = 0; num.textContent = value; }
  $(".unit", t).textContent = unit;
  $(".t-cap", t).innerHTML = cap;
}

function sohColor(soh, eol) {
  return soh <= eol ? "var(--bad)" : soh <= eol + 5 ? "var(--watch)" : "var(--good)";
}

function render(res, b) {
  const a = res.assessment, p = res.profile, r = res.range;
  const eol = S.meta.eol_soh_pct;
  const rows = a.history ? a.history.length : 0;

  // Header
  const noun = p && p.dataset === "ev_degradation" ? "battery" : "vehicle";
  $("#out-sub").textContent = b.source === "manual"
    ? `Manual entry · ${b.vehicle_type} · ${fmt(b.capacity_kwh, 1)} kWh · ${fmt(b.odometer_km)} km`
    : `${b.source === "sample" ? `Held-out test ${noun}` : "Uploaded file"} ${p.unit} · ${p.vehicle} · ${fmt(rows)} readings`;
  const risk = $("#risk");
  risk.hidden = !a.risk;
  if (a.risk) { risk.className = `risk ${a.risk}`; risk.textContent = `${a.risk} risk`; }

  // SOH
  const si = a.soh_interval;
  const capBits = [si ? `Estimated model output · 90% range <b>${fmt(si[0], 1)}–${fmt(si[1], 1)}%</b>`
                      : `Estimated model output · typical error ±${fmt(a.soh_error_pct, 1)} pts`];
  if (a.soh_measured_pct != null) capBits.push(`recorded in data: <b>${fmt(a.soh_measured_pct, 1)}%</b>`);
  setTile("#t-soh", { value: a.soh_pct, decimals: 1, unit: "%", cap: capBits.join(" · ") });
  const sm = $("#t-soh .meter i");
  sm.style.width = `${clamp((a.soh_pct - 50) / 50, 0, 1) * 100}%`;
  sm.style.backgroundColor = sohColor(a.soh_pct, eol);
  $("#t-soh .eol").style.left = `${(eol - 50) / 50 * 100}%`;
  const band = $("#t-soh .band"), pos = (v) => clamp((v - 50) / 50, 0, 1) * 100;
  band.hidden = !si;
  if (si) { band.style.left = `${pos(si[0])}%`; band.style.width = `${pos(si[1]) - pos(si[0])}%`; band.title = `90% range ${fmt(si[0], 1)}–${fmt(si[1], 1)}%`; }

  // Range
  if (a.range_km != null && r) {
    setTile("#t-range", { value: a.range_km, unit: "km",
      cap: `At ${fmt(r.soc_pct)}% charge and ${fmt(r.consumption_kwh_100km, 1)} kWh/100 km · full charge ${fmt(a.full_range_km)} km` });
    $("#t-range .meter s").style.width = `${clamp(a.full_range_km / r.new_range_km, 0, 1) * 100}%`;
    $("#t-range .meter i").style.width = `${clamp(a.range_km / r.new_range_km, 0, 1) * 100}%`;
    $("#t-range .meter").title = `Now ${fmt(a.range_km)} km · full charge ${fmt(a.full_range_km)} km · when new ${fmt(r.new_range_km)} km`;
  } else {
    setTile("#t-range", { value: "—", cap: "Needs capacity and consumption" });
  }

  // RUL
  if (a.rul_cycles == null) {
    setTile("#t-rul", { value: "Needs chemistry", word: true,
      cap: "Pick the cell chemistry under <i>Improve the estimate</i>" });
  } else if (a.soh_pct <= eol) {
    setTile("#t-rul", { value: "End of life", word: true, cap: `SOH at or below ${eol}%: plan a capacity test` });
  } else {
    const yrs = a.rul_years != null ? `≈ ${fmt(a.rul_years, 1)} years at this pace · ` : "";
    const ri = a.rul_interval;
    setTile("#t-rul", { value: a.rul_cycles, unit: "cycles",
      cap: `${yrs}${ri ? `90% range <b>${fmt(ri[0])}–${fmt(ri[1])}</b>` : `±${fmt(a.rul_error_cycles)}`} cycles to ${eol}% SOH` });
  }

  // Anomalies
  const pulse = $("#t-anom .pulse");
  const ta = $("#t-anom");
  if (!a.anomaly_assessed) {
    setTile("#t-anom", { value: "Not assessed", word: true, cap: "Needs a history of readings: use a sample or upload a CSV" });
    pulse.className = "pulse on"; ta.classList.remove("alert");
  } else if (a.anomalies.length) {
    setTile("#t-anom", { value: "Detected", word: true,
      cap: `${fmt(a.anomalies.length)} finding${a.anomalies.length > 1 ? "s" : ""} in ${fmt(rows)} readings` });
    pulse.className = "pulse on bad"; ta.classList.add("alert");
  } else {
    setTile("#t-anom", { value: "None", word: true, cap: `Based on ${fmt(rows)} available readings` });
    pulse.className = "pulse on good"; ta.classList.remove("alert");
  }

  $$(".tile").forEach((t, i) => {
    t.classList.remove("enter"); void t.offsetWidth;
    t.style.animationDelay = `${i * 90}ms`; t.classList.add("enter");
  });

  // Insights and warnings
  const ul = $("#insights");
  ul.classList.remove("placeholder");
  ul.innerHTML = a.insights.map((x) => `<li>${emphasise(x)}</li>`).join("");
  $$("li", ul).forEach((li, i) => { li.style.animationDelay = `${350 + i * 110}ms`; li.classList.add("enter"); });
  const notes = [...a.warnings];
  if (a.notes && a.notes.rul_note) notes.push(a.notes.rul_note);
  $("#warnings").innerHTML = notes.map((w, i) =>
    `<p class="warn" style="animation-delay:${600 + i * 90}ms">${icon("alert")}<span>${esc(w)}</span></p>`).join("");

  renderDeep(a, p, r, b);
  printInputs(a, p, r, b);
  $("#report").hidden = false;
}

/* Inputs block shown only on the printed / PDF report. */
function printInputs(a, p, r, b) {
  $("#print-date").textContent = new Date().toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" });
  const rows = b.source === "manual"
    ? [["Source", "Manual entry"], ["Vehicle", b.vehicle_type], ["Capacity", `${fmt(b.capacity_kwh, 1)} kWh`],
       ["Age", `${fmt(b.age_years, 1)} years`], ["Odometer", `${fmt(b.odometer_km)} km`],
       ["Consumption", `${fmt(b.consumption_kwh_100km, 1)} kWh/100 km`], ["Charge", `${fmt(b.soc_pct)} %`],
       ["Chemistry", b.chemistry], ["Climate", CLIMATE_LABELS[b.climate] || b.climate]]
    : [["Source", b.source === "sample" ? "Held-out sample" : `Upload: ${S.fileName}`], ["Unit", p.unit],
       ["Vehicle", p.vehicle], ["Chemistry", p.chemistry], ["Capacity", `${fmt(p.capacity_kwh, 1)} kWh`],
       ["Readings", fmt(p.rows)], ["Age", `${fmt(p.age_years, 1)} years`],
       ["Charge (range)", r ? `${fmt(r.soc_pct)} %` : "—"],
       ["Consumption (range)", r ? `${fmt(r.consumption_kwh_100km, 1)} kWh/100 km` : "—"]];
  $("#print-inputs").innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
}

function renderDeep(a, p, r, b) {
  const deep = $("#deep");
  const wasHidden = deep.hidden;
  deep.hidden = false;
  const eol = S.meta.eol_soh_pct;

  // Gauge
  const bar = $("#g-bar"), C = 2 * Math.PI * 82;
  bar.style.strokeDasharray = C;
  if (wasHidden) { bar.style.transition = "none"; bar.style.strokeDashoffset = C; void bar.getBoundingClientRect(); bar.style.transition = ""; }
  requestAnimationFrame(() => (bar.style.strokeDashoffset = C * (1 - clamp(a.soh_pct, 0, 100) / 100)));
  document.documentElement.style.setProperty("--ring-a", "var(--ink)");
  document.documentElement.style.setProperty("--ring-b", a.soh_pct <= eol ? "var(--bad)" : a.soh_pct <= eol + 5 ? "var(--watch)" : "var(--good)");
  const mark = $("#g-eol");
  mark.setAttribute("x1", 168); mark.setAttribute("x2", 192); mark.setAttribute("y1", 100); mark.setAttribute("y2", 100);
  mark.setAttribute("transform", `rotate(${eol / 100 * 360} 100 100)`);
  countTo($("#g-num"), a.soh_pct, 1, 1500);

  const kv = [["Model SOH", `${fmt(a.soh_pct, 2)} %`]];
  if (a.soh_measured_pct != null) {
    kv.push(["Recorded SOH", `${fmt(a.soh_measured_pct, 2)} %`],
            ["Difference", `${a.soh_pct - a.soh_measured_pct >= 0 ? "+" : ""}${fmt(a.soh_pct - a.soh_measured_pct, 2)} pts`]);
  }
  if (a.soh_interval) kv.push(["90% range", `${fmt(a.soh_interval[0], 1)}–${fmt(a.soh_interval[1], 1)} %`]);
  kv.push(["Typical error", `±${fmt(a.soh_error_pct, 2)} pts`]);
  const efc = a.notes && a.notes.equivalent_full_cycles;
  if (efc != null) kv.push(["Equivalent full cycles", fmt(efc)]);
  kv.push(["Source", b.source === "manual" ? "Quick models" : "Full models"]);
  $("#kv").innerHTML = kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");

  // Trend chart, or usable energy for a single reading
  const hist = (a.history || []).filter((h) => h.age_years != null && h.soh_pred != null);
  if (hist.length > 1) {
    $("#chart-title").textContent = "Battery degradation trend";
    const hasMeas = hist.some((h) => h.soh_measured != null);
    $("#legend").innerHTML = `<span><i></i>Model estimate</span>` +
      (hasMeas ? `<span><i class="dotk"></i>Recorded in data</span>` : "") + `<span><i class="dash"></i>End of life ${eol}%</span>`;
    drawChart(hist, eol);
  } else {
    $("#chart-title").textContent = "Usable battery energy";
    $("#legend").innerHTML = "";
    drawEnergy(a, r);
  }

  // Findings
  const items = [];
  if (!a.anomaly_assessed) {
    items.push(`<li class="na">${icon("shield")}Anomaly checks need a history of readings. Use a sample battery or upload a CSV.</li>`);
  } else if (!a.anomalies.length) {
    items.push(`<li class="ok">${icon("check")}No abnormal temperature, voltage, resistance or health jumps in ${fmt((a.history || []).length)} readings.</li>`);
  } else {
    a.anomalies.forEach((m, i) => items.push(`<li class="bad" style="animation-delay:${i * 60}ms">${icon("alert")}${esc(m[0].toUpperCase() + m.slice(1))}</li>`));
  }
  $("#findings").innerHTML = items.join("");
}

function drawEnergy(a, r) {
  const el = $("#chart");
  el.classList.remove("drawn", "hover");
  if (!r) { el.innerHTML = `<p class="empty-note">Not available.</p>`; return; }
  const cap = r.capacity_kwh, now = cap * a.soh_pct / 100, avail = now * r.soc_pct / 100;
  const row = (label, v, cls) => `<div class="energy-row"><header><span>${label}</span><b>${fmt(v, 1)} kWh</b></header>` +
    `<div class="bar ${cls}"><i data-w="${clamp(v / cap, 0, 1) * 100}"></i></div></div>`;
  el.innerHTML = `<div class="energy">${row("When new", cap, "soft")}${row(`Now, at ${fmt(a.soh_pct, 1)}% SOH`, now, "")}` +
    `${row(`Available at ${fmt(r.soc_pct)}% charge`, avail, "volt")}</div>` +
    `<p class="empty-note">A single reading has no trend. Use a sample or upload a CSV to see how health changed over time.</p>`;
  requestAnimationFrame(() => requestAnimationFrame(() => $$(".bar i", el).forEach((i, k) => {
    i.style.transitionDelay = `${k * 120}ms`; i.style.width = `${i.dataset.w}%`;
  })));
}

function drawChart(hist, eol) {
  const el = $("#chart");
  const W = 640, H = 270, m = { l: 46, r: 18, t: 16, b: 34 };
  const pts = [...hist].sort((x, y) => x.age_years - y.age_years);
  const xsAll = pts.map((p) => p.age_years);
  const ysAll = pts.flatMap((p) => [p.soh_pred, p.soh_measured].filter((v) => v != null));
  let x0 = Math.min(...xsAll), x1 = Math.max(...xsAll);
  if (x1 - x0 < 0.05) { x0 -= 0.05; x1 += 0.05; }
  let y0 = Math.min(...ysAll), y1 = Math.max(...ysAll);
  if (y0 < eol + 8) y0 = Math.min(y0, eol);
  y0 = Math.floor(y0 - 1); y1 = Math.min(100.5, Math.ceil(y1 + 0.6));
  if (y1 - y0 < 3) y0 = y1 - 3;
  const X = (v) => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = (v) => m.t + (y1 - v) / (y1 - y0) * (H - m.t - m.b);
  const ticks = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
  const yt = ticks(y0, y1, 5), xt = ticks(x0, x1, 5);
  const dx = x1 - x0 < 1 ? 2 : 1;
  const line = pts.map((p, i) => `${i ? "L" : "M"}${X(p.age_years).toFixed(1)},${Y(p.soh_pred).toFixed(1)}`).join("");
  const area = `${line}L${X(pts.at(-1).age_years).toFixed(1)},${H - m.b}L${X(pts[0].age_years).toFixed(1)},${H - m.b}Z`;
  const eolIn = eol >= y0 && eol <= y1;
  el.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Predicted state of health against battery age">
      <defs><linearGradient id="g-area" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" style="stop-color:var(--ink)" stop-opacity=".12"/><stop offset="1" style="stop-color:var(--ink)" stop-opacity="0"/>
      </linearGradient></defs>
      ${yt.map((v) => `<line class="grid-l" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/>
        <text class="ax" x="${m.l - 10}" y="${Y(v) + 4}" text-anchor="end">${fmt(v, y1 - y0 < 6 ? 1 : 0)}%</text>`).join("")}
      ${xt.map((v) => `<text class="ax" x="${X(v)}" y="${H - 10}" text-anchor="middle">${fmt(v, dx)}y</text>`).join("")}
      ${eolIn ? `<line class="eol-l" x1="${m.l}" x2="${W - m.r}" y1="${Y(eol)}" y2="${Y(eol)}"/>
        <text class="eol-t" x="${W - m.r}" y="${Y(eol) - 6}" text-anchor="end">end of life ${eol}%</text>` : ""}
      <path class="area" d="${area}"/>
      <path class="line" d="${line}"/>
      ${pts.filter((p) => p.soh_measured != null).map((p) =>
        `<circle class="meas" cx="${X(p.age_years)}" cy="${Y(p.soh_measured)}" r="3.5"/>`).join("")}
      <line class="cross" y1="${m.t}" y2="${H - m.b}"/>
      <circle class="hot" r="5.5"/>
      <rect x="${m.l}" y="0" width="${W - m.l - m.r}" height="${H}" fill="transparent" class="hit"/>
    </svg><div class="tip"></div>`;
  if (!eolIn) $("#legend").lastElementChild.remove();

  const path = $(".line", el);
  const len = path.getTotalLength();
  el.classList.remove("drawn");
  if (!REDUCE) {
    path.style.strokeDasharray = len; path.style.strokeDashoffset = len;
    path.getBoundingClientRect();
    path.style.transition = "stroke-dashoffset 1.6s cubic-bezier(.22,1,.36,1)";
    requestAnimationFrame(() => (path.style.strokeDashoffset = 0));
  }
  setTimeout(() => el.classList.add("drawn"), REDUCE ? 0 : 500);

  const svg = $("svg", el), tip = $(".tip", el), cross = $(".cross", el), hot = $(".hot", el);
  const move = (ev) => {
    const box = svg.getBoundingClientRect();
    const vx = (ev.clientX - box.left) / box.width * W;
    let best = 0;
    pts.forEach((p, i) => { if (Math.abs(X(p.age_years) - vx) < Math.abs(X(pts[best].age_years) - vx)) best = i; });
    const p = pts[best], cx = X(p.age_years), cy = Y(p.soh_pred);
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx);
    hot.setAttribute("cx", cx); hot.setAttribute("cy", cy);
    tip.style.left = `${cx / W * box.width}px`; tip.style.top = `${cy / H * box.height}px`;
    tip.innerHTML = `${fmt(p.age_years, 2)} years<br>model ${fmt(p.soh_pred, 1)}%` +
      (p.soh_measured != null ? `<br>recorded ${fmt(p.soh_measured, 1)}%` : "") +
      (p.rul_pred != null ? `<br>RUL ${fmt(p.rul_pred)} cycles` : "");
    el.classList.add("hover");
  };
  svg.addEventListener("pointermove", move);
  svg.addEventListener("pointerleave", () => el.classList.remove("hover"));
}

/* ---------------------------------------------------------------- ambience */
/* Cursor light on the dot grid, eased so it trails the pointer. */
function cursorLight() {
  const bg = $(".bg");
  if (REDUCE || matchMedia("(hover: none)").matches) return;
  let x = innerWidth / 2, y = innerHeight / 3, tx = x, ty = y, running = false;
  const step = () => {
    x += (tx - x) * 0.12; y += (ty - y) * 0.12;
    bg.style.setProperty("--cx", `${x}px`); bg.style.setProperty("--cy", `${y}px`);
    running = Math.abs(tx - x) > 0.5 || Math.abs(ty - y) > 0.5;
    if (running) requestAnimationFrame(step);
  };
  addEventListener("pointermove", (e) => {
    tx = e.clientX; ty = e.clientY; bg.classList.add("active");
    if (!running) { running = true; requestAnimationFrame(step); }
  }, { passive: true });
  document.addEventListener("pointerleave", () => bg.classList.remove("active"));
}

/* Light by default; dark only when the visitor picks it (remembered in this browser). */
function themeToggle() {
  const root = document.documentElement;
  const dark = () => root.dataset.theme === "dark";
  const sync = () => (root.dataset.theme = dark() ? "dark" : "light");
  sync();
  $("#theme").addEventListener("click", () => {
    const next = dark() ? "light" : "dark";
    root.classList.add("theming");
    root.dataset.theme = next;
    try { localStorage.setItem("voltara-theme", next); } catch { /* private mode */ }
    setTimeout(() => root.classList.remove("theming"), 600);
  });
}

function ambience() {
  themeToggle();
  cursorLight();
  $$(".card").forEach((card) => card.addEventListener("pointermove", (e) => {
    const b = card.getBoundingClientRect();
    card.style.setProperty("--mx", `${e.clientX - b.left}px`);
    card.style.setProperty("--my", `${e.clientY - b.top}px`);
  }));
  const io = new IntersectionObserver((es) => es.forEach((x) => x.isIntersecting && (x.target.classList.add("in"), io.unobserve(x.target))),
    { threshold: 0.08 });
  $$(".reveal").forEach((el) => io.observe(el));
}

/* ---------------------------------------------------------------- init */
async function init() {
  ["capacity", "soc", "age", "odometer", "consumption", "unit"].forEach((k) => (F[k] = $(`#${k}`)));
  ambience();
  paintSoc();
  $$(".tile").forEach((t) => t.classList.add("placeholder-state"));

  const status = $("#status");
  if (window.VOLTARA_PAGES_PREVIEW) {
    status.className = "status";
    $("span", status).textContent = "Frontend preview";
    $$("input, select, button", $("#form")).forEach((el) => { el.disabled = true; });
    $("#form").addEventListener("submit", (event) => event.preventDefault());
    $("#go").textContent = "Requires the Python application";
    $("#out-sub").textContent = "No assessment has been run. Live predictions require the Python server.";
    $("#tiles").hidden = true;
    $("#tiles").style.display = "none";
    $("#insights").hidden = true;
    $("#hero-stats").hidden = true;
    $("#hero-stats").style.display = "none";
    return;
  }
  try {
    S.meta = await api("/api/meta");
    status.className = "status ok";
    $("span", status).textContent = "Models ready";
  } catch (e) {
    status.className = "status err";
    $("span", status).textContent = "Models unavailable";
    toast(`${e.message} Run python -m src.models.train_ev and reload.`, 10000);
    return;
  }
  $$("[data-stat]").forEach((b) => (b.textContent = `±${fmt(S.meta.errors[b.dataset.stat], 2)}`));
  buildForm(S.meta);
  fillUnits(false);

  // Events
  $("#form").addEventListener("submit", submit);
  $$("input[name=source]").forEach((r) => r.addEventListener("change", () => setSource(r.value)));
  $("#vehicle-radios").addEventListener("change", (e) => {
    if (e.target.name !== "vehicle") return;
    S.vehicle = e.target.value;
    const d = S.meta.vehicle_types[S.vehicle];
    F.capacity.value = d.capacity_kwh; F.consumption.value = d.consumption_kwh_100km;
    [F.capacity, F.consumption].forEach((x) => { x.classList.remove("invalid"); x.classList.remove("flash"); void x.offsetWidth; x.classList.add("flash"); });
    changed();
  });
  F.soc.addEventListener("input", () => {
    paintSoc();
    if (S.source !== "manual" && !S.socTouched) { S.socTouched = true; $("#soc-reset").hidden = false; $("#soc-hint").textContent = "Your value is used for the range."; }
    changed();
  });
  $("#soc-reset").addEventListener("click", () => { S.socTouched = false; applyMode(); });
  F.consumption.addEventListener("input", () => {
    F.consumption.classList.remove("invalid");
    if (S.source !== "manual" && !S.consTouched) {
      S.consTouched = true;
      $("#cons-hint").innerHTML = `Your value is used for the range. <button type="button" class="reset" id="cons-reset">use data value</button>`;
    }
    changed();
  });
  $("#cons-hint").addEventListener("click", (e) => { if (e.target.id === "cons-reset") { S.consTouched = false; applyMode(); } });
  ["capacity", "age", "odometer"].forEach((k) => F[k].addEventListener("input", () => { F[k].classList.remove("invalid"); changed(); }));
  F.unit.addEventListener("change", () => { S.unit = F.unit.value; loadSample(); });
  $("#shuffle").addEventListener("click", () => {
    const units = S.meta.datasets[S.dataset].units;
    let u = S.unit;
    while (units.length > 1 && u === S.unit) u = units[Math.floor(Math.random() * units.length)];
    F.unit.value = S.unit = u;
    loadSample();
  });
  $("#report").addEventListener("click", () => window.print());
  $("#file").addEventListener("change", (e) => readFile(e.target.files[0]));
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => readFile(e.dataTransfer.files[0]));

  applyMode();
  await loadSample();
  const q = new URLSearchParams(location.search);
  if (["manual", "sample"].includes(q.get("source"))) {
    $(`input[name=source][value=${q.get("source")}]`).checked = true;
    setSource(q.get("source"));
  }
  if (q.has("run")) $("#form").requestSubmit();
}

document.addEventListener("DOMContentLoaded", init);
