/* EchoLoop front end. Talks to backend/main.py; every ID here exists in index.html. */
"use strict";

const MODES = [
  ["autism_neurodivergent", "Autism / neurodivergent", "Communication support"],
  ["speech_language", "Speech difficulty", "Speaking can be challenging"],
  ["aphasia_word_finding", "Aphasia", "Support with word-finding"],
  ["cognitive_fatigue", "Cognitive fatigue", "Support during low energy"],
  ["aac", "AAC support", "Alternative communication"],
  ["other", "Other", "Something else"],
  ["prefer_not_to_say", "Prefer not to say", "Just the basics"],
];
const TITLES = {
  live: ["Live communication", "A little context. A clearer connection."],
  onboarding: ["Onboarding", "Tell us how we can best support your communication."],
  hist: ["Learning history", "Your past conversations help me better understand you."],
  trace: ["Agent trace", "What each agent observed and decided."],
  eval: ["Evaluation", "Track progress and see how communication improves over time."],
  settings: ["Settings", "Customize your experience."],
};
const ICONS = {
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/>',
  video: '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3z"/>',
  history: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  loop: '<path d="M4 12a8 8 0 0 1 14-5l2 2M20 12a8 8 0 0 1-14 5l-2-2"/><path d="M18 3v4h-4M6 21v-4h4"/>',
  chart: '<path d="M4 20V10M10 20V4M16 20v-8M22 20H2"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M19 12a7 7 0 0 0-.1-1l2-1.5-2-3.5-2.3.9a7 7 0 0 0-1.7-1L14.5 3h-5l-.4 2.4a7 7 0 0 0-1.7 1L5.1 5.5l-2 3.5 2 1.5a7 7 0 0 0 0 2l-2 1.5 2 3.5 2.3-.9a7 7 0 0 0 1.7 1l.4 2.4h5l.4-2.4a7 7 0 0 0 1.7-1l2.3.9 2-3.5-2-1.5a7 7 0 0 0 .1-1z"/>',
  heart: '<path d="M12 21s-8-5.3-8-11a4.5 4.5 0 0 1 8-2.8A4.5 4.5 0 0 1 20 10c0 5.7-8 11-8 11z"/>',
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
  spark: '<path d="M12 3l2 6 6 2-6 2-2 6-2-6-6-2 6-2z"/>',
  shield: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
  wave: '<path d="M3 12h2l2-6 3 12 3-9 2 6 2-3h4"/>',
  eye: '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  box: '<path d="M12 3l9 4.5v9L12 21l-9-4.5v-9z"/><path d="M3 7.5l9 4.5 9-4.5M12 12v9"/>',
  hand: '<path d="M8 13V6a1.5 1.5 0 0 1 3 0v6M11 12V4a1.5 1.5 0 0 1 3 0v8M14 12V6a1.5 1.5 0 0 1 3 0v9a6 6 0 0 1-12 0v-3a1.5 1.5 0 0 1 3 0"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-8 8"/>',
  message: '<path d="M4 5h16v11H9l-5 4z"/>',
  check: '<path d="M4 12l5 5L20 6"/>',
  edit: '<path d="M4 20h4l11-11-4-4L4 16z"/>',
  bulb: '<path d="M9 18h6M10 21h4M8 13a5 5 0 1 1 8 0c-1 1-1 2-1 3H9c0-1 0-2-1-3z"/>',
  database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  volume: '<path d="M4 9v6h4l5 4V5L8 9z"/><path d="M16 9a4 4 0 0 1 0 6M18.5 6.5a8 8 0 0 1 0 11"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
};

const $ = (id) => document.getElementById(id);
const api = (path, body) =>
  fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {})
    .then(async (r) => (r.ok ? r.json() : Promise.reject(await r.json().catch(() => ({ detail: r.statusText })))));

let USER = localStorage.getItem("echoloop_user") || "user_01";
let profile = null, quickPhrases = [], last = null, selected = 0, stream = null, rec = null, audio = null, confirmed = null;

document.querySelectorAll("[data-icon]").forEach((el) => {
  const d = ICONS[el.dataset.icon];
  if (d) el.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${d}</svg>`;
});

function notify(msg, ms = 4000) {
  const n = $("notice"); n.textContent = msg; n.classList.remove("hidden");
  clearTimeout(n._t); n._t = setTimeout(() => n.classList.add("hidden"), ms);
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x) => `${Math.round(x * 100)}%`;

// ---------------- navigation
function showTab(tab) {
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("hidden", v.id !== `tab-${tab}`));
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  const [t, s] = TITLES[tab] || TITLES.live;
  $("pageTitle").textContent = t; $("pageSubtitle").textContent = s;
  if (tab === "hist") loadHistory();
  if (tab === "eval") loadEval();
  if (tab === "settings") fillSettings();
  location.hash = tab;
}
document.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---------------- profile / onboarding
$("modes").innerHTML = MODES.map(([k, t, s]) =>
  `<button type="button" class="mode" data-mode="${k}"><strong>${t}</strong><small>${s}</small></button>`).join("");
document.querySelectorAll(".mode").forEach((b) => b.addEventListener("click", async () => {
  USER = $("userId").value.trim() || "user_01";
  localStorage.setItem("echoloop_user", USER);
  localStorage.setItem("echoloop_onboarded", "1");
  const r = await api(`/api/onboarding/${USER}`, { support_mode: b.dataset.mode });
  applyProfile(r.profile, r.quick_phrases);
  notify("Preferences saved. This is your choice, never a diagnosis.");
  showTab("live");
}));

function applyProfile(p, quick) {
  profile = p; quickPhrases = quick || [];
  const m = MODES.find((x) => x[0] === p.support_mode);
  $("modeBadge").textContent = m ? m[1] : "Personal communication";
  $("userName").textContent = USER; $("avatar").textContent = USER[0].toUpperCase();
  document.querySelectorAll(".mode").forEach((b) => b.classList.toggle("selected", b.dataset.mode === p.support_mode));
  $("quick").innerHTML = (p.quick_phrases_enabled ? quickPhrases : []).map((q) => `<button class="quick">${esc(q)}</button>`).join("");
  document.querySelectorAll(".quick").forEach((b) => b.addEventListener("click", () => { $("fragment").value = b.textContent; process(); }));
  $("hint").textContent = p.pause_tolerance === "long" ? "Take your time. Long pauses are fine." : "Take your time. We're listening.";
  if (p.camera_enabled && !stream) startCamera();
}

async function saveProfile(patch) {
  profile = { ...profile, ...patch };
  await api(`/api/profile/${USER}`, profile);
}

// ---------------- settings
function fillSettings() {
  if (!profile) return;
  $("pauseSetting").value = profile.pause_tolerance;
  $("countSetting").value = String(profile.suggestion_count);
  $("quickSetting").checked = profile.quick_phrases_enabled;
  $("literalSetting").checked = profile.literal_language;
  $("voiceSpeed").value = localStorage.getItem("echoloop_rate") || "1";
}
$("settingsForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  localStorage.setItem("echoloop_rate", $("voiceSpeed").value);
  await saveProfile({
    pause_tolerance: $("pauseSetting").value,
    suggestion_count: +$("countSetting").value,
    quick_phrases_enabled: $("quickSetting").checked,
    literal_language: $("literalSetting").checked,
  });
  applyProfile(profile, quickPhrases);
  $("settingsStatus").textContent = "Saved.";
});

// ---------------- camera: live object detection + hand pointing, in the browser.
// COCO-SSD boxes objects (the "person" class is discarded, never drawn, never sent).
// MediaPipe Hands gives an index-finger ray; the object it hits is the pointing target.
// Frames never leave the browser unless a server vision key is configured.
const CDN = {
  tf: "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js",
  coco: "https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js",
  hands: "https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/hands.js",
};
let detector = null, hands = null, overlay = null, detectTimer = null, detecting = false;
let live = { objects: [], pointing: null, fingertip: null, ray: null };

const loadScript = (src) => new Promise((ok, fail) => {
  if (document.querySelector(`script[src="${src}"]`)) return ok();
  const el = document.createElement("script"); el.src = src; el.onload = ok; el.onerror = () => fail(new Error("failed " + src));
  document.head.appendChild(el);
});

async function loadVision() {
  if (detector) return;
  $("sceneSummary").textContent = "Loading object detector…";
  await loadScript(CDN.tf); await loadScript(CDN.coco);
  detector = await cocoSsd.load({ base: "lite_mobilenet_v2" });
  try {
    await loadScript(CDN.hands);
    hands = new Hands({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${f}` });
    hands.setOptions({ maxNumHands: 1, modelComplexity: 0, minDetectionConfidence: 0.6, minTrackingConfidence: 0.5 });
    hands.onResults(onHands);
  } catch (e) { hands = null; notify("Pointing detection unavailable; objects only."); }
}

function ensureOverlay() {
  if (overlay) return overlay;
  overlay = document.createElement("canvas"); overlay.className = "overlay";
  document.querySelector(".camera-panel").appendChild(overlay);
  return overlay;
}

function onHands(res) {
  const lm = res.multiHandLandmarks && res.multiHandLandmarks[0];
  if (!lm) { live.ray = null; live.fingertip = null; return; }
  const v = $("video"), W = v.videoWidth, H = v.videoHeight;
  const tip = lm[8], base = lm[5];                     // index fingertip, index knuckle
  const extended = Math.hypot(tip.x - lm[0].x, tip.y - lm[0].y) > 1.15 * Math.hypot(lm[12].x - lm[0].x, lm[12].y - lm[0].y);
  if (!extended) { live.ray = null; live.fingertip = null; return; }
  live.fingertip = [tip.x * W, tip.y * H];
  live.ray = [(tip.x - base.x) * W, (tip.y - base.y) * H];
}

function pointedObject(objects) {
  if (!live.fingertip || !live.ray) return null;
  const [fx, fy] = live.fingertip, [dx, dy] = live.ray, n = Math.hypot(dx, dy) || 1;
  let best = null, bestScore = 0;
  for (const o of objects) {
    const [x, y, w, h] = o.bbox, cx = x + w / 2, cy = y + h / 2;
    const vx = cx - fx, vy = cy - fy, dist = Math.hypot(vx, vy) || 1;
    const cos = (vx * dx + vy * dy) / (dist * n);              // alignment with finger ray
    const inside = fx >= x && fx <= x + w && fy >= y && fy <= y + h;
    const score = inside ? 2 : cos > 0.75 ? cos - dist / 4000 : 0;
    if (score > bestScore) { bestScore = score; best = o; }
  }
  return best;
}

async function detectLoop() {
  const v = $("video");
  if (!stream || !detector || !v.videoWidth || detecting) return;
  detecting = true;
  try {
    const raw = await detector.detect(v, 8, 0.45);
    const objects = raw.filter((o) => o.class !== "person");   // never box or report people
    if (hands) await hands.send({ image: v });
    const target = pointedObject(objects);
    live.objects = objects.map((o) => ({ label: o.class, confidence: +o.score.toFixed(2), bbox: o.bbox }));
    live.pointing = target ? target.class : null;
    drawOverlay(objects, target);
    const labels = [...new Set(live.objects.map((o) => o.label))];
    $("objectsSummary").textContent = labels.length ? labels.join(", ") : "No objects detected yet";
    $("gestureSummary").textContent = live.pointing ? `pointing at ${live.pointing}` : live.fingertip ? "hand raised, not at an object" : "No pointing detected";
    $("sceneSummary").textContent = `Live · ${labels.length} object type(s) · on-device detection`;
    renderObjectTags(live.objects, live.pointing);
  } finally { detecting = false; }
}

function drawOverlay(objects, target) {
  const v = $("video"), c = ensureOverlay();
  c.width = v.videoWidth; c.height = v.videoHeight;
  const g = c.getContext("2d"); g.clearRect(0, 0, c.width, c.height);
  g.lineWidth = 3; g.font = "600 16px Segoe UI, sans-serif";
  for (const o of objects) {
    const [x, y, w, h] = o.bbox, hit = o === target, color = hit ? "#59d8ff" : "#7fe7c7";
    g.strokeStyle = color; g.strokeRect(x, y, w, h);
    const label = `${o.class} ${Math.round(o.score * 100)}%`, tw = g.measureText(label).width + 12;
    g.fillStyle = color; g.fillRect(x, Math.max(0, y - 22), tw, 22);
    g.fillStyle = "#0b3a4a"; g.fillText(label, x + 6, Math.max(16, y - 6));
  }
  if (live.fingertip) {
    const [fx, fy] = live.fingertip; g.fillStyle = "#ffd166";
    g.beginPath(); g.arc(fx, fy, 7, 0, 7); g.fill();
    if (live.ray) {
      const [dx, dy] = live.ray; g.strokeStyle = "#ffd166"; g.setLineDash([6, 6]);
      g.beginPath(); g.moveTo(fx, fy); g.lineTo(fx + dx * 6, fy + dy * 6); g.stroke(); g.setLineDash([]);
    }
  }
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 960 } } });
  } catch (e) { notify("Camera unavailable: " + e.message); return; }
  $("video").srcObject = stream;
  $("cameraEmpty").classList.add("hidden");
  $("camBtn").classList.add("on"); $("camLabel").textContent = "Camera on";
  $("cameraStatus").textContent = "LIVE"; $("cameraStatus").classList.add("live");
  if (profile && !profile.camera_enabled) saveProfile({ camera_enabled: true });
  try { await loadVision(); }
  catch (e) { notify("Object detection failed to load: " + e.message); $("sceneSummary").textContent = "Camera on · detection unavailable"; return; }
  detectTimer = setInterval(() => detectLoop().catch((e) => console.warn(e)), 450);
}
function stopCamera() {
  clearInterval(detectTimer); detectTimer = null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null; $("video").srcObject = null;
  live = { objects: [], pointing: null, fingertip: null, ray: null };
  if (overlay) overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  $("cameraEmpty").classList.remove("hidden");
  $("camBtn").classList.remove("on"); $("camLabel").textContent = "Camera off";
  $("cameraStatus").textContent = "PRIVATE SPACE"; $("cameraStatus").classList.remove("live");
  $("sceneSummary").textContent = "Camera is off"; $("objectsSummary").textContent = "Waiting for context"; $("gestureSummary").textContent = "No pointing detected";
  renderObjectTags([]);
  if (profile && profile.camera_enabled) saveProfile({ camera_enabled: false });
}
$("enableCamera").addEventListener("click", startCamera);
$("camBtn").addEventListener("click", () => (stream ? stopCamera() : startCamera()));

function grabFrame() {
  const v = $("video");
  if (!stream || !v.videoWidth) return null;
  const c = document.createElement("canvas");
  c.width = 640; c.height = Math.round((640 * v.videoHeight) / v.videoWidth);
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.7);
}
function renderObjectTags(objects, pointing) {
  let box = document.querySelector(".object-tags");
  if (!box) { box = document.createElement("div"); box.className = "object-tags"; document.querySelector(".camera-panel").appendChild(box); }
  const seen = new Set();
  box.innerHTML = objects.filter((o) => !seen.has(o.label) && seen.add(o.label))
    .map((o) => `<span class="object-tag ${o.label === pointing ? "pointing" : ""}">${esc(o.label)}</span>`).join("");
}

// ---------------- microphone: raw words, no cleanup
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
$("mic").addEventListener("click", () => {
  if (!SR) { notify("This browser has no speech recognition (use Chrome). Typing works too."); return; }
  if (rec) { rec.stop(); return; }
  rec = new SR(); rec.lang = "en-US"; rec.interimResults = true; rec.continuous = true;
  const wait = { long: 6000, medium: 3500, short: 2000 }[(profile && profile.pause_tolerance) || "medium"];
  let timer;
  const bump = () => { clearTimeout(timer); timer = setTimeout(() => rec && rec.stop(), wait); };
  rec.onresult = (e) => { let t = ""; for (const r of e.results) t += r[0].transcript; $("fragment").value = t; bump(); };
  rec.onend = () => { rec = null; $("mic").classList.remove("on"); $("micLabel").textContent = "Mic off"; $("listenStatus").textContent = "Ready"; $("listenStatus").className = "pill grey"; if ($("fragment").value.trim()) process(); };
  rec.onerror = (e) => notify("Microphone: " + e.error);
  rec.start(); bump();
  $("mic").classList.add("on"); $("micLabel").textContent = "Mic on";
  $("listenStatus").textContent = "Listening…"; $("listenStatus").className = "pill";
});

// ---------------- interaction
$("send").addEventListener("click", process);
$("fragment").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); process(); } });

async function process() {
  const transcript = $("fragment").value.trim();
  if (!transcript) return;
  $("cands").innerHTML = '<div class="candidate-empty"><p>Thinking…</p></div>';
  $("confirmRow").classList.add("hidden");
  const useScene = $("sceneEnabled").checked;
  const liveLabels = [...new Set(live.objects.map((o) => o.label))];   // live camera wins over the manual box
  try {
    last = await api("/interaction/process", {
      user_id: USER, transcript,
      frame: grabFrame(),
      scene_hint: liveLabels.length ? liveLabels
                : useScene ? $("scene").value.split(",").map((s) => s.trim()).filter(Boolean) : [],
      pointing_hint: liveLabels.length ? live.pointing
                   : useScene ? $("pointing").value.trim() || null : null,
    });
  } catch (e) { notify(e.detail || "Could not reach EchoLoop"); return; }
  selected = 0; confirmed = null;
  renderPerception(last.perception);
  renderCandidates();
  renderAgents(last.trace_summary, null);
  $("improvement").innerHTML = `<span class="growth-mark">↗</span><div><strong>Waiting for your feedback.</strong><p>Confirm or edit to teach EchoLoop what you meant.</p></div>`;
  $("rewardBadge").innerHTML = "";
  renderTrace(last.agent_messages);
}

function renderPerception(p) {
  $("objectsSummary").textContent = p.objects.length ? p.objects.map((o) => o.label).join(", ") : "No objects in context";
  $("gestureSummary").textContent = p.gesture.target ? `${p.gesture.type} near ${p.gesture.target}` : "No pointing detected";
  if (p.camera_enabled) $("sceneSummary").textContent = `Visual context on · confidence ${pct(p.perception_confidence)}`;
  renderObjectTags(p.objects, p.gesture.target);
}

function renderCandidates() {
  const d = last.candidate_detail;
  $("cands").innerHTML = d.map((c, i) =>
    `<button class="candidate ${i === selected ? "selected" : ""}" role="radio" aria-checked="${i === selected}" data-i="${i}">
       <span class="radio"></span><span class="text">${esc(c.text)}</span>
       <small>${c.features.judgment ? `System One ${pct(c.features.judgment)} · ` : ""}memory ${pct(c.memory_similarity)} · visual ${pct(c.visual_support)}</small></button>`).join("") +
    `<button class="candidate none" id="noneBtn"><span class="radio"></span><span class="text">None of these / Edit message</span><span data-icon="edit"></span></button>`;
  document.querySelectorAll(".candidate[data-i]").forEach((b) => b.addEventListener("click", () => { selected = +b.dataset.i; renderCandidates(); }));
  $("noneBtn").addEventListener("click", () => openEdit(""));
  const ic = $("noneBtn").querySelector("[data-icon]"); ic.innerHTML = `<svg viewBox="0 0 24 24">${ICONS.edit}</svg>`;
  $("confirmRow").classList.remove("hidden");
  const m = last.memory_matches[0];
  $("memMatch").textContent = m ? `Memory match: ${pct(m.similarity)}` : "No memory yet";
  $("memMatch").className = m ? "pill" : "pill grey";
}

function renderAgents(summary, reflection) {
  const pick = (k) => (summary.find((s) => s.startsWith(k)) || "").replace(/^[^:]+:\s*/, "");
  $("perceptionAgent").textContent = pick("Perception") || "Ready to see and hear";
  $("intentAgent").textContent = pick("Intent") || "Finding possible meanings";
  $("learningAgent").textContent = pick("Learning") || "Connecting your context";
  $("reflectionAgent").textContent = reflection ? `${reflection.failure_type.replace(/_/g, " ").toLowerCase()} — ${reflection.recommendation}` : "Learning from your feedback";
}

function renderTrace(msgs) {
  const AGENT = { perception_agent: "Perception", intent_agent: "Intent", learning_agent: "Learning",
                  reflection_agent: "Reflection", user: "You", typesafe: "TypeSafe System One" };
  const brief = (m) => {
    const p = m.payload;
    switch (m.message_type) {
      case "perception_summary": return `“${p.transcript}” · objects: ${p.objects.map((o) => o.label).join(", ") || "none"} · pointing: ${p.gesture.target || "none"} · confidence ${pct(p.perception_confidence)}`;
      case "candidate_set": return p.candidates.map((c) => `${c.text} (base ${c.base_score.toFixed(2)})`).join("  |  ");
      case "ranked_candidates": return p.ranked.map((c, i) => `#${i + 1} ${c.text} (${c.score.toFixed(2)})`).join("  |  ") + ` · policy v${p.policy_version}`;
      case "system_one_judgment": return Object.entries(p.probabilities).sort((a, b) => b[1] - a[1]).map(([t, v]) => `${t} ${pct(v)}`).join("  |  ") + ` · confidence ${pct(p.confidence)} · ${p.model}`;
      case "user_feedback": return `${p.accepted ? "accepted" : "rejected"} → reward ${p.reward > 0 ? "+1" : "−1"}${p.confirmed_text ? ` · confirmed “${p.confirmed_text}”` : " · none fit"}`;
      case "policy_update": return Object.keys(p.weights_after).filter((k) => p.weights_after[k] !== p.weights_before[k])
        .map((k) => `${k} ${p.weights_before[k]} → ${p.weights_after[k]}`).join(", ") || "no weight change (generator miss, ranker left alone)";
      case "reflection_result": return `${p.failure_type} · ${p.reason_code} · ${p.recommendation}`;
      default: return JSON.stringify(p);
    }
  };
  $("trace").innerHTML = msgs.map((m) =>
    `<div class="msg"><div class="msg-head"><b>${AGENT[m.from] || m.from}</b> → <b>${AGENT[m.to] || m.to}</b><code>${m.message_type}</code></div>
     <div class="msg-body">${esc(brief(m))}</div><details><summary>payload</summary><pre>${esc(JSON.stringify(m.payload, null, 2))}</pre></details></div>`).join("");
}

// ---------------- feedback
$("yes").addEventListener("click", () => {
  const c = last.candidate_detail[selected];
  sendFeedback({ accepted: selected === 0, confirmed_text: c.text, chosen_candidate_id: c.id });
});
$("no").addEventListener("click", () => openEdit(last.candidate_detail[selected].text));
function openEdit(text) { $("editedText").value = text; $("editDialog").showModal(); $("editedText").focus(); }
$("closeEdit").addEventListener("click", () => $("editDialog").close());
$("editForm").addEventListener("submit", (e) => {
  e.preventDefault(); $("editDialog").close();
  const t = $("editedText").value.trim();
  if (t) sendFeedback({ accepted: false, confirmed_text: t });
});
$("noneConfirm").addEventListener("click", () => { $("editDialog").close(); sendFeedback({ accepted: false, none_fit: true }); });

async function sendFeedback(fb) {
  let r;
  try { r = await api("/interaction/feedback", { interaction_id: last.interaction_id, elapsed_ms: 0, ...fb }); }
  catch (e) { notify(e.detail || "Feedback failed"); return; }
  $("confirmRow").classList.add("hidden");
  renderImprovement(r);
  renderAgents(last.trace_summary, r.reflection);
  if (r.speak) {
    confirmed = r.speak;
    $("confirmedLabel").textContent = "Confirmed message:"; $("confirmedText").textContent = confirmed;
    $("speak").disabled = false;
  } else {
    $("confirmedLabel").textContent = "Nothing confirmed"; $("confirmedText").textContent = "No suggestion fit. Nothing will be spoken.";
    $("speak").disabled = true;
  }
  renderTrace([...last.agent_messages, ...r.agent_messages]);
}

function renderImprovement(r) {
  const cands = last.candidate_detail;
  const dot = (f, w) => Object.keys(w).reduce((s, k) => s + (f[k] || 0) * w[k], 0);
  const norm = (xs) => { const m = Math.min(...xs); const sh = xs.map((x) => x - m + 0.05); const t = sh.reduce((a, b) => a + b, 0); return sh.map((x) => x / t); };
  const before = norm(cands.map((c) => c.score));
  const after = norm(cands.map((c) => dot(c.features, r.policy_weights)));
  const col = (title, vals, cls) => `<div class="ba-col ${cls}"><strong>${title}</strong>` +
    cands.map((c, i) => ({ c, v: vals[i] })).sort((a, b) => b.v - a.v).map(({ c, v }) =>
      `<div class="bar-row"><span>${esc(c.text.replace(/^I need (my )?/, ""))}</span><span>${pct(v)}</span>
       <span class="bar ${c.text === r.confirmed_text ? "win" : ""}"><i style="width:${pct(v)}"></i></span></div>`).join("") + "</div>";
  const good = r.reward > 0;
  $("improvement").innerHTML = `<div class="before-after">${col("Before feedback", before, "")}<span class="ba-arrow">→</span>${col("After feedback", after, "after")}</div>`;
  $("rewardBadge").innerHTML = `<div class="badges"><span class="badge ${good ? "" : "neg"}">${good ? "+1" : "−1"} reward</span>` +
    (r.memory_saved ? `<span class="badge">Memory updated</span>` : "") +
    `<span class="badge grey">Policy v${r.policy_version}</span></div>` +
    `<div class="reflection"><strong>Reflection:</strong> ${esc(r.reflection.failure_type)} · ${esc(r.reflection.reason_code)}<br>${esc(r.reflection.recommendation)}</div>`;
}

// ---------------- speech out: confirmed text only
$("speak").addEventListener("click", async () => {
  if (!confirmed) return;
  const r = await fetch("/api/speak", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: confirmed, interaction_id: last.interaction_id }) });
  if ((r.headers.get("content-type") || "").includes("audio")) {
    audio = new Audio(URL.createObjectURL(await r.blob())); audio.play();
  } else {
    const u = new SpeechSynthesisUtterance(confirmed);
    u.rate = +(localStorage.getItem("echoloop_rate") || 1);
    speechSynthesis.speak(u);
  }
});
$("stopSpeak").addEventListener("click", () => { speechSynthesis.cancel(); if (audio) audio.pause(); });

// ---------------- history / evaluation
async function loadHistory() {
  const h = await api(`/api/history/${USER}`);
  $("mems").innerHTML = h.memories.map((m) =>
    `<div class="row"><div><strong>${esc(m.confirmed_text)}</strong><small>from “${esc(m.fragment)}” · ${esc(m.visual_summary)}</small></div>
     <span class="badge">✓ ${m.success_count}</span>${m.failure_count ? `<span class="badge neg">✗ ${m.failure_count}</span>` : ""}
     <button class="forget" data-t="${esc(m.confirmed_text)}">Forget</button></div>`).join("") ||
    `<p class="muted">No confirmed memories yet.</p>`;
  document.querySelectorAll(".forget").forEach((b) => b.addEventListener("click", async () => {
    await api(`/api/forget/${USER}`, { confirmed_text: b.dataset.t }); loadHistory();
  }));
  $("hist").innerHTML = h.history.map((i) => {
    const top = (i.observation.candidates[0] || {}).text || "";
    const fb = i.feedback;
    return `<div class="row"><div><strong>“${esc(i.observation.transcript)}”</strong>
      <small>suggested “${esc(top)}”${fb ? ` · confirmed “${esc(fb.confirmed_text || "none")}” · ${esc(i.reflection.failure_type)}` : " · awaiting feedback"} · policy v${i.policy_version}</small></div>
      ${fb ? `<span class="badge ${fb.reward > 0 ? "" : "neg"}">${fb.reward > 0 ? "+1" : "−1"} reward</span>` : ""}</div>`;
  }).join("") || `<p class="muted">Nothing yet.</p>`;
  $("policy").innerHTML = Object.entries(h.policy).map(([k, v]) => `<div class="metric"><small>${k}</small><strong>${v}</strong></div>`).join("");
}

async function loadEval() {
  let r;
  try { r = await api("/api/evaluation"); }
  catch { $("eval").innerHTML = `<p class="muted">No report yet — run <code>python -m eval.run_eval</code>.</p>`; return; }
  const a = r.ablations, best = a[a.length - 1], frozen = a[a.length - 2];
  $("eval").innerHTML =
    `<div class="stat-grid">
      <div class="stat"><small>Top-1 accuracy</small><strong>${pct(best.top1)}</strong><em>${(r.personalization_gain * 100 >= 0 ? "+" : "")}${(r.personalization_gain * 100).toFixed(1)} pts from learning</em></div>
      <div class="stat"><small>Top-3 accuracy</small><strong>${pct(best.top3)}</strong></div>
      <div class="stat"><small>Avg. clarification turns</small><strong>${best.clarification_burden.toFixed(2)}</strong><em>${((best.clarification_burden / frozen.clarification_burden - 1) * 100).toFixed(0)}% vs frozen</em></div>
      <div class="stat"><small>Average reward</small><strong>${best.avg_reward.toFixed(2)}</strong></div>
    </div>
    <table><tr><th>Configuration</th><th>Top-1</th><th>Top-3</th><th>Reward</th><th>Clarification</th><th>ms</th></tr>` +
    a.map((x) => `<tr><td>${esc(x.name)}</td><td>${pct(x.top1)}</td><td>${pct(x.top3)}</td><td>${x.avg_reward.toFixed(3)}</td><td>${x.clarification_burden.toFixed(2)}</td><td>${x.latency_ms_per_interaction.toFixed(1)}</td></tr>`).join("") +
    `</table><p class="muted">${esc(r.dataset)} · ${r.n_test} held-out interactions · ${esc(r.generated)}</p>`;
}

// ---------------- boot
(async () => {
  try {
    const st = await api("/api/status");
    if (!st.integrations.elevenlabs) $("speak").title = "Browser voice (no ElevenLabs key set)";
  } catch { notify("Backend not reachable. Start it with: python -m uvicorn backend.main:app"); }
  if (localStorage.getItem("echoloop_onboarded")) {
    $("userId").value = USER;
    const p = await api(`/api/profile/${USER}`);
    applyProfile(p.profile, p.quick_phrases);
    showTab(location.hash.replace("#", "") || "live");
  } else {
    showTab("onboarding");
  }
})();
