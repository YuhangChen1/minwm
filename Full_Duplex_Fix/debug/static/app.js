const state = {
  events: [],
  eventById: new Map(),
  selectedId: null,
  latestId: 0,
  followLatest: true,
  run: null,
  losses: [],
};

const elements = Object.fromEntries([
  "sessionId", "deviceValue", "stepValue", "eventCount", "statusBadge", "progressBar",
  "errorBanner", "nextButton", "autoButton", "pauseButton", "stopButton", "resetButton",
  "delaySelect", "phaseFilter", "eventSearch", "eventList", "eventPath", "eventTitle",
  "eventTiming", "shapeFlow", "tensorGrid", "metadataGrid", "parameterSection",
  "parameterSearch", "parameterSummary", "parameterRows", "tracebackSection", "tracebackValue",
  "modeValue", "lossCanvas", "latestLoss", "latestInit", "latestTransition", "latestGrad",
  "forwardCount", "backwardCount", "optimizerCount", "outputPath"
].map((id) => [id, document.getElementById(id)]));

function formatNumber(value, digits = 6) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const number = Number(value);
  if (number === 0) return "0";
  if (Math.abs(number) < 0.001 || Math.abs(number) >= 10000) return number.toExponential(3);
  return number.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function formatValue(value) {
  if (Array.isArray(value)) return `[${value.map((item) => formatValue(item)).join(", ")}]`;
  if (value && typeof value === "object") return JSON.stringify(value, null, 2);
  if (typeof value === "number") return formatNumber(value);
  return String(value ?? "-");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function showError(message) {
  elements.errorBanner.hidden = !message;
  elements.errorBanner.textContent = message || "";
}

function updateRunState(run) {
  state.run = run;
  const status = run.status || "not_started";
  elements.sessionId.textContent = run.session_id || "not started";
  elements.deviceValue.textContent = run.device || "cuda:2";
  elements.stepValue.textContent = `${run.step || 0} / ${run.total_steps || 5}`;
  elements.eventCount.textContent = String(run.event_count || state.events.length);
  elements.statusBadge.textContent = status.replaceAll("_", " ").toUpperCase();
  elements.statusBadge.className = `status-badge ${status}`;
  elements.modeValue.textContent = String(run.mode || "step").toUpperCase();
  elements.outputPath.textContent = run.output_dir || "-";
  elements.progressBar.style.width = `${Math.min(100, ((run.step || 0) / (run.total_steps || 5)) * 100)}%`;

  const active = ["starting", "running", "paused", "stopping"].includes(status);
  elements.nextButton.disabled = !active || status === "stopping";
  elements.autoButton.disabled = !active || status === "stopping";
  elements.pauseButton.disabled = !active || status === "stopping";
  elements.stopButton.disabled = !active || status === "stopping";
  elements.resetButton.disabled = active;
  if (status === "error") showError(run.message || "Training failed");
}

function phaseLabel(event) {
  if (event.phase === "backward") return "BWD";
  if (event.phase === "optimizer") return "OPT";
  if (event.phase === "error") return "ERR";
  return "FWD";
}

function filteredEvents() {
  const phase = elements.phaseFilter.value;
  const query = elements.eventSearch.value.trim().toLowerCase();
  return state.events.filter((event) => {
    const phaseMatches = phase === "all" || event.phase === phase;
    const textMatches = !query || `${event.name} ${event.category}`.toLowerCase().includes(query);
    return phaseMatches && textMatches;
  });
}

function renderTimeline() {
  const fragment = document.createDocumentFragment();
  for (const event of filteredEvents()) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `event-row ${event.phase}${event.id === state.selectedId ? " selected" : ""}`;
    button.title = event.name;
    button.innerHTML = `
      <span class="event-id">#${event.id}</span>
      <span><span class="event-name">${escapeHtml(event.name)}</span><span class="event-meta">step ${event.step} / ${escapeHtml(event.category)}</span></span>
      <span class="phase-mark">${phaseLabel(event)}</span>`;
    button.addEventListener("click", () => {
      state.followLatest = false;
      selectEvent(event.id);
    });
    fragment.appendChild(button);
  }
  elements.eventList.replaceChildren(fragment);
  if (state.followLatest) elements.eventList.scrollTop = elements.eventList.scrollHeight;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderShapeFlow(event) {
  const entries = Object.entries(event.tensors || {});
  if (!entries.length) {
    elements.shapeFlow.innerHTML = `<div class="shape-node"><strong>${escapeHtml(event.category)}</strong><code>no tensor output</code></div>`;
    return;
  }
  elements.shapeFlow.innerHTML = entries.map(([name, tensor], index) => `
    ${index ? '<span class="shape-arrow">-&gt;</span>' : ""}
    <div class="shape-node"><strong>${escapeHtml(name)}</strong><code>${escapeHtml(JSON.stringify(tensor.shape))}</code></div>
  `).join("");
}

function renderTensors(event) {
  const entries = Object.entries(event.tensors || {});
  if (!entries.length) {
    elements.tensorGrid.innerHTML = '<div class="tensor-item"><h4>No tensor payload</h4></div>';
    return;
  }
  elements.tensorGrid.innerHTML = entries.map(([name, tensor]) => `
    <article class="tensor-item">
      <h4>${escapeHtml(name)}</h4>
      <dl class="tensor-stats">
        <dt>shape</dt><dd>${escapeHtml(JSON.stringify(tensor.shape))}</dd>
        <dt>dtype</dt><dd>${escapeHtml(tensor.dtype)}</dd>
        <dt>device</dt><dd>${escapeHtml(tensor.device)}</dd>
        <dt>first [0:2]</dt><dd>${escapeHtml(formatValue(tensor.first_values))}</dd>
        <dt>mean / std</dt><dd>${formatNumber(tensor.mean)} / ${formatNumber(tensor.std)}</dd>
        <dt>min / max</dt><dd>${formatNumber(tensor.min)} / ${formatNumber(tensor.max)}</dd>
        <dt>sample</dt><dd>${tensor.sample_size || 0} @ stride ${tensor.sample_stride || 1}</dd>
        <dt>finite</dt><dd>${String(tensor.finite ?? true)}</dd>
      </dl>
    </article>`).join("");
}

function renderMetadata(event) {
  const details = event.details || {};
  const entries = Object.entries(details).filter(([key]) => !["parameters", "traceback"].includes(key));
  elements.metadataGrid.innerHTML = entries.length ? entries.map(([key, value]) => `
    <div class="metadata-key">${escapeHtml(key)}</div>
    <div class="metadata-value">${escapeHtml(formatValue(value))}</div>`).join("") : `
    <div class="metadata-key">event</div><div class="metadata-value">${escapeHtml(event.name)}</div>`;
}

function renderParameters(event) {
  const rows = event.details?.parameters;
  const visible = Array.isArray(rows) && rows.length > 0;
  elements.parameterSection.hidden = !visible;
  if (!visible) return;
  const query = elements.parameterSearch.value.trim().toLowerCase();
  const filtered = rows.filter((row) => !query || row.name.toLowerCase().includes(query));
  const shown = filtered.slice(0, 1500);
  elements.parameterSummary.textContent = `${shown.length} shown / ${filtered.length} matched / ${rows.length} total tensors; values are flattened [0:2]`;
  elements.parameterRows.innerHTML = shown.map((row) => {
    const changed = row.first_values_changed ? " delta-changed" : "";
    return `<tr>
      <td>${escapeHtml(row.name)}</td>
      <td>${escapeHtml(JSON.stringify(row.shape))}</td>
      <td>${escapeHtml(formatValue(row.before))}</td>
      <td>${escapeHtml(formatValue(row.after))}</td>
      <td class="${changed}">${escapeHtml(formatValue(row.delta))}</td>
      <td>${escapeHtml(formatValue(row.gradient))}</td>
    </tr>`;
  }).join("");
}

function selectEvent(id) {
  const event = state.eventById.get(id);
  if (!event) return;
  state.selectedId = id;
  elements.eventPath.textContent = `step ${event.step} / ${event.phase} / ${event.category} / event ${event.id}`;
  elements.eventTitle.textContent = event.name;
  elements.eventTiming.textContent = `${formatNumber(event.details?.duration_ms || 0, 3)} ms`;
  renderShapeFlow(event);
  renderTensors(event);
  renderMetadata(event);
  renderParameters(event);
  const traceback = event.details?.traceback;
  elements.tracebackSection.hidden = !traceback;
  elements.tracebackValue.textContent = traceback || "";
  renderTimeline();
}

function countPhases() {
  const counts = { forward: 0, backward: 0, optimizer: 0 };
  state.events.forEach((event) => { if (event.phase in counts) counts[event.phase] += 1; });
  elements.forwardCount.textContent = counts.forward;
  elements.backwardCount.textContent = counts.backward;
  elements.optimizerCount.textContent = counts.optimizer;
}

function collectMetrics(event) {
  if (event.name !== "training.step_complete") return;
  const record = event.details || {};
  if (state.losses.some((point) => point.step === record.step)) return;
  state.losses.push({
    step: record.step,
    loss: Number(record.loss),
    init: Number(record.loss_init),
    transition: Number(record.loss_transition),
    gradient: Number(record.gradient_norm),
  });
  state.losses.sort((a, b) => a.step - b.step);
  const latest = state.losses.at(-1);
  elements.latestLoss.textContent = formatNumber(latest.loss);
  elements.latestInit.textContent = formatNumber(latest.init);
  elements.latestTransition.textContent = formatNumber(latest.transition);
  elements.latestGrad.textContent = formatNumber(latest.gradient);
  drawLossChart();
}

function drawLossChart() {
  const canvas = elements.lossCanvas;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const width = rect.width;
  const height = rect.height;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#d8ddda";
  context.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const y = 12 + ((height - 30) * i) / 4;
    context.beginPath(); context.moveTo(32, y); context.lineTo(width - 10, y); context.stroke();
  }
  if (!state.losses.length) {
    context.fillStyle = "#657074";
    context.font = "12px ui-monospace";
    context.fillText("waiting for step metrics", 44, height / 2);
    return;
  }
  const values = state.losses.flatMap((point) => [point.loss, point.init, point.transition]);
  const maximum = Math.max(...values, 1e-8) * 1.08;
  const colors = { loss: "#087f72", init: "#b23832", transition: "#2267a8" };
  Object.entries(colors).forEach(([key, color]) => {
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    state.losses.forEach((point, index) => {
      const x = 32 + ((point.step - 1) / 4) * (width - 48);
      const y = 10 + (1 - point[key] / maximum) * (height - 30);
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
      context.fillStyle = color;
      context.fillRect(x - 2, y - 2, 4, 4);
    });
    context.stroke();
  });
  context.fillStyle = "#657074";
  context.font = "10px ui-monospace";
  context.fillText(`max ${formatNumber(maximum)}`, 4, 12);
  context.fillText("loss", 36, height - 6);
  context.fillStyle = colors.init; context.fillText("init", 78, height - 6);
  context.fillStyle = colors.transition; context.fillText("transition", 114, height - 6);
}

function ingest(events) {
  for (const event of events) {
    if (state.eventById.has(event.id)) continue;
    state.events.push(event);
    state.eventById.set(event.id, event);
    state.latestId = Math.max(state.latestId, event.id);
    collectMetrics(event);
  }
  state.events.sort((a, b) => a.id - b.id);
  if (events.length && state.followLatest) selectEvent(events.at(-1).id);
  else renderTimeline();
  countPhases();
}

async function poll() {
  try {
    let more = true;
    while (more) {
      const payload = await api(`/api/events?after=${state.latestId}&limit=200`);
      updateRunState(payload.state);
      ingest(payload.events);
      more = payload.events.length === 200;
    }
    if (state.run?.status !== "error") showError("");
  } catch (error) {
    showError(error.message);
  } finally {
    window.setTimeout(poll, 300);
  }
}

async function control(action) {
  try {
    const payload = await api("/api/control", {
      method: "POST",
      body: JSON.stringify({ action, delay_ms: Number(elements.delaySelect.value) }),
    });
    updateRunState(payload);
    if (action === "next" || action === "auto") state.followLatest = true;
  } catch (error) { showError(error.message); }
}

elements.nextButton.addEventListener("click", () => control("next"));
elements.autoButton.addEventListener("click", () => control("auto"));
elements.pauseButton.addEventListener("click", () => control("pause"));
elements.stopButton.addEventListener("click", () => control("stop"));
elements.resetButton.addEventListener("click", async () => {
  try {
    await api("/api/reset", { method: "POST", body: "{}" });
    window.location.reload();
  } catch (error) { showError(error.message); }
});
elements.phaseFilter.addEventListener("change", renderTimeline);
elements.eventSearch.addEventListener("input", renderTimeline);
elements.parameterSearch.addEventListener("input", () => {
  if (state.selectedId) renderParameters(state.eventById.get(state.selectedId));
});
window.addEventListener("resize", drawLossChart);

async function initialize() {
  drawLossChart();
  try {
    const initial = await api("/api/state");
    updateRunState(initial);
    const autoStart = new URLSearchParams(window.location.search).get("autostart") !== "0";
    if (autoStart && initial.status === "not_started") {
      updateRunState(await api("/api/start", { method: "POST", body: "{}" }));
      await control("auto");
    }
  } catch (error) { showError(error.message); }
  poll();
}

initialize();
