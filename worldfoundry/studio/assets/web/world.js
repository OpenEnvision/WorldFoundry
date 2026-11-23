const studioAuthToken = new URLSearchParams(window.location.search).get("token") || "";

// Mirrors DEFAULT_REALTIME_KEYS in core/execution/realtime/contracts.py; a spec
// without a "keys" field uses these bindings.
const DEFAULT_REALTIME_KEYS = [
  { key: "w", action: "forward", group: "forward_backward", label: "" },
  { key: "s", action: "backward", group: "forward_backward", label: "" },
  { key: "a", action: "left", group: "left_right", label: "" },
  { key: "d", action: "right", group: "left_right", label: "" },
  { key: "i", action: "camera_up", group: "camera_up_down", label: "" },
  { key: "k", action: "camera_down", group: "camera_up_down", label: "" },
  { key: "j", action: "camera_l", group: "camera_left_right", label: "" },
  { key: "l", action: "camera_r", group: "camera_left_right", label: "" },
];

// Correlate the DataChannel receipt with the exact frame presented by WebRTC.
// Both control submission and presentation times come from this browser clock.
class PresentationMetrics {
  constructor(report) {
    this.report = report;
    this.reset();
  }
  reset() {
    this.controls = new Map();
    this.sent = new Map();
    this.displayed = new Map();
  }
  remember(map, key, value) {
    map.set(key, value);
    while (map.size > 256) map.delete(map.keys().next().value);
  }
  submitted(id, now) { this.remember(this.controls, id, now); }
  frameSent(message) {
    if (!Number.isInteger(message.rtp_timestamp)) return;
    this.remember(this.sent, message.rtp_timestamp, message);
    this.match(message.rtp_timestamp);
  }
  presented(timestamp, now) {
    if (!Number.isInteger(timestamp)) return;
    if (!this.displayed.has(timestamp)) this.remember(this.displayed, timestamp, now);
    this.match(timestamp);
  }
  match(timestamp) {
    const frame = this.sent.get(timestamp);
    const now = this.displayed.get(timestamp);
    if (!frame || now === undefined) return;
    for (const id of frame.control_ids || []) {
      const started = this.controls.get(id);
      if (started === undefined || now < started) continue;
      this.controls.delete(id);
      this.report({type: "frame_presented", rtp_timestamp: timestamp,
        control_id: id, control_to_display_ms: now - started});
    }
    this.sent.delete(timestamp);
    this.displayed.delete(timestamp);
  }
}

const state = {
  modelLoaded: false,
  mode: "image",
  seedPath: "",
  selectedFile: null,
  uploadedPath: "",
  previewUrl: "",
  examples: [],
  promptScheduled: false,
  queuedSegments: false,
  segmentBusy: false,
  segmentIndex: 0,
  denseFile: null,
  denseUploadedPath: "",
  sparseFile: null,
  sparseUploadedPath: "",
  inflightDenseFile: null,
  inflightDensePath: "",
  inflightSparseFile: null,
  inflightSparsePath: "",
  iceServers: [],
  preferWebSocket: false,
  peer: null,
  socket: null,
  channel: null,
  transport: "",
  connected: false,
  connecting: false,
  fallbackStarted: false,
  sessionStartedAt: 0,
  heartbeatTimer: null,
  transportStatsTimer: null,
  transportStatsPending: false,
  previousTransportStats: new Map(),
  transportStats: {},
  presentation: new PresentationMetrics((sample) => send(sample)),
  presentationSummary: null,
  connectionTimer: null,
  stats: { fps: 0, generationMs: 0, enqueueMs: 0, latencyMs: 0, queueDepth: 0, dropped: 0 },
  lastVideoFrameAt: 0,
  streamFrameUrl: "",
  playbackFps: 16,
  socketFrameQueue: [],
  socketFramePump: false,
  socketPlaybackEpoch: 0,
  nextPresentationAt: 0,
  presentedFrames: 0,
  segmentPresentedFrames: 0,
  pendingChunkDone: null,
  pendingChunkDones: [],
  inflightSegments: 0,
  keySources: new Map(),
  stickKeys: { move: new Set(), look: new Set() },
  stickPointers: { move: null, look: null },
  stickPositions: { move: { x: 0, y: 0 }, look: { x: 0, y: 0 } },
  realtimeKeys: DEFAULT_REALTIME_KEYS,
  declaredKeys: false,
  mouseAxes: false,
  gamepadAxes: false,
  gamepadSources: [],
  axisChannel: null,
  pointerLocked: false,
  mouseDelta: { dx: 0, dy: 0 },
  mouseRaf: 0,
  gamepadRaf: 0,
  gamepadSent: { lx: 0, ly: 0, rx: 0, ry: 0 },
  continuous: false,
  rolloutChunks: 0,
  maxChunks: null,
  paused: false,
  immersive: false,
  nativeFullscreen: false,
  logOpen: false,
  logPending: [],
  logFlushRaf: 0,
  logCount: 0,
  lastLoggedStatus: "",
  runtimePhase: "",
  capabilities: {},
  textEvents: [],
  activeEventId: null,
  catalogRevision: 0,
  requestSequence: 0,
  pendingEventRequestId: "",
  pendingCatalogRequestId: "",
  pendingOutputRequestId: "",
  pendingStepRequestId: "",
  pendingPauseRequestId: "",
  requestTimers: new Map(),
  outputResolution: { mode: "native" },
  outputResolutionOptions: [],
  panelZ: 70,
  sensorLayout: [],
  hiddenSensors: new Set(),
  focusedSensor: "",
  sensorCanvases: new Map(),
  logBeforeImmersive: false,
};

const emptyFrame =
  "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";
const allowedKeys = new Set(DEFAULT_REALTIME_KEYS.map((binding) => binding.key));
const keyAliases = new Map([
  ["arrowup", "i"],
  ["arrowdown", "k"],
  ["arrowleft", "j"],
  ["arrowright", "l"],
]);
const MOVE_STICK_KEYS = ["w", "a", "s", "d"];
const LOOK_STICK_KEYS = ["i", "j", "k", "l"];
const STICK_AXIS_SOURCES = {
  move: { x: "gamepad_lx", y: "gamepad_ly" },
  look: { x: "gamepad_rx", y: "gamepad_ry" },
};
// Smallest non-neutral stick change that produces a new axis message.
const GAMEPAD_EPSILON = 0.02;
// Axis samples wait for the next animation frame while this many bytes are
// still queued on their channel; mouse deltas keep accumulating meanwhile.
const AXIS_BUFFERED_LIMIT_BYTES = 16 * 1024;
const SOCKET_SETUP_TIMEOUT_MS = 120000;

function authenticatedUrl(raw) {
  const value = String(raw || "");
  if (!studioAuthToken || !value) return value;
  try {
    const url = new URL(value, window.location.href);
    if (url.origin === window.location.origin && ["http:", "https:"].includes(url.protocol)) {
      url.searchParams.set("token", studioAuthToken);
    }
    return url.toString();
  } catch {
    return value;
  }
}

const el = {
  root: document.body,
  viewport: document.getElementById("viewport"),
  image: document.getElementById("frameImage"),
  video: document.getElementById("frameVideo"),
  canvas: document.getElementById("frameCanvas"),
  sensorGrid: document.getElementById("sensorGrid"),
  sensorControls: document.getElementById("sensorControls"),
  startButton: document.getElementById("startButton"),
  resetButton: document.getElementById("resetButton"),
  livePill: document.getElementById("livePill"),
  overlayStatus: document.getElementById("overlayStatus"),
  timer: document.getElementById("sessionTimer"),
  streamStats: document.getElementById("streamStats"),
  thumbTray: document.getElementById("thumbTray"),
  imageInput: document.getElementById("imageInput"),
  videoInput: document.getElementById("videoInput"),
  denseVideoInput: document.getElementById("denseVideoInput"),
  sparseVideoInput: document.getElementById("sparseVideoInput"),
  imageInputLabel: document.getElementById("imageInputLabel"),
  denseInputLabel: document.getElementById("denseInputLabel"),
  sparseInputLabel: document.getElementById("sparseInputLabel"),
  imageFileState: document.getElementById("imageFileState"),
  denseFileState: document.getElementById("denseFileState"),
  sparseFileState: document.getElementById("sparseFileState"),
  promptInput: document.getElementById("promptInput"),
  loadModelButton: document.getElementById("loadModelButton"),
  moveStick: document.getElementById("moveStick"),
  lookStick: document.getElementById("lookStick"),
  fullscreenButton: document.getElementById("fullscreenButton"),
  fullscreenLabel: document.getElementById("fullscreenLabel"),
  logToggleButton: document.getElementById("logToggleButton"),
  clearLogButton: document.getElementById("clearLogButton"),
  runtimeLog: document.getElementById("runtimeLog"),
  runtimeLogList: document.getElementById("runtimeLogList"),
  logCount: document.getElementById("logCount"),
  stepButton: document.getElementById("stepButton"),
  pauseButton: document.getElementById("pauseButton"),
  pointerHint: document.getElementById("pointerHint"),
  interactionHint: document.getElementById("interactionHint"),
  resolutionSelect: document.getElementById("resolutionSelect"),
  controlAck: document.getElementById("controlAck"),
  eventTriggerBar: document.getElementById("eventTriggerBar"),
  addEventButton: document.getElementById("addEventButton"),
  applyEventsButton: document.getElementById("applyEventsButton"),
  eventEditorRows: document.getElementById("eventEditorRows"),
  eventCatalogStatus: document.getElementById("eventCatalogStatus"),
};

const MAX_RUNTIME_LOG_ENTRIES = 120;

function runtimeLogTime(date = new Date()) {
  const milliseconds = String(date.getMilliseconds()).padStart(3, "0");
  return `${date.toLocaleTimeString([], { hour12: false })}.${milliseconds}`;
}

function flushRuntimeLogs() {
  state.logFlushRaf = 0;
  if (!el.runtimeLogList || !state.logPending.length) return;
  const shouldFollow = el.runtimeLogList.scrollHeight - el.runtimeLogList.scrollTop
    - el.runtimeLogList.clientHeight < 48;
  const fragment = document.createDocumentFragment();
  for (const entry of state.logPending.splice(0)) {
    const row = document.createElement("li");
    row.className = "runtime-log-entry";
    row.dataset.level = entry.level;
    const timestamp = document.createElement("time");
    timestamp.className = "runtime-log-time";
    timestamp.dateTime = entry.iso;
    timestamp.textContent = entry.time;
    const message = document.createElement("span");
    message.className = "runtime-log-message";
    message.textContent = entry.message;
    if (entry.detail) {
      const detail = document.createElement("span");
      detail.className = "runtime-log-detail";
      detail.textContent = entry.detail;
      message.appendChild(detail);
    }
    row.append(timestamp, message);
    fragment.appendChild(row);
  }
  el.runtimeLogList.appendChild(fragment);
  while (el.runtimeLogList.childElementCount > MAX_RUNTIME_LOG_ENTRIES) {
    el.runtimeLogList.firstElementChild?.remove();
  }
  if (shouldFollow) el.runtimeLogList.scrollTop = el.runtimeLogList.scrollHeight;
}

function appendRuntimeLog(message, level = "info", detail = "") {
  const text = String(message || "").trim();
  if (!text || !el.runtimeLogList) return;
  const now = new Date();
  state.logCount += 1;
  state.logPending.push({
    message: text,
    detail: String(detail || "").trim(),
    level: ["live", "metric", "warn", "error"].includes(level) ? level : "info",
    time: runtimeLogTime(now),
    iso: now.toISOString(),
  });
  if (state.logPending.length > MAX_RUNTIME_LOG_ENTRIES) {
    state.logPending.splice(0, state.logPending.length - MAX_RUNTIME_LOG_ENTRIES);
  }
  if (el.logCount) el.logCount.textContent = state.logCount > 999 ? "999+" : String(state.logCount);
  if (!state.logFlushRaf) state.logFlushRaf = requestAnimationFrame(flushRuntimeLogs);
}

function clearRuntimeLog() {
  state.logPending.length = 0;
  if (state.logFlushRaf) cancelAnimationFrame(state.logFlushRaf);
  state.logFlushRaf = 0;
  state.logCount = 0;
  state.lastLoggedStatus = "";
  if (el.runtimeLogList) el.runtimeLogList.replaceChildren();
  if (el.logCount) el.logCount.textContent = "0";
}

function nextRequestId(kind) {
  state.requestSequence += 1;
  return `${kind}-${Date.now().toString(36)}-${state.requestSequence.toString(36)}`;
}

function setControlAck(text, kind = "") {
  if (!el.controlAck) return;
  el.controlAck.textContent = String(text || "");
  el.controlAck.classList.toggle("is-ok", kind === "ok");
  el.controlAck.classList.toggle("is-error", kind === "error");
}

function setCatalogStatus(text, kind = "") {
  if (!el.eventCatalogStatus) return;
  el.eventCatalogStatus.textContent = String(text || "");
  el.eventCatalogStatus.classList.toggle("is-ok", kind === "ok");
  el.eventCatalogStatus.classList.toggle("is-error", kind === "error");
}

const pendingFieldByKind = {
  event: "pendingEventRequestId",
  catalog: "pendingCatalogRequestId",
  output: "pendingOutputRequestId",
  step: "pendingStepRequestId",
  pause: "pendingPauseRequestId",
};

function clearPending(kind) {
  const field = pendingFieldByKind[kind];
  if (!field) return;
  const timer = state.requestTimers.get(kind);
  if (timer) window.clearTimeout(timer);
  state.requestTimers.delete(kind);
  state[field] = "";
  updateControlAvailability();
}

function setPending(kind, requestId) {
  clearPending(kind);
  const field = pendingFieldByKind[kind];
  state[field] = requestId;
  state.requestTimers.set(kind, window.setTimeout(() => {
    if (state[field] !== requestId) return;
    state[field] = "";
    state.requestTimers.delete(kind);
    if (kind === "catalog") setCatalogStatus("ACK TIMEOUT · RETRY APPLY", "error");
    else setControlAck(`${kind.toUpperCase()} ACK TIMEOUT`, "error");
    appendRuntimeLog(`${kind} acknowledgement timed out`, "error");
    renderEventTriggers();
    updateControlAvailability();
  }, 7000));
  updateControlAvailability();
}

function acceptAck(kind, message) {
  const field = pendingFieldByKind[kind];
  const expected = field ? state[field] : "";
  if (!expected || message.request_id !== expected) {
    appendRuntimeLog(`Ignored stale ${kind} acknowledgement`, "warn", String(message.request_id || "missing request_id"));
    return false;
  }
  clearPending(kind);
  return true;
}

function clearAllPending() {
  for (const kind of Object.keys(pendingFieldByKind)) clearPending(kind);
}

function addEventRow(event = {}, focus = true) {
  if (!el.eventEditorRows || el.eventEditorRows.childElementCount >= 12) {
    setCatalogStatus("MAXIMUM 12 EVENTS", "error");
    return;
  }
  const row = document.createElement("div");
  row.className = "event-editor-row";
  row.dataset.category = String(event.category || "event");
  const id = document.createElement("input");
  id.className = "event-id";
  id.maxLength = 64;
  id.placeholder = "event_id";
  id.ariaLabel = "Text event id";
  id.value = String(event.event_id || event.id || "");
  const label = document.createElement("input");
  label.className = "event-label";
  label.maxLength = 64;
  label.placeholder = "Button label";
  label.ariaLabel = "Text event label";
  label.value = String(event.label || "");
  const prompt = document.createElement("input");
  prompt.className = "event-prompt";
  prompt.maxLength = 1000;
  prompt.placeholder = "Text condition applied at the next generation boundary";
  prompt.ariaLabel = "Text event prompt";
  prompt.value = String(event.prompt || "");
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "event-remove";
  remove.textContent = "REMOVE";
  remove.addEventListener("click", () => {
    row.remove();
    setCatalogStatus("UNAPPLIED CHANGES");
  });
  for (const input of [id, label, prompt]) {
    input.addEventListener("input", () => setCatalogStatus("UNAPPLIED CHANGES"));
  }
  row.append(id, label, prompt, remove);
  el.eventEditorRows.appendChild(row);
  if (focus) id.focus();
  setCatalogStatus("UNAPPLIED CHANGES");
}

function collectTextEvents() {
  const events = [];
  const ids = new Set();
  for (const row of el.eventEditorRows?.querySelectorAll(".event-editor-row") || []) {
    const eventId = row.querySelector(".event-id")?.value.trim() || "";
    const label = row.querySelector(".event-label")?.value.trim() || eventId;
    const prompt = row.querySelector(".event-prompt")?.value.trim() || "";
    if (!/^[A-Za-z0-9_.:-]{1,64}$/.test(eventId)) {
      throw new Error("Every event needs a unique 1-64 character event_id using letters, numbers, or _.:-");
    }
    if (ids.has(eventId)) throw new Error(`Duplicate event_id: ${eventId}`);
    if (!label || label.length > 64) throw new Error(`Event ${eventId} needs a 1-64 character label`);
    if (!prompt || prompt.length > 1000) throw new Error(`Event ${eventId} needs a 1-1000 character prompt`);
    ids.add(eventId);
    events.push({ event_id: eventId, label, prompt, category: row.dataset.category || "event" });
  }
  return events;
}

function replaceEventRows(events) {
  if (!el.eventEditorRows) return;
  el.eventEditorRows.replaceChildren();
  for (const event of events || []) addEventRow(event, false);
  setCatalogStatus(events?.length ? `${events.length} EVENT${events.length === 1 ? "" : "S"} READY` : "EMPTY CATALOG", "ok");
}

function updateControlAvailability() {
  const connected = state.connected && Boolean(state.channel);
  // A running continuous rollout already advances every chunk; STEP applies
  // while it is paused, and after it completes the server explains why not.
  const autoAdvancing = state.continuous && !state.paused && !rolloutComplete();
  const segmentLocked = state.promptScheduled && !state.continuous && state.segmentBusy;
  if (el.stepButton) {
    el.stepButton.disabled = !connected || !state.capabilities.no_action_step
      || Boolean(state.pendingStepRequestId) || autoAdvancing || segmentLocked;
    el.stepButton.title = !state.capabilities.no_action_step
      ? "This model requires explicit segment controls"
      : segmentLocked
        ? "Wait for the current segment to finish playing"
        : autoAdvancing
          ? "Pause the rollout to generate one chunk at a time"
          : "Generate exactly one chunk without a keyboard action";
  }
  if (el.pauseButton) {
    el.pauseButton.disabled = !connected || Boolean(state.pendingPauseRequestId) || rolloutComplete();
  }
  el.eventTriggerBar?.querySelectorAll("button").forEach((button) => {
    button.disabled = !connected || !state.capabilities.text_events
      || Boolean(state.pendingEventRequestId) || segmentLocked;
  });
  if (el.addEventButton) el.addEventButton.disabled = !state.capabilities.text_events || Boolean(state.pendingCatalogRequestId);
  if (el.applyEventsButton) el.applyEventsButton.disabled = !state.capabilities.text_events || Boolean(state.pendingCatalogRequestId);
  if (el.resolutionSelect) el.resolutionSelect.disabled = Boolean(state.pendingOutputRequestId);
}

function renderEventTriggers() {
  if (!el.eventTriggerBar) return;
  el.eventTriggerBar.replaceChildren();
  if (!state.textEvents.length) {
    const empty = document.createElement("span");
    empty.className = "event-empty";
    empty.textContent = "NO TEXT EVENTS";
    el.eventTriggerBar.appendChild(empty);
    return;
  }
  for (const event of state.textEvents) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "event-trigger";
    button.textContent = event.label || event.event_id;
    button.dataset.eventId = event.event_id;
    button.classList.toggle("is-active", event.event_id === state.activeEventId);
    button.classList.toggle("is-pending", Boolean(state.pendingEventRequestId));
    button.addEventListener("click", () => {
      const requestId = nextRequestId("event");
      setPending("event", requestId);
      renderEventTriggers();
      if (!send({ type: "event", event_id: event.event_id, state: "trigger", request_id: requestId })) {
        clearPending("event");
        setControlAck("NOT CONNECTED", "error");
        renderEventTriggers();
        return;
      }
      setControlAck("EVENT QUEUED");
    });
    el.eventTriggerBar.appendChild(button);
  }
  const clear = document.createElement("button");
  clear.type = "button";
  clear.className = "event-clear";
  clear.textContent = "CLEAR EVENT";
  clear.addEventListener("click", () => {
    const requestId = nextRequestId("event");
    setPending("event", requestId);
    renderEventTriggers();
    if (send({ type: "event", state: "clear", request_id: requestId })) {
      setControlAck("CLEAR QUEUED");
    } else {
      clearPending("event");
      setControlAck("NOT CONNECTED", "error");
      renderEventTriggers();
    }
  });
  el.eventTriggerBar.appendChild(clear);
  updateControlAvailability();
}

function applyEventCatalog() {
  let events;
  try {
    events = collectTextEvents();
  } catch (error) {
    setCatalogStatus(error.message || String(error), "error");
    return;
  }
  state.textEvents = events;
  renderEventTriggers();
  if (!state.connected) {
    setCatalogStatus(events.length ? "WILL APPLY WHEN SESSION STARTS" : "EMPTY CATALOG", "ok");
    return;
  }
  const requestId = nextRequestId("catalog");
  setPending("catalog", requestId);
  if (send({
    type: "event_catalog",
    events,
    base_revision: state.catalogRevision,
    request_id: requestId,
  })) {
    setCatalogStatus("APPLYING…");
  } else {
    clearPending("catalog");
    setCatalogStatus("NOT CONNECTED", "error");
  }
}

function resolutionValue(resolution) {
  if (!resolution || resolution.mode === "native") return "native";
  return `${Number(resolution.width)}x${Number(resolution.height)}`;
}

function resolutionPayload(value) {
  if (!value || value === "native") return { mode: "native" };
  const match = /^(\d+)x(\d+)$/.exec(value);
  if (!match) return { mode: "native" };
  return { mode: "fixed", width: Number(match[1]), height: Number(match[2]) };
}

function renderResolutionOptions(options) {
  if (!el.resolutionSelect) return;
  el.resolutionSelect.replaceChildren();
  const list = options?.length ? options : [{ mode: "native", label: "Native" }];
  for (const resolution of list) {
    const option = document.createElement("option");
    option.value = resolutionValue(resolution);
    option.textContent = String(resolution.label || (option.value === "native" ? "NATIVE" : option.value));
    el.resolutionSelect.appendChild(option);
  }
  const selected = resolutionValue(state.outputResolution);
  if (![...el.resolutionSelect.options].some((option) => option.value === selected)) {
    const option = document.createElement("option");
    option.value = selected;
    option.textContent = selected.toUpperCase();
    el.resolutionSelect.appendChild(option);
  }
  el.resolutionSelect.value = selected;
}

const PANEL_STORAGE_KEY = "worldfoundry:realtime-panels:v2";

function storedPanelState() {
  try {
    return JSON.parse(localStorage.getItem(PANEL_STORAGE_KEY) || "{}") || {};
  } catch {
    return {};
  }
}

function persistPanels() {
  const panels = {};
  for (const panel of document.querySelectorAll(".studio-panel[data-panel-id]")) {
    panels[panel.dataset.panelId] = {
      collapsed: panel.classList.contains("is-collapsed"),
      floating: panel.classList.contains("is-floating"),
      left: Number.parseFloat(panel.style.left) || 0,
      top: Number.parseFloat(panel.style.top) || 0,
      width: Number.parseFloat(panel.style.width) || 0,
    };
  }
  try {
    localStorage.setItem(PANEL_STORAGE_KEY, JSON.stringify(panels));
  } catch {
    // Private browsing can disable storage without affecting panel controls.
  }
}

function bringPanelFront(panel) {
  state.panelZ += 1;
  if (state.panelZ > 160) {
    state.panelZ = 70;
    for (const item of document.querySelectorAll(".studio-panel.is-floating")) {
      item.style.zIndex = String(++state.panelZ);
    }
  }
  panel.style.zIndex = String(state.panelZ);
}

function clampFloatingPanel(panel) {
  if (!panel.classList.contains("is-floating") || window.matchMedia("(max-width: 820px)").matches) return;
  let rect = panel.getBoundingClientRect();
  const maxWidth = Math.max(window.innerWidth - 16, 1);
  if (rect.width > maxWidth) {
    panel.style.width = `${maxWidth}px`;
    rect = panel.getBoundingClientRect();
  }
  const left = Math.min(Math.max(rect.left, 8), Math.max(window.innerWidth - rect.width - 8, 8));
  const top = Math.min(Math.max(rect.top, 8), Math.max(window.innerHeight - rect.height - 8, 8));
  panel.style.left = `${left}px`;
  panel.style.top = `${top}px`;
}

function setupPanels() {
  const saved = storedPanelState();
  const mobileQuery = window.matchMedia("(max-width: 820px)");
  for (const panel of document.querySelectorAll(".studio-panel[data-panel-id]")) {
    const panelId = panel.dataset.panelId;
    const record = saved[panelId] || {};
    const collapse = panel.querySelector(":scope > .studio-panel-header [data-collapse-panel], :scope > .runtime-log-header [data-collapse-panel]");
    const handle = panel.querySelector(":scope > .panel-handle, :scope > .studio-panel-header");
    const setCollapsed = (collapsed) => {
      panel.classList.toggle("is-collapsed", collapsed);
      collapse?.setAttribute("aria-expanded", String(!collapsed));
      if (collapse) collapse.textContent = collapsed ? "+" : "−";
      requestAnimationFrame(() => clampFloatingPanel(panel));
    };
    setCollapsed(Boolean(record.collapsed));
    if (record.floating && Number.isFinite(record.left) && Number.isFinite(record.top)) {
      panel.classList.add("is-floating");
      panel.style.left = `${record.left}px`;
      panel.style.top = `${record.top}px`;
      if (Number(record.width) > 0) panel.style.width = `${record.width}px`;
      if (!mobileQuery.matches) {
        bringPanelFront(panel);
        requestAnimationFrame(() => clampFloatingPanel(panel));
      }
    }
    collapse?.addEventListener("pointerdown", (event) => event.stopPropagation());
    collapse?.addEventListener("click", (event) => {
      event.stopPropagation();
      setCollapsed(!panel.classList.contains("is-collapsed"));
      persistPanels();
    });
    panel.addEventListener("pointerdown", () => bringPanelFront(panel), { capture: true });
    if (!handle) continue;
    let drag = null;
    let dragRaf = 0;
    const paintDrag = () => {
      dragRaf = 0;
      if (!drag) return;
      panel.style.transform = `translate3d(${drag.dx}px, ${drag.dy}px, 0)`;
    };
    handle.addEventListener("pointerdown", (event) => {
      if (mobileQuery.matches || event.button !== 0 || event.target.closest("button,input,textarea,select,label,.stick-well")) return;
      event.preventDefault();
      bringPanelFront(panel);
      const before = panel.getBoundingClientRect();
      const width = Math.min(before.width, window.innerWidth - 16, 672);
      panel.classList.add("is-floating");
      panel.style.width = `${width}px`;
      panel.style.left = `${Math.min(Math.max(before.left, 8), Math.max(window.innerWidth - width - 8, 8))}px`;
      panel.style.top = `${Math.min(Math.max(before.top, 8), Math.max(window.innerHeight - before.height - 8, 8))}px`;
      panel.style.transform = "translate3d(0, 0, 0)";
      const rect = panel.getBoundingClientRect();
      drag = {
        pointerId: event.pointerId,
        originX: event.clientX,
        originY: event.clientY,
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height,
        dx: 0,
        dy: 0,
      };
      handle.setPointerCapture(event.pointerId);
    });
    handle.addEventListener("pointermove", (event) => {
      if (!drag || event.pointerId !== drag.pointerId) return;
      const wantedX = event.clientX - drag.originX;
      const wantedY = event.clientY - drag.originY;
      drag.dx = Math.min(Math.max(wantedX, 8 - drag.left), window.innerWidth - 8 - drag.width - drag.left);
      drag.dy = Math.min(Math.max(wantedY, 8 - drag.top), window.innerHeight - 8 - drag.height - drag.top);
      if (!dragRaf) dragRaf = requestAnimationFrame(paintDrag);
    });
    const finishDrag = (event) => {
      if (!drag || event.pointerId !== drag.pointerId) return;
      if (dragRaf) cancelAnimationFrame(dragRaf);
      panel.style.left = `${drag.left + drag.dx}px`;
      panel.style.top = `${drag.top + drag.dy}px`;
      panel.style.transform = "";
      drag = null;
      dragRaf = 0;
      persistPanels();
    };
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
      handle.addEventListener(name, finishDrag);
    }
  }
  let resizeRaf = 0;
  window.addEventListener("resize", () => {
    if (resizeRaf) return;
    resizeRaf = requestAnimationFrame(() => {
      resizeRaf = 0;
      document.querySelectorAll(".studio-panel.is-floating").forEach(clampFloatingPanel);
      persistPanels();
    });
  });
}

function setLogOpen(open) {
  state.logOpen = Boolean(open);
  el.root.classList.toggle("log-open", state.logOpen);
  el.logToggleButton?.setAttribute("aria-pressed", String(state.logOpen));
  el.runtimeLog?.setAttribute("aria-hidden", String(!state.logOpen));
  if (state.logOpen) requestAnimationFrame(() => {
    if (el.runtimeLogList) el.runtimeLogList.scrollTop = el.runtimeLogList.scrollHeight;
  });
}

function applyImmersive(active) {
  const wasImmersive = state.immersive;
  if (active && !wasImmersive) state.logBeforeImmersive = state.logOpen;
  state.immersive = Boolean(active);
  el.root.classList.toggle("is-immersive", state.immersive);
  el.fullscreenButton?.setAttribute("aria-pressed", String(state.immersive));
  if (el.fullscreenLabel) el.fullscreenLabel.textContent = state.immersive ? "EXIT" : "FULLSCREEN";
  if (state.immersive) setLogOpen(true);
  else if (wasImmersive) setLogOpen(state.logBeforeImmersive);
}

async function toggleImmersive(force) {
  const active = typeof force === "boolean" ? force : !state.immersive;
  if (active === state.immersive && active === Boolean(document.fullscreenElement)) return;
  applyImmersive(active);
  if (active) {
    appendRuntimeLog("Immersive viewport enabled", "live", "Live log shares the existing realtime channel.");
    if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
      state.nativeFullscreen = true;
      try {
        try {
          await document.documentElement.requestFullscreen({ navigationUI: "hide" });
        } catch (error) {
          if (error instanceof TypeError) await document.documentElement.requestFullscreen();
          else throw error;
        }
      } catch (error) {
        state.nativeFullscreen = false;
        appendRuntimeLog("Browser fullscreen unavailable", "warn", "Using viewport-filling mode instead.");
      }
    }
    return;
  }
  appendRuntimeLog("Immersive viewport disabled");
  state.nativeFullscreen = false;
  if (document.fullscreenElement && document.exitFullscreen) {
    await document.exitFullscreen().catch(() => {});
  }
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (!(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (studioAuthToken) headers.set("Authorization", `Bearer ${studioAuthToken}`);
  const response = await fetch(authenticatedUrl(path), { ...options, headers });
  if (!response.ok) {
    let message = await response.text();
    try {
      const parsed = JSON.parse(message);
      message = parsed.error || parsed.message || message;
    } catch {
      // Keep the server response.
    }
    throw new Error(message || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function setStatus(text, live = false, error = false) {
  el.livePill.textContent = text;
  el.livePill.title = text;
  el.livePill.classList.toggle("is-live", live);
  el.livePill.style.color = error ? "var(--danger)" : "";
  el.overlayStatus.textContent = text;
  if (text !== state.lastLoggedStatus) {
    appendRuntimeLog(text, error ? "error" : live ? "live" : "info");
    state.lastLoggedStatus = text;
  }
}

function setMode(mode) {
  if (state.connected || state.connecting) return;
  state.mode = mode === "video" ? "video" : "image";
  el.root.classList.toggle("image-mode", state.mode === "image");
  el.root.classList.toggle("video-mode", state.mode === "video");
  document.querySelectorAll(".mode-tab").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === state.mode);
  });
}

function showImage(src) {
  resetSocketPlayback();
  el.video.pause();
  el.video.srcObject = null;
  el.video.removeAttribute("src");
  el.video.load();
  el.video.classList.remove("is-visible");
  el.canvas.classList.remove("is-visible");
  el.image.src = authenticatedUrl(src) || emptyFrame;
  el.image.classList.add("is-visible");
}

function resetSocketPlayback() {
  setSensorLayout([]);
  state.hiddenSensors.clear();
  state.focusedSensor = "";
  state.socketPlaybackEpoch += 1;
  state.socketFrameQueue.length = 0;
  state.nextPresentationAt = 0;
  state.segmentPresentedFrames = 0;
  state.pendingChunkDone = null;
  state.pendingChunkDones.length = 0;
  state.inflightSegments = 0;
  state.lastVideoFrameAt = 0;
  state.stats.fps = 0;
}

function sensorLayoutSignature(layout) {
  if (!Array.isArray(layout) || !layout.length) return "";
  return JSON.stringify(
    layout.map((view) => [
      String(view.key ?? ""),
      String(view.label ?? ""),
      String(view.kind ?? ""),
      Number(view.channel_index ?? 0),
      Number(view.x ?? 0),
      Number(view.y ?? 0),
      Number(view.width ?? 0),
      Number(view.height ?? 0),
    ]),
  );
}

function setSensorLayout(layout) {
  const next = Array.isArray(layout) ? layout : [];
  if (sensorLayoutSignature(next) === sensorLayoutSignature(state.sensorLayout)) {
    return;
  }
  state.sensorLayout = next;
  if (!state.sensorLayout.some((view) => view.key === state.focusedSensor)) state.focusedSensor = "";
  state.sensorCanvases.clear();
  el.sensorGrid.replaceChildren();
  el.sensorControls.replaceChildren();
  el.sensorGrid.classList.toggle("is-visible", state.sensorLayout.length > 0);
  if (!state.sensorLayout.length) return;
  const focus = document.createElement("select");
  focus.setAttribute("aria-label", "Focus sensor");
  focus.add(new Option("All sensors", ""));
  for (const view of state.sensorLayout) {
    focus.add(new Option(view.label, view.key));
    const tile = document.createElement("div");
    tile.className = "sensor-tile";
    const canvas = document.createElement("canvas");
    const label = document.createElement("span");
    label.textContent = view.label;
    tile.append(canvas, label);
    el.sensorGrid.append(tile);
    state.sensorCanvases.set(view.key, { canvas, tile });
    const control = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = !state.hiddenSensors.has(view.key);
    checkbox.onchange = () => {
      if (checkbox.checked) state.hiddenSensors.delete(view.key);
      else state.hiddenSensors.add(view.key);
      drawSensorViews(el.video.srcObject ? el.video : el.canvas);
    };
    control.append(checkbox, document.createTextNode(view.label));
    el.sensorControls.append(control);
  }
  focus.value = state.focusedSensor;
  focus.onchange = () => {
    state.focusedSensor = focus.value;
    drawSensorViews(el.video.srcObject ? el.video : el.canvas);
  };
  el.sensorControls.prepend(focus);
  drawSensorViews(el.video.srcObject ? el.video : el.canvas);
}

function drawSensorViews(source) {
  const width = source.videoWidth || source.width;
  const height = source.videoHeight || source.height;
  const visible = state.sensorLayout.filter((view) =>
    !state.hiddenSensors.has(view.key) && (!state.focusedSensor || state.focusedSensor === view.key));
  el.sensorGrid.style.gridTemplateColumns = `repeat(${Math.max(1, Math.ceil(Math.sqrt(visible.length)))}, minmax(0, 1fr))`;
  for (const view of state.sensorLayout) {
    const { canvas, tile } = state.sensorCanvases.get(view.key);
    tile.style.display = visible.includes(view) ? "block" : "none";
    if (!width || !height || !visible.includes(view)) continue;
    const w = Math.max(1, Math.round(view.width * width));
    const h = Math.max(1, Math.round(view.height * height));
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    canvas.getContext("2d", { alpha: false }).drawImage(source,
      view.x * width, view.y * height, view.width * width, view.height * height, 0, 0, w, h);
  }
}

function waitForAnimationFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}

function waitMilliseconds(delay) {
  return delay > 0
    ? new Promise((resolve) => window.setTimeout(resolve, delay))
    : Promise.resolve();
}

function finishSegmentPlayback(message, keepBusy = false) {
  state.segmentIndex = Number(message.chunk_index || state.segmentIndex + 1);
  if (keepBusy) {
    state.segmentBusy = true;
    if (state.continuous) {
      setStatus(continuousStatus(), true);
    } else {
      setStatus(`GENERATING SEGMENT ${state.segmentIndex + 1}`, true);
    }
  } else {
    state.segmentBusy = false;
    if (state.queuedSegments) {
      el.startButton.classList.remove("is-hidden");
      el.startButton.disabled = false;
      el.startButton.textContent = "EXTEND";
      clearQueuedControlSelection();
      setStatus(`SEGMENT ${state.segmentIndex} READY`, true);
    } else if (state.continuous) {
      setStatus(continuousStatus(), true);
    } else {
      setStatus(`SEGMENT ${state.segmentIndex} READY · EDIT PROMPT + ENTER`, true);
    }
  }
  updateControlAvailability();
  renderStats();
}

function drainPendingChunkCompletions() {
  while (state.pendingChunkDones.length) {
    const pending = state.pendingChunkDones[0];
    const required = Math.max(Number(pending.frames || 0), 0);
    if (state.segmentPresentedFrames < required) {
      state.pendingChunkDone = pending;
      return false;
    }
    state.pendingChunkDones.shift();
    state.pendingChunkDone = state.pendingChunkDones[0] || null;
    const hasMore = state.pendingChunkDones.length > 0 || state.inflightSegments > 0;
    if (hasMore) {
      state.segmentPresentedFrames = Math.max(state.segmentPresentedFrames - required, 0);
    }
    finishSegmentPlayback(pending, hasMore);
  }
  return true;
}

async function pumpSocketFrames(epoch) {
  try {
    const segmentMode = state.queuedSegments || state.promptScheduled;
    if (segmentMode && state.segmentPresentedFrames === 0) {
      // A tiny decode buffer smooths TCP/SSH bursts without adding control
      // latency to keyboard-driven worlds.
      await waitMilliseconds(Math.min(160, 2000 / Math.max(state.playbackFps, 1)));
    }
    while (epoch === state.socketPlaybackEpoch && state.socketFrameQueue.length) {
      const blob = state.socketFrameQueue.shift();
      let bitmap;
      try {
        bitmap = await createImageBitmap(blob);
      } catch (error) {
        if (epoch !== state.socketPlaybackEpoch) break;
        console.error(`JPEG decode failed: ${error.message || error}`);
        state.stats.dropped += 1;
        continue;
      }
      if (epoch !== state.socketPlaybackEpoch) {
        bitmap.close();
        break;
      }
      const now = performance.now();
      const interval = 1000 / Math.max(state.playbackFps, 1);
      if (!state.nextPresentationAt || now - state.nextPresentationAt > interval * 2) {
        state.nextPresentationAt = now;
      }
      await waitMilliseconds(Math.max(state.nextPresentationAt - now, 0));
      if (epoch !== state.socketPlaybackEpoch) {
        bitmap.close();
        break;
      }
      state.nextPresentationAt += interval;
      if (el.canvas.width !== bitmap.width) el.canvas.width = bitmap.width;
      if (el.canvas.height !== bitmap.height) el.canvas.height = bitmap.height;
      const context = el.canvas.getContext("2d", { alpha: false });
      context.drawImage(bitmap, 0, 0);
      drawSensorViews(el.canvas);
      bitmap.close();
      el.video.pause();
      el.video.srcObject = null;
      el.video.classList.remove("is-visible");
      el.image.classList.remove("is-visible");
      el.canvas.classList.add("is-visible");
      const presentedAt = await waitForAnimationFrame();
      if (epoch !== state.socketPlaybackEpoch) break;
      if (state.lastVideoFrameAt > 0) {
        const instant = 1000 / Math.max(presentedAt - state.lastVideoFrameAt, 1);
        state.stats.fps = state.stats.fps
          ? state.stats.fps * 0.82 + instant * 0.18
          : instant;
      }
      state.lastVideoFrameAt = presentedAt;
      state.presentedFrames += 1;
      state.segmentPresentedFrames += 1;
      window.__worldfoundryPresentedFrames = state.presentedFrames;
      renderStats();
      drainPendingChunkCompletions();
    }
  } finally {
    state.socketFramePump = false;
    if (state.socketFrameQueue.length) startSocketFramePump();
  }
}

function startSocketFramePump() {
  if (state.socketFramePump || !state.socketFrameQueue.length) return;
  state.socketFramePump = true;
  pumpSocketFrames(state.socketPlaybackEpoch).catch((error) => {
    state.socketFramePump = false;
    setStatus(`PLAYBACK ERROR: ${error.message || error}`, false, true);
    console.error(error);
  });
}

function showSocketFrame(blob) {
  const segmentMode = state.queuedSegments || state.promptScheduled;
  const maxQueued = segmentMode ? 256 : 2;
  while (state.socketFrameQueue.length >= maxQueued) {
    state.socketFrameQueue.shift();
    state.stats.dropped += 1;
  }
  state.socketFrameQueue.push(blob);
  startSocketFramePump();
}

function showLocalVideo(src) {
  resetSocketPlayback();
  el.image.classList.remove("is-visible");
  el.canvas.classList.remove("is-visible");
  el.video.srcObject = null;
  el.video.src = authenticatedUrl(src);
  el.video.loop = true;
  el.video.muted = true;
  el.video.classList.add("is-visible");
  el.video.play().catch(() => {});
}

function revealRemoteVideo() {
  if (!el.video.srcObject || el.video.classList.contains("is-visible")) return;
  el.image.classList.remove("is-visible");
  el.canvas.classList.remove("is-visible");
  el.video.classList.add("is-visible");
}

function attachRemoteStream(stream) {
  resetSocketPlayback();
  el.video.removeAttribute("src");
  el.video.srcObject = stream;
  el.video.loop = false;
  el.video.muted = true;
  // Keep the local seed visible while the first model chunk is in flight.
  // Switching here would expose a blank video element for the full generation
  // latency and an initial seed frame could change the track resolution.
  el.video.classList.remove("is-visible");
  el.video.addEventListener("loadeddata", revealRemoteVideo, { once: true });
  el.video.play().catch(() => {});
  watchVideoFrames();
}

function watchVideoFrames() {
  if (typeof el.video.requestVideoFrameCallback !== "function") return;
  const stream = el.video.srcObject;
  const onFrame = (now, metadata) => {
    if (el.video.srcObject !== stream) return;
    state.presentation.presented(metadata.rtpTimestamp, now);
    revealRemoteVideo();
    drawSensorViews(el.video);
    if (state.lastVideoFrameAt > 0) {
      const instant = 1000 / Math.max(now - state.lastVideoFrameAt, 1);
      state.stats.fps = state.stats.fps ? state.stats.fps * 0.82 + instant * 0.18 : instant;
    }
    state.lastVideoFrameAt = now;
    renderStats();
    if (el.video.srcObject) el.video.requestVideoFrameCallback(onFrame);
  };
  el.video.requestVideoFrameCallback(onFrame);
}

function renderStats() {
  const label = state.transport || (state.preferWebSocket ? "WS" : "RTC");
  const transport = state.connected ? label : state.connecting ? `${label}…` : `${label} OFF`;
  const fps = state.stats.fps ? `${state.stats.fps.toFixed(1)} FPS` : "-- FPS";
  const gen = state.stats.generationMs
    ? state.stats.generationMs >= 10000
      ? `${(state.stats.generationMs / 1000).toFixed(1)}s GEN`
      : `${Math.round(state.stats.generationMs)}ms GEN`
    : "-- GEN";
  const latency = state.presentationSummary
    ? `${Math.round(state.presentationSummary.p50_ms)}ms E2E`
    : "-- E2E";
  const resolution = resolutionValue(state.outputResolution) === "native"
    ? "NATIVE"
    : resolutionValue(state.outputResolution).replace("x", "×");
  el.streamStats.textContent = state.continuous
    ? `${transport} / ${fps} / ${gen} / CHUNK ${rolloutProgress()} / ${resolution}`
    : state.queuedSegments || state.promptScheduled
      ? `${transport} / ${fps} / ${gen} / SEG ${state.segmentIndex} / ${resolution}`
      : `${transport} / ${fps} / ${gen} / ${latency} / ${resolution}`;
  const p90 = state.presentationSummary ? `${Math.round(state.presentationSummary.p90_ms)}ms` : "unavailable";
  const net = state.transportStats;
  el.streamStats.title = `control→present p90=${p90} · server control→queue=${Math.round(state.stats.latencyMs)}ms · output=${resolution} · enqueue=${Math.round(state.stats.enqueueMs)}ms queue=${state.stats.queueDepth} dropped=${state.stats.dropped} · RTT=${net.rtt_ms?.toFixed(1) ?? "--"}ms jitter-buffer=${net.jitter_buffer_ms?.toFixed(1) ?? "--"}ms decode=${net.decode_ms?.toFixed(1) ?? "--"}ms`;
  el.streamStats.classList.toggle("is-live", state.connected);
}

function normalizeKey(key) {
  const normalized = String(key || "").toLowerCase();
  return keyAliases.get(normalized) || normalized;
}

function withPresentationControlId(payload) {
  if (state.transport !== "RTC" || !["action", "step", "event", "prompt_update", "axis"].includes(payload.type)) return payload;
  const prefix = payload.type === "axis" ? `axis-${payload.source}` : "control";
  const control_id = nextRequestId(prefix);
  state.presentation.submitted(control_id, performance.now());
  return {...payload, control_id};
}

function send(payload) {
  if (!state.channel) return false;
  const isOpen = state.channel instanceof WebSocket
    ? state.channel.readyState === WebSocket.OPEN
    : state.channel.readyState === "open";
  if (!isOpen) return false;
  payload = withPresentationControlId(payload);
  try {
    state.channel.send(JSON.stringify(payload));
    return true;
  } catch (error) {
    appendRuntimeLog("Realtime control send failed", "error", error.message || String(error));
    window.queueMicrotask(() => markDisconnected("CONTROL CHANNEL CLOSED"));
    return false;
  }
}

function setKeyHeld(key, source, held) {
  const normalized = normalizeKey(key);
  if (!allowedKeys.has(normalized)) return;
  let sources = state.keySources.get(normalized);
  if (!sources) {
    sources = new Set();
    state.keySources.set(normalized, sources);
  }
  const wasActive = sources.size > 0;
  if (held) sources.add(source);
  else sources.delete(source);
  const active = sources.size > 0;
  if (active !== wasActive) {
    send({
      type: "action",
      action: { event: active ? "keydown" : "keyup", key: normalized },
    });
  }
}

function releaseAllKeys() {
  for (const [key, sources] of state.keySources.entries()) {
    if (!sources.size) continue;
    sources.clear();
    send({ type: "action", action: { event: "keyup", key } });
  }
  releaseJoystick("move");
  releaseJoystick("look");
}

function replayHeldKeys() {
  for (const [key, sources] of state.keySources.entries()) {
    if (sources.size) send({ type: "action", action: { event: "keydown", key } });
  }
}

function updateStickVisual(stick, dx, dy) {
  stick.style.setProperty("--stick-offset-x", `${dx * 1.45}rem`);
  stick.style.setProperty("--stick-offset-y", `${dy * 1.45}rem`);
  stick.classList.toggle("is-active", Math.abs(dx) > 0.05 || Math.abs(dy) > 0.05);
}

function updateStickKeys(kind, desired) {
  const current = state.stickKeys[kind];
  for (const key of current) {
    if (!desired.has(key)) setKeyHeld(key, `stick:${kind}`, false);
  }
  for (const key of desired) {
    if (!current.has(key)) setKeyHeld(key, `stick:${kind}`, true);
  }
  state.stickKeys[kind] = desired;
}

function stickUsesAxis(kind, dimension) {
  return state.gamepadSources.includes(STICK_AXIS_SOURCES[kind][dimension]);
}

function releaseJoystick(kind) {
  const stick = kind === "move" ? el.moveStick : el.lookStick;
  const pointerId = state.stickPointers[kind];
  state.stickPointers[kind] = null;
  state.stickPositions[kind] = { x: 0, y: 0 };
  updateStickKeys(kind, new Set());
  updateStickVisual(stick, 0, 0);
  if (pointerId !== null && stick.hasPointerCapture(pointerId)) stick.releasePointerCapture(pointerId);
}

function bindJoystick(stick, kind) {
  const update = (event) => {
    const rect = stick.getBoundingClientRect();
    const radius = Math.max(Math.min(rect.width, rect.height) / 2, 1);
    let dx = (event.clientX - (rect.left + rect.width / 2)) / radius;
    let dy = (event.clientY - (rect.top + rect.height / 2)) / radius;
    const magnitude = Math.hypot(dx, dy);
    if (magnitude > 1) {
      dx /= magnitude;
      dy /= magnitude;
    }
    updateStickVisual(stick, dx, dy);
    state.stickPositions[kind] = { x: dx, y: dy };
    const desired = new Set();
    const threshold = kind === "move" ? 0.22 : 0.16;
    if (!stickUsesAxis(kind, "y")) {
      if (dy < -threshold) desired.add(kind === "move" ? "w" : "i");
      if (dy > threshold) desired.add(kind === "move" ? "s" : "k");
    }
    if (!stickUsesAxis(kind, "x")) {
      if (dx < -threshold) desired.add(kind === "move" ? "a" : "j");
      if (dx > threshold) desired.add(kind === "move" ? "d" : "l");
    }
    updateStickKeys(kind, desired);
  };
  stick.addEventListener("pointerdown", (event) => {
    if (state.stickPointers[kind] !== null) return;
    event.preventDefault();
    state.stickPointers[kind] = event.pointerId;
    stick.setPointerCapture(event.pointerId);
    update(event);
  });
  stick.addEventListener("pointermove", (event) => {
    if (event.pointerId !== state.stickPointers[kind]) return;
    event.preventDefault();
    update(event);
  });
  for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
    stick.addEventListener(name, (event) => {
      if (event.pointerId === state.stickPointers[kind]) releaseJoystick(kind);
    });
  }
}

function shouldIgnoreKey(event) {
  const target = event.target;
  return target instanceof Element && (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName));
}

function bindKeyboard() {
  window.addEventListener("keydown", (event) => {
    const key = normalizeKey(event.key);
    if (event.defaultPrevented || shouldIgnoreKey(event) || !allowedKeys.has(key) || event.repeat) return;
    event.preventDefault();
    setKeyHeld(key, `keyboard:${key}`, true);
  });
  window.addEventListener("keyup", (event) => {
    const key = normalizeKey(event.key);
    const source = `keyboard:${key}`;
    if (state.keySources.get(key)?.has(source)) {
      setKeyHeld(key, source, false);
      if (!shouldIgnoreKey(event)) event.preventDefault();
      return;
    }
    if (event.defaultPrevented || shouldIgnoreKey(event) || !allowedKeys.has(key)) return;
    event.preventDefault();
    setKeyHeld(key, source, false);
  });
  window.addEventListener("blur", releaseAllInputs);
}

// Adopt the keys, axes and advance mode that the session's RealtimeSpec declares.
function applyRealtimeSpec(spec) {
  if (!spec || typeof spec !== "object") return;
  state.declaredKeys = Array.isArray(spec.keys);
  state.realtimeKeys = state.declaredKeys ? spec.keys : DEFAULT_REALTIME_KEYS;
  allowedKeys.clear();
  // Prompt-scheduled and queued-segment sessions take no keyboard actions.
  if (!state.promptScheduled && !state.queuedSegments) {
    for (const binding of state.realtimeKeys) allowedKeys.add(binding.key);
  }
  const sources = Array.isArray(spec.axes) ? spec.axes.map((axis) => String(axis.source)) : [];
  state.mouseAxes = sources.some((source) => source.startsWith("mouse_"));
  state.gamepadSources = sources.filter((source) => source.startsWith("gamepad_"));
  state.gamepadAxes = state.gamepadSources.length > 0;
  state.continuous = spec.advance === "continuous";
  if (!state.connected) state.maxChunks = Number.isInteger(spec.max_chunks) ? spec.max_chunks : null;
  const sticksEnabled = !state.promptScheduled && !state.queuedSegments;
  el.moveStick.classList.toggle("is-hidden", !sticksEnabled || !(MOVE_STICK_KEYS.some((key) => allowedKeys.has(key))
    || stickUsesAxis("move", "x") || stickUsesAxis("move", "y")));
  el.lookStick.classList.toggle("is-hidden", !sticksEnabled || !(LOOK_STICK_KEYS.some((key) => allowedKeys.has(key))
    || stickUsesAxis("look", "x") || stickUsesAxis("look", "y")));
  renderRollout();
  renderPointerHint();
  if (state.connected) startGamepadPolling();
}

// Empty when the server-rendered hint already describes the session.
function controlLegend() {
  if (state.queuedSegments) return "";
  if (state.promptScheduled) {
    return state.continuous ? "OPTIONAL IMAGE · ENTER A NEW PROMPT TO START A NEW ROLLOUT" : "";
  }
  const parts = [];
  if (!state.declaredKeys) {
    parts.push("HOLD WASD / DRAG STICKS");
  } else if (state.realtimeKeys.length) {
    const sticks = !el.moveStick.classList.contains("is-hidden") || !el.lookStick.classList.contains("is-hidden");
    const keys = state.realtimeKeys.map((binding) => binding.key.toUpperCase()).join(" ");
    parts.push(`HOLD ${keys}${sticks ? " / DRAG STICKS" : ""}`);
  } else if (state.gamepadAxes) {
    parts.push("DRAG STICKS");
  }
  if (state.mouseAxes) parts.push("CLICK VIEW FOR MOUSE");
  // Browsers expose gamepads only on secure origins and after a button press.
  if (state.gamepadAxes && connectedGamepads().length) parts.push("GAMEPAD");
  if (state.continuous) parts.push("NEW PROMPT STARTS A NEW ROLLOUT");
  return parts.join(" · ");
}

function renderControlLegend() {
  if (!el.interactionHint) return;
  const legend = controlLegend();
  if (legend) el.interactionHint.textContent = legend;
  if (!state.promptScheduled && !state.queuedSegments) {
    el.interactionHint.title = state.realtimeKeys
      .map((binding) => `${binding.key.toUpperCase()} ${binding.label || binding.action}`)
      .join(" · ");
  }
}

function pointerLockAvailable() {
  return state.mouseAxes && typeof el.viewport.requestPointerLock === "function";
}

function renderPointerHint() {
  if (!el.pointerHint) return;
  el.pointerHint.classList.toggle("is-hidden", !state.connected || !pointerLockAvailable());
  el.pointerHint.classList.toggle("is-locked", state.pointerLocked);
  el.pointerHint.textContent = state.pointerLocked ? "MOUSE LOOK · ESC TO RELEASE" : "CLICK VIEW FOR MOUSE LOOK";
}

function requestMouseLook(event) {
  if (!state.connected || state.pointerLocked || !pointerLockAvailable()) return;
  if (event.target instanceof Element && event.target.closest("button,select,input,label,.runtime-log,#sensorControls")) return;
  const request = el.viewport.requestPointerLock();
  request?.catch?.((error) => appendRuntimeLog("Pointer lock unavailable", "warn", error.message || String(error)));
}

function onPointerLockChange() {
  state.pointerLocked = document.pointerLockElement === el.viewport;
  // Releasing the lock (Esc) ends mouse look; motion after it is not sent.
  if (!state.pointerLocked) state.mouseDelta = { dx: 0, dy: 0 };
  renderPointerHint();
}

// Mouse deltas ride the unordered "axes" DataChannel on WebRTC and the socket
// otherwise. A congested channel defers the delta to a later animation frame.
function sendMouseDelta(payload) {
  const axisChannel = state.axisChannel?.readyState === "open" ? state.axisChannel : null;
  const channel = axisChannel || state.channel;
  if (!channel || channel.bufferedAmount > AXIS_BUFFERED_LIMIT_BYTES) return false;
  if (!axisChannel) return send(payload);
  try {
    axisChannel.send(JSON.stringify(withPresentationControlId(payload)));
    return true;
  } catch {
    return false;
  }
}

// Mouse motion is summed and sent at most once per animation frame.
function onMouseMove(event) {
  if (!state.pointerLocked) return;
  state.mouseDelta.dx += event.movementX || 0;
  state.mouseDelta.dy += event.movementY || 0;
  if (!state.mouseRaf) state.mouseRaf = requestAnimationFrame(flushMouseDelta);
}

function flushMouseDelta() {
  state.mouseRaf = 0;
  const { dx, dy } = state.mouseDelta;
  if (!state.pointerLocked || (!dx && !dy)) return;
  if (sendMouseDelta({ type: "axis", source: "mouse", dx, dy })) state.mouseDelta = { dx: 0, dy: 0 };
  else if (state.connected) state.mouseRaf = requestAnimationFrame(flushMouseDelta);
}

// The server holds a gamepad level until the next sample, so stick samples use
// the reliable control channel; a congested channel defers them to a later frame.
function sendGamepadSample(sample) {
  if (!state.channel || state.channel.bufferedAmount > AXIS_BUFFERED_LIMIT_BYTES) return false;
  return send({ type: "axis", source: "gamepad", ...sample });
}

function connectedGamepads() {
  if (typeof navigator.getGamepads !== "function") return [];
  return Array.from(navigator.getGamepads()).filter((item) => item?.connected);
}

function currentGamepadSample() {
  const [pad] = connectedGamepads();
  const level = (index) => Number(pad?.axes[index]) || 0;
  const sample = { lx: level(0), ly: level(1), rx: level(2), ry: level(3) };
  // A captured virtual stick owns its declared directions until release;
  // other directions continue to use the physical gamepad.
  for (const [kind, sources] of Object.entries(STICK_AXIS_SOURCES)) {
    if (state.stickPointers[kind] === null) continue;
    for (const [dimension, source] of Object.entries(sources)) {
      if (stickUsesAxis(kind, dimension)) sample[source.slice("gamepad_".length)] = state.stickPositions[kind][dimension];
    }
  }
  return sample;
}

function pollGamepad() {
  state.gamepadRaf = 0;
  if (!state.connected || !state.gamepadAxes) return;
  state.gamepadRaf = requestAnimationFrame(pollGamepad);
  // Focus loss already sent the neutral sample; the physical stick resumes on return.
  if (document.visibilityState !== "visible" || !document.hasFocus()) return;
  const sample = currentGamepadSample();
  const moved = Object.keys(sample).some((name) => Math.abs(sample[name] - state.gamepadSent[name]) > GAMEPAD_EPSILON
    || (sample[name] === 0 && state.gamepadSent[name] !== 0));
  if (moved && sendGamepadSample(sample)) state.gamepadSent = sample;
}

function startGamepadPolling() {
  if (!state.gamepadAxes || state.gamepadRaf) return;
  state.gamepadRaf = requestAnimationFrame(pollGamepad);
}

// The server holds stick levels until reset, so focus loss sends a neutral sample.
function sendNeutralGamepad() {
  if (!state.connected || !state.gamepadAxes) return;
  state.gamepadSent = { lx: 0, ly: 0, rx: 0, ry: 0 };
  send({ type: "axis", source: "gamepad", ...state.gamepadSent });
}

function releaseAllInputs() {
  releaseAllKeys();
  sendNeutralGamepad();
}

function stopAxisInputs() {
  releaseJoystick("move");
  releaseJoystick("look");
  if (state.gamepadRaf) cancelAnimationFrame(state.gamepadRaf);
  state.gamepadRaf = 0;
  state.gamepadSent = { lx: 0, ly: 0, rx: 0, ry: 0 };
  if (state.mouseRaf) cancelAnimationFrame(state.mouseRaf);
  state.mouseRaf = 0;
  state.mouseDelta = { dx: 0, dy: 0 };
  if (state.pointerLocked) document.exitPointerLock();
  renderPointerHint();
}

function bindAxisInputs() {
  el.viewport.addEventListener("click", requestMouseLook);
  document.addEventListener("pointerlockchange", onPointerLockChange);
  document.addEventListener("mousemove", onMouseMove);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") releaseAllInputs();
  });
  window.addEventListener("gamepadconnected", renderControlLegend);
  window.addEventListener("gamepaddisconnected", () => {
    if (document.visibilityState !== "visible" || !document.hasFocus()) {
      sendNeutralGamepad();
    } else if (state.connected && state.gamepadAxes) {
      const sample = currentGamepadSample();
      if (send({ type: "axis", source: "gamepad", ...sample })) state.gamepadSent = sample;
    }
    renderControlLegend();
  });
}

function rolloutComplete() {
  return state.continuous && state.maxChunks !== null && state.rolloutChunks >= state.maxChunks;
}

// "3/7" for a bounded rollout, a running count for an unbounded one.
function rolloutProgress() {
  return state.maxChunks === null ? String(state.rolloutChunks) : `${state.rolloutChunks}/${state.maxChunks}`;
}

function continuousStatus() {
  if (rolloutComplete()) return `ROLLOUT COMPLETE · ${rolloutProgress()} · ENTER A NEW PROMPT`;
  return `${state.paused ? "PAUSED" : "GENERATING"} · CHUNK ${rolloutProgress()}`;
}

// Continuous sessions report progress in ready, chunk_done and pause/resume acks.
function applyContinuousState(message) {
  state.continuous = message.advance === "continuous";
  state.rolloutChunks = Number(message.rollout_chunks || 0);
  state.maxChunks = Number.isInteger(message.max_chunks) ? message.max_chunks : null;
  state.paused = Boolean(message.paused);
  renderRollout();
}

function renderRollout() {
  if (el.pauseButton) {
    el.pauseButton.classList.toggle("is-hidden", !state.continuous);
    el.pauseButton.textContent = state.paused ? "RESUME" : "PAUSE";
    el.pauseButton.setAttribute("aria-pressed", String(state.paused));
  }
  if (state.continuous && state.promptScheduled) {
    el.promptInput.title = "Enter a new prompt to start a new rollout";
  }
  renderControlLegend();
  updateControlAvailability();
  renderStats();
}

function togglePause() {
  const type = state.paused ? "resume" : "pause";
  const requestId = nextRequestId("pause");
  setPending("pause", requestId);
  if (send({ type, request_id: requestId })) {
    setControlAck(`${type.toUpperCase()} QUEUED`);
  } else {
    clearPending("pause");
    setControlAck("NOT CONNECTED", "error");
  }
}

async function waitForIce(pc) {
  if (pc.iceGatheringState === "complete") return;
  await Promise.race([
    new Promise((resolve) => {
      const onChange = () => {
        if (pc.iceGatheringState !== "complete") return;
        pc.removeEventListener("icegatheringstatechange", onChange);
        resolve();
      };
      pc.addEventListener("icegatheringstatechange", onChange);
    }),
    new Promise((_, reject) => window.setTimeout(() => reject(new Error("ICE gathering timed out")), 10000)),
  ]);
}

function resetPresentationMetrics() {
  window.clearInterval(state.transportStatsTimer);
  state.transportStatsTimer = null;
  state.previousTransportStats.clear();
  state.transportStats = {};
  state.presentation.reset();
  state.presentationSummary = null;
  state.lastVideoFrameAt = 0;
}

async function pollTransportStats() {
  const peer = state.peer;
  if (!peer || !state.connected || state.transportStatsPending) return;
  state.transportStatsPending = true;
  try {
    const reports = await peer.getStats();
    if (state.peer !== peer || !state.connected) return;
    const metrics = {};
    for (const report of reports.values()) {
      if (report.type === "transport" && report.selectedCandidatePairId) {
        const pair = reports.get(report.selectedCandidatePairId);
        if (Number.isFinite(pair?.currentRoundTripTime)) metrics.rtt_ms = pair.currentRoundTripTime * 1000;
      }
      if (report.type !== "inbound-rtp" || (report.kind || report.mediaType) !== "video") continue;
      const previous = state.previousTransportStats.get(report.id);
      for (const [source, target] of [["packetsLost", "packets_lost"], ["framesDropped", "frames_dropped"], ["framesDecoded", "frames_decoded"]]) {
        if (Number.isFinite(report[source])) metrics[target] = report[source];
      }
      if (Number.isFinite(report.jitter)) metrics.jitter_ms = report.jitter * 1000;
      if (previous) {
        const elapsed = report.timestamp - previous.timestamp;
        const bytes = report.bytesReceived - previous.bytesReceived;
        if (elapsed > 0 && bytes >= 0) metrics.bitrate_bps = bytes * 8000 / elapsed;
        const emitted = report.jitterBufferEmittedCount - previous.jitterBufferEmittedCount;
        const buffered = report.jitterBufferDelay - previous.jitterBufferDelay;
        if (emitted > 0 && buffered >= 0) metrics.jitter_buffer_ms = buffered * 1000 / emitted;
        const decoded = report.framesDecoded - previous.framesDecoded;
        const decodeTime = report.totalDecodeTime - previous.totalDecodeTime;
        if (decoded > 0 && decodeTime >= 0) metrics.decode_ms = decodeTime * 1000 / decoded;
      }
      state.previousTransportStats.set(report.id, report);
    }
    state.transportStats = metrics;
    send({type: "browser_transport", ...metrics});
    renderStats();
  } catch (error) {
    if (state.peer === peer) console.debug("WebRTC stats unavailable", error);
  } finally {
    state.transportStatsPending = false;
  }
}

function markConnected(transport) {
  window.clearTimeout(state.connectionTimer);
  state.connectionTimer = null;
  resetPresentationMetrics();
  state.transport = transport;
  state.connected = true;
  if (transport === "RTC") state.transportStatsTimer = window.setInterval(pollTransportStats, 2000);
  state.connecting = false;
  state.fallbackStarted = false;
  state.sessionStartedAt = Date.now();
  appendRuntimeLog(`${transport} channel connected`, "live", `Playback target ${state.playbackFps} FPS`);
  el.startButton.classList.toggle("is-hidden", !state.queuedSegments);
  el.resetButton.classList.remove("is-hidden");
  if (state.queuedSegments) {
    el.startButton.disabled = state.segmentBusy;
    el.startButton.textContent = state.segmentBusy ? "RUNNING" : state.segmentIndex ? "EXTEND" : "RUN";
    setStatus(state.segmentBusy ? `GENERATING SEGMENT ${state.segmentIndex + 1}` : "CONNECTED", true);
  } else if (state.promptScheduled) {
    setStatus("SESSION READY", true);
  } else {
    setStatus("LIVE", true);
  }
  renderStats();
  updateControlAvailability();
  replayHeldKeys();
  renderPointerHint();
  startGamepadPolling();
  window.clearInterval(state.heartbeatTimer);
  state.heartbeatTimer = window.setInterval(
    () => send({ type: "heartbeat", t: Date.now() }),
    5000,
  );
}

function markDisconnected(message = "DISCONNECTED · START A NEW SESSION") {
  if (state.fallbackStarted && state.connecting) return;
  window.clearInterval(state.heartbeatTimer);
  state.heartbeatTimer = null;
  window.clearTimeout(state.connectionTimer);
  state.connectionTimer = null;
  const axisChannel = state.axisChannel;
  const channel = state.channel;
  const socket = state.socket;
  const peer = state.peer;
  state.axisChannel = null;
  state.channel = null;
  state.socket = null;
  state.peer = null;
  if (axisChannel && axisChannel.readyState !== "closed") axisChannel.close();
  if (channel && channel !== socket && channel.readyState !== "closed") channel.close();
  if (socket && socket.readyState !== (WebSocket.CLOSED ?? 3) && socket.readyState !== "closed") socket.close();
  if (peer && peer.connectionState !== "closed") peer.close();
  state.connected = false;
  state.connecting = false;
  state.fallbackStarted = false;
  state.segmentBusy = false;
  state.sessionStartedAt = 0;
  state.segmentIndex = 0;
  state.uploadedPath = "";
  state.denseUploadedPath = "";
  state.sparseUploadedPath = "";
  state.inflightDenseFile = null;
  state.inflightDensePath = "";
  state.inflightSparseFile = null;
  state.inflightSparsePath = "";
  resetPresentationMetrics();
  resetSocketPlayback();
  clearAllPending();
  for (const sources of state.keySources.values()) sources.clear();
  stopAxisInputs();
  state.rolloutChunks = 0;
  state.paused = false;
  renderRollout();
  el.startButton.classList.remove("is-hidden");
  el.startButton.disabled = false;
  el.startButton.textContent = state.queuedSegments ? "RUN" : "START";
  el.resetButton.classList.add("is-hidden");
  setStatus(message, false, true);
  setControlAck("DISCONNECTED", "error");
  renderEventTriggers();
  updateControlAvailability();
  renderStats();
}

function handleServerMessage(raw) {
  let message;
  try {
    message = JSON.parse(raw);
  } catch {
    return;
  }
  if (message.type === "frame_sent") {
    state.presentation.frameSent(message);
  } else if (message.type === "presentation_timing") {
    state.presentationSummary = message.control_to_display;
    renderStats();
  } else if (message.type === "log") {
    appendRuntimeLog(message.message || message.event || "Runtime event", message.level, message.detail);
  } else if (message.type === "configuring") {
    setStatus("PREPARING SESSION", true);
  } else if (message.type === "ready") {
    state.playbackFps = Math.max(Number(message.fps || state.playbackFps), 1);
    if (Array.isArray(message.event_catalog)) state.textEvents = message.event_catalog;
    state.catalogRevision = Number(message.catalog_revision ?? state.catalogRevision);
    state.activeEventId = message.active_event_id || null;
    if (message.resolution) state.outputResolution = message.resolution;
    renderEventTriggers();
    renderResolutionOptions(state.outputResolutionOptions);
    applyContinuousState(message);
    if (state.transport === "WS") markConnected("WS");
    if (state.continuous && !state.queuedSegments && !state.promptScheduled) setStatus(continuousStatus(), true);
    else if (!state.queuedSegments && !state.promptScheduled) setStatus("LIVE", true);
  } else if (message.type === "event_catalog_ack") {
    if (!acceptAck("catalog", message)) return;
    if (!message.ok) {
      if (Array.isArray(message.event_catalog)) {
        state.textEvents = message.event_catalog;
        replaceEventRows(state.textEvents);
        renderEventTriggers();
      }
      state.catalogRevision = Number(message.catalog_revision ?? state.catalogRevision);
      state.activeEventId = message.active_event_id || null;
      renderEventTriggers();
      setCatalogStatus(message.message || "EVENT CATALOG REJECTED", "error");
      appendRuntimeLog("Text event catalog rejected", "error", message.message || "Invalid catalog");
      return;
    }
    state.textEvents = Array.isArray(message.event_catalog) ? message.event_catalog : state.textEvents;
    state.catalogRevision = Number(message.catalog_revision || state.catalogRevision);
    state.activeEventId = message.active_event_id || null;
    setCatalogStatus(`REV ${state.catalogRevision} ACKNOWLEDGED`, "ok");
    appendRuntimeLog("Text event catalog applied", "live", `${state.textEvents.length} event(s) · revision ${state.catalogRevision}`);
    renderEventTriggers();
  } else if (message.type === "event_ack") {
    if (!acceptAck("event", message)) return;
    if (!message.ok) {
      setControlAck("EVENT REJECTED", "error");
      appendRuntimeLog("Text event rejected", "error", message.message || "Unknown event");
      renderEventTriggers();
      return;
    }
    state.activeEventId = message.active_event_id || null;
    state.catalogRevision = Number(message.catalog_revision || state.catalogRevision);
    setControlAck(state.activeEventId ? "EVENT ACK" : "CLEAR ACK", "ok");
    appendRuntimeLog(
      state.activeEventId ? `Text event ${state.activeEventId} accepted` : "Text event cleared",
      "live",
      `Applies at ${String(message.applies_at || "next_chunk").replace("_", " ")}`,
    );
    renderEventTriggers();
  } else if (message.type === "step_ack") {
    if (!acceptAck("step", message)) return;
    if (!message.ok) {
      setControlAck(message.message || "STEP REJECTED", "error");
      appendRuntimeLog("Idle step rejected", "error", message.message || "Unsupported");
    } else {
      setControlAck("STEP ACK", "ok");
      appendRuntimeLog("Idle step queued", "live", `Queue depth ${Number(message.queued_steps || 1)}`);
    }
  } else if (message.type === "pause_ack" || message.type === "resume_ack") {
    if (!acceptAck("pause", message)) return;
    if (!message.ok) {
      setControlAck(message.message || "PAUSE REJECTED", "error");
      appendRuntimeLog("Pause/resume rejected", "error", message.message || "Unsupported");
      return;
    }
    applyContinuousState(message);
    setControlAck(state.paused ? "PAUSED" : "RESUMED", "ok");
    appendRuntimeLog(state.paused ? "Rollout paused" : "Rollout resumed", "live", `Chunk ${rolloutProgress()}`);
    setStatus(continuousStatus(), true);
  } else if (message.type === "output_config_ack") {
    if (!acceptAck("output", message)) return;
    if (!message.ok) {
      if (message.resolution) state.outputResolution = message.resolution;
      renderResolutionOptions(state.outputResolutionOptions);
      renderStats();
      setControlAck("OUTPUT REJECTED", "error");
      appendRuntimeLog("Output resolution rejected", "error", message.message || "Invalid resolution");
      return;
    }
    state.outputResolution = message.resolution || state.outputResolution;
    renderResolutionOptions(state.outputResolutionOptions);
    renderStats();
    const queued = message.status === "queued";
    setControlAck(queued ? "OUTPUT QUEUED" : "OUTPUT ACK", "ok");
    appendRuntimeLog(
      queued ? "Transport resolution queued" : "Transport resolution accepted",
      "live",
      `${resolutionValue(state.outputResolution)} · ${String(message.applies_at || "next_chunk").replace("_", " ")}`,
    );
  } else if (message.type === "segment_queued") {
    state.segmentBusy = true;
    el.startButton.disabled = true;
    el.startButton.textContent = "QUEUED";
    setStatus(`SEGMENT ${message.segment_index || state.segmentIndex + 1} QUEUED`, true);
    updateControlAvailability();
  } else if (message.type === "segment_started") {
    const idle = !state.pendingChunkDones.length && state.inflightSegments === 0 && !state.socketFrameQueue.length;
    state.inflightSegments += 1;
    state.segmentBusy = true;
    if (idle) {
      state.segmentPresentedFrames = 0;
      state.pendingChunkDone = null;
      state.nextPresentationAt = 0;
    }
    if (state.queuedSegments) {
      el.startButton.disabled = true;
      el.startButton.textContent = "RUNNING";
    }
    if (idle) {
      setStatus(
        state.continuous ? "GENERATING NEXT CHUNK" : `GENERATING SEGMENT ${message.segment_index || state.segmentIndex + 1}`,
        true,
      );
    }
    updateControlAvailability();
  } else if (message.type === "segment_progress") {
    const seconds = Math.max(Number(message.elapsed_ms || 0) / 1000, 0);
    const label = state.continuous ? "NEXT CHUNK" : `SEGMENT ${message.segment_index || state.segmentIndex + 1}`;
    setStatus(`GENERATING ${label} · ${seconds.toFixed(0)}s`, true);
  } else if (message.type === "chunk_done") {
    setSensorLayout(message.sensor_layout);
    applyContinuousState(message);
    state.stats.generationMs = Number(message.generation_ms || 0);
    state.stats.enqueueMs = Number(message.enqueue_ms || 0);
    state.stats.latencyMs = Number(message.control_latency_ms || 0);
    state.stats.queueDepth = Number(message.queue_depth || 0);
    state.stats.dropped = Number(message.dropped_frames || 0);
    if (message.resolution) {
      const previous = resolutionValue(state.outputResolution);
      state.outputResolution = message.resolution;
      const current = resolutionValue(state.outputResolution);
      if (current !== previous || (el.resolutionSelect && el.resolutionSelect.value !== current)) {
        renderResolutionOptions(state.outputResolutionOptions);
      }
    }
    const chunkIndex = Number(message.chunk_index || state.segmentIndex + 1);
    const frames = Number(message.frames || 0);
    const generation = state.stats.generationMs >= 10000
      ? `${(state.stats.generationMs / 1000).toFixed(1)}s`
      : `${Math.round(state.stats.generationMs)}ms`;
    appendRuntimeLog(
      `Chunk ${chunkIndex} ready`,
      "metric",
      `${frames} frames · gen ${generation} · queue ${state.stats.queueDepth} · dropped ${state.stats.dropped}`,
    );
    if (state.queuedSegments || state.promptScheduled) {
      state.inflightSegments = Math.max(state.inflightSegments - 1, 0);
      state.pendingChunkDones.push(message);
      if (!drainPendingChunkCompletions()) {
        const activePending = state.pendingChunkDones[0];
        setStatus(
          `PLAYING SEGMENT ${activePending.chunk_index || state.segmentIndex + 1} · ${state.segmentPresentedFrames}/${activePending.frames || 0}`,
          true,
        );
      }
    } else if (state.continuous) {
      setStatus(continuousStatus(), true);
    } else {
      setStatus("LIVE", true);
    }
    renderStats();
  } else if (message.type === "segment_rejected") {
    state.inflightSegments = Math.max(state.inflightSegments - 1, 0);
    if (state.inflightSegments === 0 && !state.pendingChunkDones.length) {
      state.segmentBusy = false;
    }
    el.startButton.disabled = false;
    el.startButton.textContent = state.segmentIndex ? "EXTEND" : "RUN";
    setStatus(`SEGMENT REJECTED: ${message.message || "invalid inputs"}`, false, true);
    updateControlAvailability();
  } else if (message.type === "error") {
    state.inflightSegments = 0;
    state.segmentBusy = false;
    el.startButton.disabled = false;
    el.startButton.textContent = state.segmentIndex ? "EXTEND" : "RUN";
    setStatus(`STREAM ERROR: ${message.message || "unknown"}`, false, true);
    updateControlAvailability();
    console.error(message.message || message);
  }
}

async function connectSocket(session) {
  state.transport = "WS";
  state.connecting = true;
  setStatus("CONNECTING VIA TUNNEL", true);
  const socketUrl = new URL(authenticatedUrl("/api/realtime/ws"), window.location.href);
  socketUrl.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(socketUrl.toString());
  socket.binaryType = "blob";
  state.socket = socket;
  state.channel = socket;

  await new Promise((resolve, reject) => {
    let settled = false;
    state.connectionTimer = window.setTimeout(() => {
      if (settled || state.connected) return;
      settled = true;
      socket.close();
      reject(new Error("same-port realtime connection timed out"));
    }, SOCKET_SETUP_TIMEOUT_MS);
    socket.addEventListener("open", () => {
      appendRuntimeLog("Realtime socket opened", "live", "Configuring resident session");
      socket.send(JSON.stringify({ type: "configure", session }));
    });
    socket.addEventListener("message", (event) => {
      if (typeof event.data !== "string") {
        showSocketFrame(event.data instanceof Blob ? event.data : new Blob([event.data], { type: "image/jpeg" }));
        return;
      }
      handleServerMessage(event.data);
      try {
        if (!settled && JSON.parse(event.data).type === "ready") {
          settled = true;
          resolve();
        }
      } catch {
        // Ignore malformed telemetry; the server will send an explicit error.
      }
    });
    socket.addEventListener("error", () => {
      if (settled) return;
      settled = true;
      reject(new Error("same-port realtime connection failed"));
    });
    socket.addEventListener("close", () => {
      if (state.socket !== socket) return;
      if (!settled) {
        settled = true;
        reject(new Error("same-port realtime connection closed"));
      }
      markDisconnected();
    });
  });
}

async function fallbackToSocket(session, pc) {
  if (state.fallbackStarted || state.connected || state.peer !== pc) return;
  state.fallbackStarted = true;
  state.connecting = true;
  setStatus("SWITCHING TO TUNNEL MODE", true);
  try {
    await closePeer(false, { keepConnecting: true });
    await api("/api/session/reset", {
      method: "POST",
      body: JSON.stringify({ preserve_uploads: true }),
    }).catch(() => {});
    await connectSocket(session);
  } catch (error) {
    state.fallbackStarted = false;
    markDisconnected(`START ERROR: ${error.message || error}`);
    throw error;
  }
}

async function uploadSelectedInput() {
  if (!state.selectedFile) return state.seedPath;
  if (state.uploadedPath) return state.uploadedPath;
  setStatus("UPLOADING");
  const body = new FormData();
  body.append("file", state.selectedFile, state.selectedFile.name);
  const result = await api("/api/session/input", { method: "POST", body });
  state.uploadedPath = result.path;
  return state.uploadedPath;
}

async function uploadFile(file) {
  const body = new FormData();
  body.append("file", file, file.name);
  return api("/api/session/input", { method: "POST", body });
}

async function uploadQueuedInitialInputs() {
  setStatus("UPLOADING 3 INPUTS", true);
  const imageFile = state.selectedFile;
  const imagePath = state.uploadedPath || state.seedPath;
  const denseFile = state.denseFile;
  const densePath = state.denseUploadedPath;
  const sparseFile = state.sparseFile;
  const sparsePath = state.sparseUploadedPath;
  const uploads = [];
  uploads.push(imagePath ? Promise.resolve({ path: imagePath }) : uploadFile(imageFile));
  uploads.push(densePath ? Promise.resolve({ path: densePath }) : uploadFile(denseFile));
  uploads.push(sparsePath ? Promise.resolve({ path: sparsePath }) : uploadFile(sparseFile));
  const [image, dense, sparse] = await Promise.all(uploads);
  if (state.selectedFile === imageFile && !state.seedPath) state.uploadedPath = image.path;
  if (state.denseFile === denseFile) state.denseUploadedPath = dense.path;
  if (state.sparseFile === sparseFile) state.sparseUploadedPath = sparse.path;
  state.inflightDenseFile = denseFile;
  state.inflightDensePath = dense.path;
  state.inflightSparseFile = sparseFile;
  state.inflightSparsePath = sparse.path;
  return {
    imagePath: image.path,
    densePath: dense.path,
    sparsePath: sparse.path,
  };
}

async function uploadQueuedControls() {
  setStatus("UPLOADING CONTROLS", true);
  const denseFile = state.denseFile;
  const densePath = state.denseUploadedPath;
  const sparseFile = state.sparseFile;
  const sparsePath = state.sparseUploadedPath;
  const [dense, sparse] = await Promise.all([
    densePath ? Promise.resolve({ path: densePath }) : uploadFile(denseFile),
    sparsePath ? Promise.resolve({ path: sparsePath }) : uploadFile(sparseFile),
  ]);
  if (state.denseFile === denseFile) state.denseUploadedPath = dense.path;
  if (state.sparseFile === sparseFile) state.sparseUploadedPath = sparse.path;
  state.inflightDenseFile = denseFile;
  state.inflightDensePath = dense.path;
  state.inflightSparseFile = sparseFile;
  state.inflightSparsePath = sparse.path;
  return { densePath: dense.path, sparsePath: sparse.path };
}

async function loadModel() {
  if (state.modelLoaded) return;
  setStatus("LOADING MODEL");
  el.loadModelButton.disabled = true;
  el.loadModelButton.textContent = "LOADING";
  try {
    await api("/api/runtime/load", { method: "POST", body: "{}" });
    state.modelLoaded = true;
    el.loadModelButton.textContent = "READY";
    setStatus("READY");
    await syncRuntimeStatus().catch(() => false);
  } catch (error) {
    state.modelLoaded = false;
    el.loadModelButton.textContent = "RETRY LOAD";
    setStatus(`LOAD ERROR: ${error.message || error}`, false, true);
    throw error;
  } finally {
    el.loadModelButton.disabled = false;
  }
}

async function connectSession() {
  if (state.connecting || state.connected) return;
  try {
    state.textEvents = collectTextEvents();
  } catch (error) {
    setCatalogStatus(error.message || String(error), "error");
    return;
  }
  if ((state.promptScheduled || state.queuedSegments) && !el.promptInput.value.trim()) {
    setStatus("ENTER PROMPT", false, true);
    el.promptInput.focus();
    return;
  }
  if (state.queuedSegments && (!state.selectedFile && !state.seedPath)) {
    setStatus("SELECT INITIAL IMAGE", false, true);
    return;
  }
  if (state.queuedSegments && !state.denseFile && !state.denseUploadedPath) {
    setStatus("SELECT DENSE DEPTH CONTROL", false, true);
    return;
  }
  if (state.queuedSegments && !state.sparseFile && !state.sparseUploadedPath) {
    setStatus("SELECT SPARSE TRACK CONTROL", false, true);
    return;
  }
  if (state.mode === "image" && !state.seedPath && !state.selectedFile && !state.promptScheduled) {
    setStatus("SELECT IMAGE", false, true);
    return;
  }
  if (state.mode === "video" && !state.seedPath && !state.selectedFile) {
    setStatus("SELECT VIDEO", false, true);
    return;
  }
  if (state.channel || state.peer || state.socket) await closePeer(false);
  state.connecting = true;
  state.segmentBusy = state.queuedSegments;
  setStatus("CONNECTING", true);
  try {
    await loadModel();
    // Preloading resolves the model's RealtimeSpec (keys, axes, advance).
    await syncRuntimeStatus().catch(() => false);
    let session;
    if (state.queuedSegments) {
      const paths = await uploadQueuedInitialInputs();
      session = {
        prompt: el.promptInput.value.trim(),
        init_image_path: paths.imagePath,
        init_video_path: "",
        dense_video_path: paths.densePath,
        sparse_video_path: paths.sparsePath,
      };
    } else {
      const inputPath = await uploadSelectedInput();
      session = {
        prompt: el.promptInput.value || "",
        init_image_path: state.mode === "image" ? inputPath : "",
        init_video_path: state.mode === "video" ? inputPath : "",
      };
    }
    session.text_events = state.textEvents;
    session.output_resolution = state.outputResolution;
    state.fallbackStarted = false;
    if (state.preferWebSocket) {
      await connectSocket(session);
      return;
    }
    state.transport = "RTC";
    const pc = new RTCPeerConnection({ iceServers: state.iceServers });
    const channel = pc.createDataChannel("controls", { ordered: true });
    state.peer = pc;
    state.channel = channel;
    if (state.mouseAxes) {
      state.axisChannel = pc.createDataChannel("axes", { ordered: false, maxRetransmits: 0 });
    }
    pc.addTransceiver("video", { direction: "recvonly" });

    channel.addEventListener("open", () => {
      markConnected("RTC");
    });
    channel.addEventListener("message", (event) => handleServerMessage(event.data));
    channel.addEventListener("close", () => {
      if (state.channel === channel && !state.fallbackStarted) markDisconnected();
    });
    pc.addEventListener("track", (event) => {
      if (state.peer !== pc) return;
      const [stream] = event.streams;
      attachRemoteStream(stream || new MediaStream([event.track]));
    });
    pc.addEventListener("connectionstatechange", () => {
      if (state.peer !== pc) return;
      if (pc.connectionState === "failed") {
        if (state.connected) {
          markDisconnected();
        } else {
          fallbackToSocket(session, pc).catch((error) => {
            setStatus(`START ERROR: ${error.message || error}`, false, true);
          });
        }
      } else if (["closed", "disconnected"].includes(pc.connectionState)) {
        if (!state.fallbackStarted) markDisconnected();
      }
    });

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitForIce(pc);
    const answer = await api("/api/webrtc/offer", {
      method: "POST",
      body: JSON.stringify({ offer: pc.localDescription, session }),
    });
    await pc.setRemoteDescription(answer);
    if (!state.connected) {
      state.connectionTimer = window.setTimeout(async () => {
        if (state.connected || state.peer !== pc) return;
        try {
          await fallbackToSocket(session, pc);
        } catch (error) {
          setStatus(`START ERROR: ${error.message || error}`, false, true);
        }
      }, 5000);
    }
  } catch (error) {
    state.connecting = false;
    state.fallbackStarted = false;
    state.segmentBusy = false;
    el.startButton.disabled = false;
    el.startButton.textContent = state.queuedSegments ? "RUN" : "START";
    setStatus(`START ERROR: ${error.message || error}`, false, true);
    await closePeer(false);
    throw error;
  }
}

async function extendQueuedSegment() {
  if (!state.queuedSegments || !state.connected || state.segmentBusy) return;
  const prompt = el.promptInput.value.trim();
  if (!prompt) {
    setStatus("ENTER PROMPT", false, true);
    el.promptInput.focus();
    return;
  }
  if (!state.denseFile && !state.denseUploadedPath) {
    setStatus("SELECT NEW DENSE DEPTH CONTROL", false, true);
    return;
  }
  if (!state.sparseFile && !state.sparseUploadedPath) {
    setStatus("SELECT NEW SPARSE TRACK CONTROL", false, true);
    return;
  }
  state.segmentBusy = true;
  el.startButton.disabled = true;
  el.startButton.textContent = "UPLOADING";
  try {
    const paths = await uploadQueuedControls();
    if (!send({
      type: "segment_update",
      prompt,
      dense_video_path: paths.densePath,
      sparse_video_path: paths.sparsePath,
    })) {
      throw new Error("resident segment channel is not connected");
    }
    el.startButton.textContent = "QUEUED";
    setStatus(`SEGMENT ${state.segmentIndex + 1} QUEUED`, true);
  } catch (error) {
    state.segmentBusy = false;
    el.startButton.disabled = false;
    el.startButton.textContent = state.segmentIndex ? "EXTEND" : "RUN";
    setStatus(`EXTEND ERROR: ${error.message || error}`, false, true);
    throw error;
  }
}

async function closePeer(notify = true, { keepConnecting = false } = {}) {
  releaseAllKeys();
  window.clearInterval(state.heartbeatTimer);
  state.heartbeatTimer = null;
  window.clearTimeout(state.connectionTimer);
  state.connectionTimer = null;
  if (notify) send({ type: "disconnect" });
  const axisChannel = state.axisChannel;
  const channel = state.channel;
  const socket = state.socket;
  const peer = state.peer;
  state.peer = null;
  state.socket = null;
  state.channel = null;
  state.axisChannel = null;
  if (axisChannel && axisChannel.readyState !== "closed") axisChannel.close();
  if (channel && channel !== socket && channel.readyState !== "closed") channel.close();
  if (socket && socket.readyState !== (WebSocket.CLOSED ?? 3) && socket.readyState !== "closed") socket.close();
  if (peer && peer.connectionState !== "closed") peer.close();
  state.connected = false;
  if (!keepConnecting) {
    state.connecting = false;
    state.fallbackStarted = false;
  }
  stopAxisInputs();
  state.rolloutChunks = 0;
  state.paused = false;
  renderRollout();
  resetPresentationMetrics();
  resetSocketPlayback();
  clearAllPending();
  updateControlAvailability();
}

function setFileState(node, text, ready) {
  if (!node) return;
  node.textContent = text;
  node.classList.toggle("is-ready", Boolean(ready));
}

function clearQueuedControlSelection(force = false) {
  const denseWasConsumed = force || state.denseFile === state.inflightDenseFile
    && state.denseUploadedPath === state.inflightDensePath;
  const sparseWasConsumed = force || state.sparseFile === state.inflightSparseFile
    && state.sparseUploadedPath === state.inflightSparsePath;
  if (denseWasConsumed) {
    state.denseFile = null;
    state.denseUploadedPath = "";
    if (el.denseVideoInput) el.denseVideoInput.value = "";
    if (el.denseInputLabel) el.denseInputLabel.textContent = "DENSE DEPTH";
    setFileState(el.denseFileState, "NEW DENSE CONTROL REQUIRED", false);
  }
  if (sparseWasConsumed) {
    state.sparseFile = null;
    state.sparseUploadedPath = "";
    if (el.sparseVideoInput) el.sparseVideoInput.value = "";
    if (el.sparseInputLabel) el.sparseInputLabel.textContent = "SPARSE TRACK";
    setFileState(el.sparseFileState, "NEW SPARSE CONTROL REQUIRED", false);
  }
  state.inflightDenseFile = null;
  state.inflightDensePath = "";
  state.inflightSparseFile = null;
  state.inflightSparsePath = "";
}

async function resetSession() {
  setStatus("RESETTING");
  await closePeer(true);
  await api("/api/session/reset", { method: "POST", body: "{}" }).catch(() => {});
  state.sessionStartedAt = 0;
  state.segmentBusy = false;
  state.segmentIndex = 0;
  state.activeEventId = null;
  state.uploadedPath = "";
  state.denseUploadedPath = "";
  state.sparseUploadedPath = "";
  state.inflightDenseFile = null;
  state.inflightDensePath = "";
  state.inflightSparseFile = null;
  state.inflightSparsePath = "";
  state.stats = { fps: 0, generationMs: 0, enqueueMs: 0, latencyMs: 0, queueDepth: 0, dropped: 0 };
  state.lastVideoFrameAt = 0;
  if (state.streamFrameUrl) URL.revokeObjectURL(state.streamFrameUrl);
  state.streamFrameUrl = "";
  el.startButton.classList.remove("is-hidden");
  el.startButton.disabled = false;
  el.startButton.textContent = state.queuedSegments ? "RUN" : "START";
  el.resetButton.classList.add("is-hidden");
  if (state.queuedSegments) clearQueuedControlSelection(true);
  if (state.mode === "video" && state.previewUrl) showLocalVideo(state.previewUrl);
  else showImage(state.previewUrl || emptyFrame);
  setStatus("READY");
  setControlAck("LOCAL");
  setCatalogStatus(
    state.textEvents.length ? `${state.textEvents.length} EVENT${state.textEvents.length === 1 ? "" : "S"} READY` : "EMPTY CATALOG",
    "ok",
  );
  renderEventTriggers();
  updateControlAvailability();
  renderStats();
}

function renderExamples(examples) {
  el.thumbTray.innerHTML = "";
  for (const item of examples) {
    const itemUrl = authenticatedUrl(item.url);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "thumb-card";
    button.innerHTML = `<img src="${itemUrl}" alt=""><span>${item.label}</span>`;
    button.addEventListener("click", () => {
      if (state.connected || state.connecting) return;
      document.querySelectorAll(".thumb-card").forEach((node) => node.classList.remove("is-active"));
      button.classList.add("is-active");
      state.seedPath = item.path;
      state.selectedFile = null;
      state.uploadedPath = "";
      state.previewUrl = itemUrl;
      setMode("image");
      showImage(itemUrl);
      setStatus("INPUT READY");
    });
    el.thumbTray.appendChild(button);
  }
}

function bindInputPickers() {
  el.imageInput.addEventListener("change", (event) => {
    const [file] = event.target.files || [];
    if (!file) return;
    event.target.value = "";
    if (state.previewUrl.startsWith("blob:")) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = URL.createObjectURL(file);
    state.selectedFile = file;
    state.uploadedPath = "";
    state.seedPath = "";
    setMode("image");
    showImage(state.previewUrl);
    if (el.imageInputLabel) el.imageInputLabel.textContent = file.name;
    setFileState(el.imageFileState, `IMAGE · ${file.name}`, true);
    setStatus("INPUT READY");
  });
  el.videoInput.addEventListener("change", (event) => {
    const [file] = event.target.files || [];
    if (!file) return;
    event.target.value = "";
    if (state.previewUrl.startsWith("blob:")) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = URL.createObjectURL(file);
    state.selectedFile = file;
    state.uploadedPath = "";
    state.seedPath = "";
    setMode("video");
    showLocalVideo(state.previewUrl);
    setStatus("VIDEO READY");
  });
  el.denseVideoInput.addEventListener("change", (event) => {
    const [file] = event.target.files || [];
    if (!file) return;
    event.target.value = "";
    state.denseFile = file;
    state.denseUploadedPath = "";
    el.denseInputLabel.textContent = file.name;
    setFileState(el.denseFileState, `DENSE · ${file.name}`, true);
    setStatus("DENSE CONTROL READY");
  });
  el.sparseVideoInput.addEventListener("change", (event) => {
    const [file] = event.target.files || [];
    if (!file) return;
    event.target.value = "";
    state.sparseFile = file;
    state.sparseUploadedPath = "";
    el.sparseInputLabel.textContent = file.name;
    setFileState(el.sparseFileState, `SPARSE · ${file.name}`, true);
    setStatus("SPARSE CONTROL READY");
  });
}

function bindEvents() {
  document.querySelectorAll(".mode-tab").forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });
  el.loadModelButton.addEventListener("click", () => loadModel().catch(console.error));
  el.startButton.addEventListener("click", () => {
    const operation = state.queuedSegments && state.connected
      ? extendQueuedSegment()
      : connectSession();
    operation.catch(console.error);
  });
  el.resetButton.addEventListener("click", () => resetSession().catch(console.error));
  el.fullscreenButton?.addEventListener("click", () => {
    toggleImmersive().catch((error) => appendRuntimeLog(`Fullscreen error: ${error.message || error}`, "error"));
  });
  el.logToggleButton?.addEventListener("click", () => setLogOpen(!state.logOpen));
  el.clearLogButton?.addEventListener("click", clearRuntimeLog);
  el.addEventButton?.addEventListener("click", () => addEventRow());
  el.applyEventsButton?.addEventListener("click", applyEventCatalog);
  el.stepButton?.addEventListener("click", () => {
    const requestId = nextRequestId("step");
    setPending("step", requestId);
    if (send({ type: "action", action: { event: "step" }, request_id: requestId })) {
      setControlAck("STEP QUEUED");
    } else {
      clearPending("step");
      setControlAck("NOT CONNECTED", "error");
    }
  });
  el.pauseButton?.addEventListener("click", togglePause);
  el.resolutionSelect?.addEventListener("change", () => {
    state.outputResolution = resolutionPayload(el.resolutionSelect.value);
    renderStats();
    if (!state.connected) {
      setControlAck("OUTPUT LOCAL", "ok");
      return;
    }
    const requestId = nextRequestId("output");
    setPending("output", requestId);
    if (send({
      type: "output_config",
      resolution: state.outputResolution,
      request_id: requestId,
    })) {
      setControlAck("OUTPUT QUEUED");
    } else {
      clearPending("output");
      setControlAck("NOT CONNECTED", "error");
    }
  });
  window.addEventListener("keydown", (event) => {
    if (event.repeat || shouldIgnoreKey(event)) return;
    const key = String(event.key || "").toLowerCase();
    if (key === "f" && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault();
      toggleImmersive().catch((error) => appendRuntimeLog(`Fullscreen error: ${error.message || error}`, "error"));
    } else if (key === "l" && event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault();
      setLogOpen(!state.logOpen);
    } else if (key === "escape" && state.immersive && !document.fullscreenElement) {
      event.preventDefault();
      toggleImmersive(false).catch(() => {});
    }
  });
  document.addEventListener("fullscreenchange", () => {
    if (document.fullscreenElement) {
      state.nativeFullscreen = true;
      if (!state.immersive) applyImmersive(true);
      return;
    }
    if (state.nativeFullscreen) {
      state.nativeFullscreen = false;
      if (state.immersive) {
        applyImmersive(false);
        appendRuntimeLog("Browser fullscreen exited");
      }
    }
  });
  el.promptInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || !(state.promptScheduled || state.continuous) || !state.connected) return;
    event.preventDefault();
    // A continuous rollout accepts a new prompt at any chunk boundary.
    if (state.segmentBusy && !state.continuous) {
      setStatus("WAIT FOR THE CURRENT SEGMENT", true);
      return;
    }
    const prompt = el.promptInput.value.trim();
    if (!prompt) {
      setStatus("ENTER PROMPT", false, true);
      return;
    }
    if (send({ type: "prompt_update", prompt })) {
      el.promptInput.blur();
      el.viewport.scrollIntoView({ block: "center", inline: "nearest", behavior: "smooth" });
      setStatus(state.continuous ? "PROMPT QUEUED · STARTS A NEW ROLLOUT" : "PROMPT QUEUED", true);
    }
  });
  bindInputPickers();
  bindJoystick(el.moveStick, "move");
  bindJoystick(el.lookStick, "look");
  bindKeyboard();
  bindAxisInputs();
  window.addEventListener("beforeunload", () => {
    releaseAllKeys();
    send({ type: "disconnect" });
  });
}

function updateTimer() {
  if (!state.sessionStartedAt) {
    el.timer.textContent = "00:00";
    return;
  }
  const elapsed = Math.floor((Date.now() - state.sessionStartedAt) / 1000);
  el.timer.textContent = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;
}

async function syncRuntimeStatus() {
  const status = await api("/api/runtime/status");
  if (status.capabilities && typeof status.capabilities === "object") {
    const previousTextEvents = Boolean(state.capabilities.text_events);
    state.capabilities = { ...state.capabilities, ...status.capabilities };
    const currentTextEvents = Boolean(state.capabilities.text_events);
    if (currentTextEvents !== previousTextEvents) {
      if (currentTextEvents) {
        setCatalogStatus(
          state.textEvents.length
            ? `${state.textEvents.length} EVENT${state.textEvents.length === 1 ? "" : "S"} READY`
            : "EMPTY CATALOG",
          "ok",
        );
      } else {
        setCatalogStatus("MODEL HAS NO STATE-PRESERVING TEXT UPDATE", "error");
      }
      renderEventTriggers();
      updateControlAvailability();
    }
  }
  if (status.error) {
    state.runtimePhase = "error";
    state.modelLoaded = false;
    el.loadModelButton.textContent = "RETRY LOAD";
    setStatus(`LOAD ERROR: ${status.error}`, false, true);
    return true;
  }
  if (status.ready) {
    if (state.runtimePhase !== "ready") appendRuntimeLog("Model runtime ready", "live");
    state.runtimePhase = "ready";
    state.modelLoaded = true;
    applyRealtimeSpec(status.realtime_spec);
    el.loadModelButton.textContent = "READY";
    if (!state.connected && !state.connecting && (state.seedPath || state.selectedFile)) setStatus("READY");
    return true;
  }
  if (state.runtimePhase !== "loading") appendRuntimeLog("Model runtime loading");
  state.runtimePhase = "loading";
  el.loadModelButton.textContent = "LOADING";
  return false;
}

async function boot() {
  appendRuntimeLog("Studio client booting", "live");
  el.root.classList.add("image-mode");
  showImage(emptyFrame);
  bindEvents();
  setupPanels();
  renderStats();
  const data = await api("/api/session");
  state.examples = data.examples || [];
  state.promptScheduled = data.interaction_mode === "prompt-scheduled";
  state.queuedSegments = data.interaction_mode === "queued-segments";
  state.preferWebSocket = Boolean(data.prefer_websocket);
  state.capabilities = data.capabilities || {};
  state.textEvents = Array.isArray(data.event_catalog) ? data.event_catalog : [];
  state.activeEventId = data.active_event_id || null;
  state.catalogRevision = Number(data.catalog_revision || 0);
  state.outputResolution = data.output_resolution || { mode: "native" };
  state.outputResolutionOptions = Array.isArray(data.output_resolutions) ? data.output_resolutions : [];
  if (data.transport === "websocket") state.transport = "WS";
  if (state.promptScheduled || state.queuedSegments) {
    el.promptInput.title = state.queuedSegments
      ? "Prompt for the next full-quality queued segment"
      : "Edit the prompt and press Enter to generate the next native segment";
  }
  applyRealtimeSpec(data.realtime_spec);
  if (state.queuedSegments) {
    el.startButton.textContent = "RUN";
    setStatus("ADD IMAGE + DENSE + SPARSE CONTROLS");
  } else if (state.promptScheduled) {
    setStatus("ENTER PROMPT · IMAGE OPTIONAL");
  }
  state.iceServers = data.ice_servers || [];
  state.playbackFps = Math.max(Number(data.fps || state.playbackFps), 1);
  replaceEventRows(state.textEvents);
  renderEventTriggers();
  renderResolutionOptions(state.outputResolutionOptions);
  setControlAck(data.output_resolution_scope === "transport" ? "TRANSPORT ONLY" : "LOCAL");
  updateControlAvailability();
  if (!state.capabilities.text_events) {
    setCatalogStatus("MODEL HAS NO STATE-PRESERVING TEXT UPDATE", "error");
  }
  renderStats();
  renderExamples(state.examples);
  if (state.examples.length) el.thumbTray.querySelector(".thumb-card")?.click();
  const statusTimer = window.setInterval(async () => {
    try {
      if (await syncRuntimeStatus()) window.clearInterval(statusTimer);
    } catch {
      // The explicit LOAD action will surface a useful error.
    }
  }, 1000);
  await syncRuntimeStatus().catch(() => false);
  window.setInterval(updateTimer, 500);
}

boot().catch((error) => {
  setStatus(`BOOT ERROR: ${error.message || error}`, false, true);
  console.error(error);
});
