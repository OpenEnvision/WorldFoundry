// SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const file = new URL('../../worldfoundry/studio/assets/web/world.js', import.meta.url);
const script = readFileSync(file, 'utf8');
const metricsCode = script.slice(script.indexOf('class PresentationMetrics'), script.indexOf('const state ='));

function metrics() {
  const samples = [];
  const Class = vm.runInNewContext(`${metricsCode}; PresentationMetrics`);
  return { collector: new Class((sample) => samples.push(sample)), samples };
}

test('complete browser script parses', () => {
  new vm.Script(script);
});

for (const receiptFirst of [true, false]) {
  test(`matches actual RTP timestamp with receiptFirst=${receiptFirst}`, () => {
    const { collector, samples } = metrics();
    collector.submitted('key', 100);
    const receipt = {rtp_timestamp: 4294967295, chunk_index: 3, control_ids: ['key']};
    if (receiptFirst) collector.frameSent(receipt);
    collector.presented(4294967295, 240);
    if (!receiptFirst) collector.frameSent(receipt);
    assert.equal(samples.length, 1);
    assert.equal(samples[0].control_to_display_ms, 140);
    assert.equal(samples[0].rtp_timestamp, 4294967295);
    collector.frameSent({...receipt, rtp_timestamp: 0});
    collector.presented(0, 300);
    assert.equal(samples.length, 1); // Held/repeated frames cannot inflate the sample count.
  });
}

test('does not count queued, dropped, unrelated or unsupported frames as presented', () => {
  const { collector, samples } = metrics();
  collector.submitted('key', 100);
  collector.frameSent({rtp_timestamp: 1, control_ids: ['key']});
  collector.presented(2, 200);
  collector.presented(undefined, 250);
  assert.equal(samples.length, 0);
  collector.reset();
  collector.presented(1, 300);
  assert.equal(samples.length, 0);
});

test('maps are bounded during long sessions without frame metadata support', () => {
  const { collector } = metrics();
  for (let i = 0; i < 1000; i++) {
    collector.submitted(`key-${i}`, i);
    collector.frameSent({rtp_timestamp: i, control_ids: []});
    collector.presented(i + 1000, i);
  }
  assert.equal(collector.controls.size, 256);
  assert.equal(collector.sent.size, 256);
  assert.equal(collector.displayed.size, 256);
});

test('getStats uses interval deltas and keeps browser units explicit', async () => {
  const body = script.slice(script.indexOf('async function pollTransportStats'), script.indexOf('function markConnected'));
  const emitted = [];
  let snapshot;
  const peer = {getStats: async () => snapshot};
  const state = {peer, connected: true, transportStatsPending: false, previousTransportStats: new Map()};
  const poll = vm.runInNewContext(`${body}; pollTransportStats`, {state, send: (x) => emitted.push(x), renderStats() {}, console});
  const sample = (timestamp, bytes, count, buffer, decode) => new Map([
    ['v', {id: 'v', type: 'inbound-rtp', kind: 'video', timestamp, bytesReceived: bytes,
      framesDecoded: count, jitterBufferEmittedCount: count, jitterBufferDelay: buffer,
      totalDecodeTime: decode, packetsLost: 1, framesDropped: 2, jitter: 0.003}],
    ['t', {type: 'transport', selectedCandidatePairId: 'p'}],
    ['p', {currentRoundTripTime: 0.02}],
  ]);
  snapshot = sample(1000, 100, 10, 0.1, 0.02);
  await poll();
  assert.equal(emitted[0].jitter_buffer_ms, undefined);
  snapshot = sample(2000, 1100, 20, 0.3, 0.05);
  await poll();
  assert.equal(emitted[1].bitrate_bps, 8000);
  assert.ok(Math.abs(emitted[1].jitter_buffer_ms - 20) < 1e-9);
  assert.ok(Math.abs(emitted[1].decode_ms - 3) < 1e-9);
  assert.equal(emitted[1].rtt_ms, 20);
  assert.equal(emitted[1].jitter_ms, 3);
});

class FakeElement {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.attributes = {};
    this.listeners = {};
    this.style = {setProperty() {}};
    this.dataset = {};
    this.textContent = '';
    this.title = '';
    this.value = '';
    this.disabled = false;
    const classes = new Set();
    this.classList = {
      add: (...names) => names.forEach((name) => classes.add(name)),
      remove: (...names) => names.forEach((name) => classes.delete(name)),
      contains: (name) => classes.has(name),
      toggle: (name, force) => {
        const on = force === undefined ? !classes.has(name) : Boolean(force);
        if (on) classes.add(name); else classes.delete(name);
        return on;
      },
    };
  }
  get options() { return this.children; }
  getBoundingClientRect() { return {left: 0, top: 0, width: 100, height: 100}; }
  setPointerCapture(id) { this.capturedPointer = id; }
  hasPointerCapture(id) { return this.capturedPointer === id; }
  releasePointerCapture(id) {
    if (!this.hasPointerCapture(id)) return;
    this.capturedPointer = null;
    this.dispatch('lostpointercapture', {pointerId: id});
  }
  addEventListener(name, listener) { (this.listeners[name] ||= []).push(listener); }
  dispatch(name, event = {}) { for (const listener of this.listeners[name] || []) listener({target: this, ...event}); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name]; }
  removeAttribute(name) { delete this.attributes[name]; }
  appendChild(child) { this.children.push(child); return child; }
  append(...children) { this.children.push(...children); }
  prepend(...children) { this.children.unshift(...children); }
  replaceChildren(...children) { this.children = children; }
  add(option) { this.children.push(option); }
  getContext() { return {drawImage() {}}; }
  querySelector() { return null; }
  querySelectorAll(selector) {
    if (selector === 'button') return this.children.filter((child) => child.tagName === 'BUTTON');
    return [];
  }
  closest() { return null; }
  focus() {}
  blur() {}
  pause() {}
  load() {}
  remove() {}
  scrollIntoView() {}
  play() { return Promise.resolve(); }
}

class FakeWebSocket {
  static OPEN = 1;
  static CLOSED = 3;
  static onCreate = null;
  constructor(url = '') {
    this.url = url;
    this.readyState = FakeWebSocket.OPEN;
    this.bufferedAmount = 0;
    this.sent = [];
    this.listeners = {};
    this.closed = false;
    FakeWebSocket.onCreate?.(this);
  }
  addEventListener(name, listener) { (this.listeners[name] ||= []).push(listener); }
  dispatch(name, event = {}) { for (const listener of this.listeners[name] || []) listener(event); }
  send(raw) { this.sent.push(JSON.parse(raw)); }
  close() { this.closed = true; this.readyState = FakeWebSocket.CLOSED; }
}

function fakeDataChannel(label) {
  const listeners = {};
  return {
    label, readyState: 'open', bufferedAmount: 0, sent: [], closed: false, listeners,
    addEventListener(name, listener) { (listeners[name] ||= []).push(listener); },
    dispatch(name, event = {}) { for (const listener of listeners[name] || []) listener(event); },
    close() { this.closed = true; this.readyState = 'closed'; },
    send(raw) { this.sent.push(JSON.parse(raw)); },
  };
}

function listenerTarget() {
  const listeners = {};
  return {
    listeners,
    addEventListener(name, listener) { (listeners[name] ||= []).push(listener); },
    dispatch(name, event = {}) { for (const listener of listeners[name] || []) listener(event); },
  };
}

// Run the complete client against a fake DOM and a fake Studio session.
async function loadClient({
  spec = {},
  interactionMode = 'controls',
  capabilities = {no_action_step: true},
  status,
  gamepadApi = true,
  outputResolutions = [],
  fetchOverride = null,
} = {}) {
  const elements = new Map();
  const frames = [];
  const pads = [];
  const peers = [];
  const sockets = [];
  const timers = [];
  const windowTarget = listenerTarget();
  const documentTarget = listenerTarget();
  const document = {
    ...documentTarget,
    body: new FakeElement('body'),
    documentElement: new FakeElement('html'),
    visibilityState: 'visible',
    focused: true,
    pointerLockElement: null,
    hasFocus() { return this.focused; },
    exitPointerLock() { this.pointerLockElement = null; },
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new FakeElement());
      return elements.get(id);
    },
    querySelectorAll: () => [],
    createElement: (tag) => new FakeElement(tag),
    createTextNode: (text) => ({textContent: String(text)}),
    createDocumentFragment: () => new FakeElement('fragment'),
  };
  const session = {
    interaction_mode: interactionMode,
    capabilities,
    realtime_spec: spec,
    output_resolutions: outputResolutions,
  };
  const responses = {
    '/api/session': session,
    '/api/runtime/status': status || {ready: true, realtime_spec: spec},
    '/api/runtime/load': {loaded: true, ready: true},
    '/api/webrtc/offer': {type: 'answer', sdp: ''},
    '/api/session/reset': {reset: true},
  };
  class FakePeerConnection {
    constructor() {
      this.channels = [];
      this.listeners = {};
      this.iceGatheringState = 'complete';
      this.connectionState = 'new';
      this.closed = false;
      peers.push(this);
    }
    createDataChannel(label, options) { const channel = fakeDataChannel(label); this.channels.push({label, options, channel}); return channel; }
    addTransceiver() {}
    addEventListener(name, listener) { (this.listeners[name] ||= []).push(listener); }
    dispatch(name, event = {}) { for (const listener of this.listeners[name] || []) listener(event); }
    async createOffer() { return {type: 'offer', sdp: ''}; }
    async setLocalDescription(description) { this.localDescription = description; }
    async setRemoteDescription() {}
    close() { this.closed = true; this.connectionState = 'closed'; }
  }
  FakeWebSocket.onCreate = (ws) => sockets.push(ws);
  const FakeURL = class extends URL {
    static createObjectURL() { return 'blob:fake'; }
    static revokeObjectURL() {}
  };
  const FakeOption = function Option(text, value) {
    const option = new FakeElement('option');
    option.textContent = String(text);
    option.value = String(value);
    return option;
  };
  const window = {
    ...windowTarget,
    location: {search: '', href: 'http://localhost/', origin: 'http://localhost', protocol: 'http:'},
    setInterval: () => 0,
    clearInterval() {},
    setTimeout: (fn, delay) => { timers.push({fn, delay}); return timers.length; },
    clearTimeout() {},
    matchMedia: () => ({matches: false}),
    queueMicrotask,
    innerWidth: 1280,
    innerHeight: 800,
  };
  const context = vm.createContext({
    window, document, console, URL: FakeURL, URLSearchParams, Headers, FormData, Blob,
    Option: FakeOption,
    Element: FakeElement,
    WebSocket: FakeWebSocket,
    RTCPeerConnection: FakePeerConnection,
    navigator: gamepadApi ? {getGamepads: () => pads} : {},
    performance: {now: () => 0},
    localStorage: {getItem: () => null, setItem() {}},
    requestAnimationFrame: (callback) => frames.push(callback),
    cancelAnimationFrame() {},
    createImageBitmap: async () => ({width: 16, height: 16, closed: false, close() { this.closed = true; }}),
    fetch: async (url, options) => {
      if (fetchOverride) {
        const custom = await fetchOverride(url, options);
        if (custom !== undefined) return custom;
      }
      return {ok: true, json: async () => structuredClone(responses[url]), text: async () => JSON.stringify(responses[url])};
    },
  });
  vm.runInContext(script, context);
  for (let i = 0; i < 20; i++) await new Promise((resolve) => setImmediate(resolve));
  const client = vm.runInContext('({state, el, allowedKeys})', context);
  assert.doesNotMatch(client.el.livePill.textContent, /BOOT ERROR/);
  return {
    ...client, context, document, window, pads, peers, sockets, timers, frames, responses,
    tick() { for (const callback of frames.splice(0)) callback(performance.now()); },
    stick(kind, type, dx = 0, dy = 0, pointerId = 1) {
      client.el[`${kind}Stick`].dispatch(type, {
        pointerId, clientX: 50 + dx * 50, clientY: 50 + dy * 50, preventDefault() {},
      });
    },
    connect(channel = fakeDataChannel('controls'), axisChannel = null) {
      client.state.channel = channel;
      client.state.axisChannel = axisChannel;
      context.markConnected(channel instanceof FakeWebSocket ? 'WS' : 'RTC');
      return channel;
    },
    receive(message) { context.handleServerMessage(JSON.stringify(message)); },
    key(name, type = 'keydown', target = null) {
      window.dispatch(type, {key: name, repeat: false, defaultPrevented: false, target,
        preventDefault() { this.defaultPrevented = true; }, shiftKey: false, ctrlKey: false, metaKey: false, altKey: false});
    },
  };
}

const MOUSE_AXES = [{name: 'camera_yaw', source: 'mouse_x'}, {name: 'camera_pitch', source: 'mouse_y'}];
const GAMEPAD_AXES = [{name: 'camera_yaw_stick', source: 'gamepad_rx', mode: 'level'}];

function axisPayloads(channel) {
  return channel.sent.filter(message => message.type === 'axis').map(({control_id, ...payload}) => payload);
}

test('virtual sticks use declared axes per direction and retain undeclared-direction keys', async () => {
  const client = await loadClient({spec: {axes: GAMEPAD_AXES}, gamepadApi: false});
  const controls = fakeDataChannel('controls');
  const axes = fakeDataChannel('axes');
  client.connect(controls, axes);
  client.stick('look', 'pointerdown', 0.5, -0.5);
  client.tick();
  assert.deepEqual(axisPayloads(controls), [{type: 'axis', source: 'gamepad', lx: 0, ly: 0, rx: 0.5, ry: 0}]);
  assert.deepEqual(controls.sent.filter(m => m.type === 'action').map(m => m.action), [{event: 'keydown', key: 'i'}]);
  client.stick('look', 'pointerup');
  client.tick();
  assert.equal(axisPayloads(controls).at(-1).rx, 0);
  assert.deepEqual(controls.sent.filter(m => m.type === 'action').map(m => m.action),
    [{event: 'keydown', key: 'i'}, {event: 'keyup', key: 'i'}]);
  assert.equal(axes.sent.length, 0, 'virtual stick levels stay on the reliable channel');
});

test('axis-only sessions expose both virtual sticks without needing the Gamepad API', async () => {
  const axes = ['lx', 'ly', 'rx', 'ry'].map(name => ({name, source: `gamepad_${name}`, mode: 'level'}));
  const client = await loadClient({spec: {keys: [], axes}, gamepadApi: false});
  assert.equal(client.el.moveStick.classList.contains('is-hidden'), false);
  assert.equal(client.el.lookStick.classList.contains('is-hidden'), false);
  assert.equal(client.el.interactionHint.textContent, 'DRAG STICKS');
  const controls = client.connect();
  client.stick('move', 'pointerdown', 0.25, -0.5, 1);
  client.stick('look', 'pointerdown', 0.5, -0.25, 2);
  client.tick();
  assert.deepEqual(axisPayloads(controls).at(-1), {type: 'axis', source: 'gamepad', lx: 0.25, ly: -0.5, rx: 0.5, ry: -0.25});
  client.stick('move', 'pointercancel', 0, 0, 1);
  client.tick();
  assert.deepEqual(axisPayloads(controls).at(-1), {type: 'axis', source: 'gamepad', lx: 0, ly: 0, rx: 0.5, ry: -0.25});
  client.stick('look', 'lostpointercapture', 0, 0, 2);
  client.tick();
  assert.deepEqual(axisPayloads(controls).at(-1), {type: 'axis', source: 'gamepad', lx: 0, ly: 0, rx: 0, ry: 0});
  assert.equal(controls.sent.filter(m => m.type === 'action').length, 0);
});

test('captured virtual axes override the physical stick and restore it on release', async () => {
  const client = await loadClient({spec: {axes: GAMEPAD_AXES}});
  const pad = {connected: true, axes: [0, 0, -0.75, 0]};
  client.pads.push(pad);
  const controls = client.connect();
  client.tick();
  assert.equal(axisPayloads(controls).at(-1).rx, -0.75);
  client.stick('look', 'pointerdown', 0.5, 0);
  client.tick();
  assert.equal(axisPayloads(controls).at(-1).rx, 0.5);
  pad.axes[2] = -0.5;
  client.tick();
  assert.equal(axisPayloads(controls).length, 2, 'the physical stick cannot overwrite a captured virtual axis');
  client.stick('look', 'pointerup');
  client.tick();
  assert.equal(axisPayloads(controls).at(-1).rx, -0.5);
  client.stick('look', 'pointerdown', 0.75, 0);
  client.tick();
  pad.connected = false;
  client.window.dispatch('gamepaddisconnected');
  assert.equal(axisPayloads(controls).at(-1).rx, 0.75, 'disconnecting hardware does not release a captured virtual axis');
  client.stick('look', 'pointerup');
  client.tick();
  assert.equal(axisPayloads(controls).at(-1).rx, 0);
});

test('a physical gamepad disconnect cannot activate another controller in a background tab', async () => {
  const client = await loadClient({spec: {axes: GAMEPAD_AXES}});
  const first = {connected: true, axes: [0, 0, 0.5, 0]};
  client.pads.push(first, {connected: true, axes: [0, 0, -0.75, 0]});
  const controls = client.connect();
  client.tick();
  client.document.focused = false;
  client.window.dispatch('blur');
  first.connected = false;
  client.window.dispatch('gamepaddisconnected');
  assert.equal(axisPayloads(controls).at(-1).rx, 0);
  client.document.focused = true;
  client.tick();
  assert.equal(axisPayloads(controls).at(-1).rx, -0.75);
});

for (const source of ['virtual', 'physical']) {
  test(`${source} stick neutral samples are not suppressed by the movement epsilon`, async () => {
    const client = await loadClient({spec: {axes: GAMEPAD_AXES}});
    const pad = {connected: true, axes: [0, 0, 0, 0]};
    if (source === 'physical') client.pads.push(pad);
    const controls = client.connect();
    if (source === 'virtual') client.stick('look', 'pointerdown', 0.05, 0);
    else pad.axes[2] = 0.05;
    client.tick();
    if (source === 'virtual') client.stick('look', 'pointermove', 0.01, 0);
    else pad.axes[2] = 0.01;
    client.tick();
    if (source === 'virtual') client.stick('look', 'pointerup');
    else pad.axes[2] = 0;
    client.tick();
    assert.deepEqual(axisPayloads(controls).map(m => m.rx), [0.05, 0.01, 0]);
  });
}

for (const loss of ['blur', 'hidden']) {
  test(`${loss} releases virtual axes and does not revive the old pointer on focus return`, async () => {
    const client = await loadClient({spec: {axes: GAMEPAD_AXES}});
    const controls = client.connect();
    client.stick('look', 'pointerdown', 0.5, 0);
    client.stick('move', 'pointerdown', 0, -0.5, 2);
    client.tick();
    if (loss === 'blur') {
      client.document.focused = false;
      client.window.dispatch('blur');
    } else {
      client.document.visibilityState = 'hidden';
      client.document.dispatch('visibilitychange');
    }
    assert.equal(axisPayloads(controls).at(-1).rx, 0);
    assert.deepEqual(controls.sent.filter(m => m.type === 'action').map(m => m.action),
      [{event: 'keydown', key: 'w'}, {event: 'keyup', key: 'w'}]);
    const count = controls.sent.length;
    client.document.focused = true;
    client.document.visibilityState = 'visible';
    client.stick('look', 'pointermove', 0.75, 0);
    client.stick('move', 'pointermove', 0, -0.75, 2);
    client.tick();
    assert.equal(controls.sent.length, count);
    assert.equal(client.el.lookStick.classList.contains('is-active'), false);
    assert.equal(client.el.moveStick.classList.contains('is-active'), false);
  });
}

test('mouse and virtual axis IDs report latency only after a matching frame presentation', async () => {
  const client = await loadClient({spec: {axes: [...MOUSE_AXES, ...GAMEPAD_AXES]}});
  const controls = fakeDataChannel('controls');
  const axes = fakeDataChannel('axes');
  client.connect(controls, axes);
  client.state.pointerLocked = true;
  client.document.dispatch('mousemove', {movementX: 2, movementY: -1});
  client.stick('look', 'pointerdown', 0.5, 0);
  client.tick();
  const mouse = axes.sent.find(m => m.source === 'mouse');
  const stick = controls.sent.find(m => m.source === 'gamepad');
  assert.match(mouse.control_id, /^axis-mouse-/);
  assert.match(stick.control_id, /^axis-gamepad-/);
  assert.notEqual(mouse.control_id, stick.control_id);
  const ids = [mouse.control_id, stick.control_id];
  assert.equal(client.state.presentation.controls.size, 2);
  client.receive({type: 'frame_sent', rtp_timestamp: 41, control_ids: ids});
  const reports = () => controls.sent.filter(m => m.type === 'frame_presented');
  assert.equal(reports().length, 0, 'a server receipt alone is not a presentation');
  client.state.presentation.presented(40, 60);
  assert.equal(reports().length, 0, 'an unrelated displayed frame is not a match');
  client.state.presentation.presented(41, 80);
  assert.deepEqual(reports().map(m => [m.control_id, m.control_to_display_ms]), ids.map(id => [id, 80]));
  client.receive({type: 'frame_sent', rtp_timestamp: 42, control_ids: ids});
  client.state.presentation.presented(42, 100);
  assert.equal(reports().length, 2, 'held or repeated frames do not count the same inputs twice');
  axes.bufferedAmount = 1 << 20;
  client.document.dispatch('mousemove', {movementX: 3, movementY: 0});
  client.tick();
  assert.equal(client.state.presentation.controls.size, 0, 'a deferred delta is not marked as submitted');
});

test('sessions without declared keys keep the default eight keys and legend', async () => {
  const client = await loadClient();
  assert.deepEqual([...client.allowedKeys].sort(), ['a', 'd', 'i', 'j', 'k', 'l', 's', 'w']);
  assert.equal(client.el.interactionHint.textContent, 'HOLD WASD / DRAG STICKS');
  assert.equal(client.el.pauseButton.classList.contains('is-hidden'), true);
  const channel = client.connect();
  client.key('ArrowUp');
  client.key('ArrowUp', 'keyup');
  assert.deepEqual(channel.sent.filter((m) => m.type === 'action').map((m) => [m.action.event, m.action.key]),
    [['keydown', 'i'], ['keyup', 'i']]);
});

test('declared keys drive keyboard handling, sticks and the key legend', async () => {
  const keys = ['w', 's', 'a', 'd', 'z', 'c'].map((key) => ({key, action: `act_${key}`, group: 'action', label: ''}));
  const client = await loadClient({spec: {keys}});
  assert.deepEqual([...client.allowedKeys], ['w', 's', 'a', 'd', 'z', 'c']);
  assert.equal(client.el.moveStick.classList.contains('is-hidden'), false);
  assert.equal(client.el.lookStick.classList.contains('is-hidden'), true);
  assert.equal(client.el.interactionHint.textContent, 'HOLD W S A D Z C / DRAG STICKS');
  assert.match(client.el.interactionHint.title, /Z act_z/);
  const channel = client.connect();
  client.key('z');
  client.key('i');
  client.key('ArrowLeft');
  assert.deepEqual(channel.sent.filter((m) => m.type === 'action').map((m) => m.action.key), ['z']);

  client.context.applyRealtimeSpec({keys: []});
  assert.equal(client.allowedKeys.size, 0);
  assert.equal(client.el.moveStick.classList.contains('is-hidden'), true);
});

test('prompt-scheduled sessions ignore declared keys', async () => {
  const client = await loadClient({interactionMode: 'prompt-scheduled', spec: {keys: [{key: 'w', action: 'forward'}]}});
  assert.equal(client.allowedKeys.size, 0);
  assert.equal(client.el.moveStick.classList.contains('is-hidden'), true);
});

test('locked mouse motion is coalesced to one axis message per animation frame', async () => {
  const client = await loadClient({spec: {axes: MOUSE_AXES}});
  assert.match(client.el.interactionHint.textContent, /CLICK VIEW FOR MOUSE/);
  client.el.viewport.requestPointerLock = () => {
    client.document.pointerLockElement = client.el.viewport;
    client.document.dispatch('pointerlockchange');
  };
  const controls = fakeDataChannel('controls');
  const axes = fakeDataChannel('axes');
  client.document.dispatch('mousemove', {movementX: 9, movementY: 9});
  client.connect(controls, axes);
  assert.equal(client.el.pointerHint.classList.contains('is-hidden'), false);
  assert.equal(client.el.pointerHint.textContent, 'CLICK VIEW FOR MOUSE LOOK');
  client.document.dispatch('mousemove', {movementX: 9, movementY: 9});
  client.tick();
  assert.equal(axes.sent.length, 0, 'motion before the pointer lock is not sent');

  client.el.viewport.dispatch('click');
  assert.equal(client.state.pointerLocked, true);
  assert.equal(client.el.pointerHint.textContent, 'MOUSE LOOK · ESC TO RELEASE');
  for (const [movementX, movementY] of [[3, -1], [2, 4], [-1, 0]]) {
    client.document.dispatch('mousemove', {movementX, movementY});
  }
  client.tick();
  assert.deepEqual(axisPayloads(axes), [{type: 'axis', source: 'mouse', dx: 4, dy: 3}]);

  axes.bufferedAmount = 1 << 20;
  client.document.dispatch('mousemove', {movementX: 5, movementY: 0});
  client.tick();
  assert.equal(axes.sent.length, 1, 'a congested channel defers the delta');
  client.document.dispatch('mousemove', {movementX: 1, movementY: 1});
  axes.bufferedAmount = 0;
  client.tick();
  assert.deepEqual(axisPayloads(axes).at(-1), {type: 'axis', source: 'mouse', dx: 6, dy: 1});

  client.document.pointerLockElement = null;
  client.document.dispatch('pointerlockchange');
  assert.equal(client.el.pointerHint.textContent, 'CLICK VIEW FOR MOUSE LOOK');
  client.document.dispatch('mousemove', {movementX: 7, movementY: 7});
  client.tick();
  assert.equal(axes.sent.length, 2, 'releasing the lock stops sending');
  assert.equal(controls.sent.filter((m) => m.type === 'axis').length, 0);
});

test('axis messages use the axes DataChannel on WebRTC and the socket otherwise', async () => {
  const rtc = await loadClient({spec: {axes: MOUSE_AXES}});
  rtc.state.seedPath = '/seed.png';
  await rtc.context.connectSession();
  assert.deepEqual(JSON.parse(JSON.stringify(rtc.peers[0].channels.map(({label, options}) => [label, options]))),
    [['controls', {ordered: true}], ['axes', {ordered: false, maxRetransmits: 0}]]);

  const keysOnly = await loadClient();
  keysOnly.state.seedPath = '/seed.png';
  await keysOnly.context.connectSession();
  assert.deepEqual(keysOnly.peers[0].channels.map(({label}) => label), ['controls']);

  const socketClient = await loadClient({spec: {axes: MOUSE_AXES}});
  const socket = socketClient.connect(new FakeWebSocket());
  socketClient.state.pointerLocked = true;
  socketClient.document.dispatch('mousemove', {movementX: 2, movementY: -3});
  socketClient.tick();
  assert.deepEqual(socket.sent.filter((m) => m.type === 'axis'), [{type: 'axis', source: 'mouse', dx: 2, dy: -3}]);
});

test('gamepad levels use the reliable controls channel; mouse deltas use axes', async () => {
  const client = await loadClient({spec: {axes: [...MOUSE_AXES, ...GAMEPAD_AXES]}});
  const pad = {connected: true, axes: [0, 0, 0, 0]};
  client.pads.push(pad);
  const controls = fakeDataChannel('controls');
  const axes = fakeDataChannel('axes');
  client.connect(controls, axes);
  const neutral = {type: 'axis', source: 'gamepad', lx: 0, ly: 0, rx: 0, ry: 0};
  const gamepad = () => axisPayloads(controls).filter((m) => m.source === 'gamepad');

  client.tick();
  pad.axes = [0.01, 0, -0.015, 0];
  client.tick();
  assert.equal(gamepad().length, 0, 'rest and sub-epsilon drift send nothing');
  pad.axes = [0.5, 0, 0, -0.25];
  client.tick();
  client.tick();
  pad.axes = [0.51, 0, 0, -0.25];
  client.tick();
  pad.axes = [0, 0, 0, 0];
  client.tick();
  assert.deepEqual(gamepad(), [{type: 'axis', source: 'gamepad', lx: 0.5, ly: 0, rx: 0, ry: -0.25}, neutral],
    'the move and the return to rest both go over the reliable channel');

  controls.bufferedAmount = 1 << 20;
  pad.axes = [0, 0, 0.8, 0];
  client.tick();
  assert.equal(gamepad().length, 2, 'a congested channel defers the level');
  controls.bufferedAmount = 0;
  client.tick();
  assert.deepEqual(gamepad().at(-1), {type: 'axis', source: 'gamepad', lx: 0, ly: 0, rx: 0.8, ry: 0});

  client.window.dispatch('blur');
  assert.deepEqual(gamepad().at(-1), neutral);
  client.document.focused = false;
  client.tick();
  assert.equal(gamepad().length, 4, 'an unfocused page does not resend the held stick');
  client.document.focused = true;
  client.tick();
  assert.deepEqual(gamepad().at(-1), {type: 'axis', source: 'gamepad', lx: 0, ly: 0, rx: 0.8, ry: 0});

  pad.connected = false;
  client.window.dispatch('gamepaddisconnected');
  client.document.visibilityState = 'hidden';
  client.document.dispatch('visibilitychange');
  assert.deepEqual(gamepad().slice(-2), [neutral, neutral]);

  client.state.pointerLocked = true;
  client.document.dispatch('mousemove', {movementX: 4, movementY: 1});
  client.tick();
  assert.deepEqual(axisPayloads(axes), [{type: 'axis', source: 'mouse', dx: 4, dy: 1}]);
  assert.equal(controls.sent.filter((m) => m.source === 'mouse').length, 0);
});

test('the GAMEPAD legend appears only when the browser reports a connected gamepad', async () => {
  const client = await loadClient({spec: {axes: GAMEPAD_AXES}});
  assert.doesNotMatch(client.el.interactionHint.textContent, /GAMEPAD/);
  client.pads.push({connected: true, axes: [0, 0, 0, 0]});
  client.window.dispatch('gamepadconnected');
  assert.match(client.el.interactionHint.textContent, /GAMEPAD/);
  client.pads.pop();
  client.window.dispatch('gamepaddisconnected');
  assert.doesNotMatch(client.el.interactionHint.textContent, /GAMEPAD/);

  const insecure = await loadClient({spec: {axes: GAMEPAD_AXES}, gamepadApi: false});
  insecure.window.dispatch('gamepadconnected');
  assert.doesNotMatch(insecure.el.interactionHint.textContent, /GAMEPAD/);
});

test('continuous sessions show progress, pause and resume, and the step rejection', async () => {
  const spec = {controls: ['prompt_update'], keys: [], advance: 'continuous', max_chunks: 7};
  const client = await loadClient({spec, interactionMode: 'prompt-scheduled'});
  const {el} = client;
  assert.equal(el.pauseButton.classList.contains('is-hidden'), false);
  assert.equal(el.pauseButton.disabled, true);
  assert.match(el.interactionHint.textContent, /NEW PROMPT TO START A NEW ROLLOUT/);

  const socket = new FakeWebSocket();
  client.state.channel = socket;
  client.state.transport = 'WS';
  const rollout = (chunks, extra = {}) => ({advance: 'continuous', rollout_chunks: chunks, max_chunks: 7, paused: false, ...extra});
  client.receive({type: 'ready', fps: 16, ...rollout(0)});
  assert.equal(client.state.connected, true);
  assert.equal(el.pauseButton.disabled, false);
  assert.equal(el.stepButton.disabled, true, 'STEP waits for a pause while the rollout advances');
  assert.match(el.streamStats.textContent, /CHUNK 0\/7/);

  client.receive({type: 'chunk_done', chunk_index: 3, frames: 0, ...rollout(3)});
  assert.match(el.streamStats.textContent, /CHUNK 3\/7/);
  assert.equal(el.livePill.textContent, 'GENERATING · CHUNK 3/7');

  el.pauseButton.dispatch('click');
  const pause = socket.sent.at(-1);
  assert.equal(pause.type, 'pause');
  assert.equal(el.pauseButton.disabled, true);
  client.receive({type: 'pause_ack', ok: true, request_id: pause.request_id, status: 'paused', ...rollout(3, {paused: true})});
  assert.equal(el.pauseButton.textContent, 'RESUME');
  assert.equal(el.pauseButton.getAttribute('aria-pressed'), 'true');
  assert.equal(el.stepButton.disabled, false);
  assert.equal(el.livePill.textContent, 'PAUSED · CHUNK 3/7');

  el.pauseButton.dispatch('click');
  const resume = socket.sent.at(-1);
  assert.equal(resume.type, 'resume');
  client.receive({type: 'resume_ack', ok: true, request_id: resume.request_id, status: 'resumed', ...rollout(3)});
  assert.equal(el.pauseButton.textContent, 'PAUSE');

  client.receive({type: 'chunk_done', chunk_index: 7, frames: 0, ...rollout(7)});
  assert.equal(el.livePill.textContent, 'ROLLOUT COMPLETE · 7/7 · ENTER A NEW PROMPT');
  assert.equal(el.pauseButton.disabled, true, 'a finished rollout offers no PAUSE');
  assert.equal(el.stepButton.disabled, false);
  el.stepButton.dispatch('click');
  const step = socket.sent.at(-1);
  const message = 'The rollout is complete; send a new prompt to start another.';
  client.receive({type: 'step_ack', ok: false, request_id: step.request_id, message});
  assert.equal(el.controlAck.textContent, message);
  assert.equal(el.controlAck.classList.contains('is-error'), true);

  client.state.segmentBusy = true;
  el.promptInput.value = 'a lighthouse at dusk';
  el.promptInput.dispatch('keydown', {key: 'Enter', preventDefault() {}});
  assert.deepEqual(socket.sent.at(-1), {type: 'prompt_update', prompt: 'a lighthouse at dusk'});
  assert.equal(el.livePill.textContent, 'PROMPT QUEUED · STARTS A NEW ROLLOUT');
});

test('unbounded continuous rollouts show a running count', async () => {
  const client = await loadClient({spec: {advance: 'continuous'}});
  client.connect();
  client.receive({type: 'ready', fps: 16, advance: 'continuous', rollout_chunks: 0, max_chunks: null, paused: false});
  client.receive({type: 'chunk_done', chunk_index: 12, frames: 4, advance: 'continuous', rollout_chunks: 12, max_chunks: null, paused: false});
  assert.match(client.el.streamStats.textContent, /CHUNK 12 \//);
  assert.equal(client.el.livePill.textContent, 'GENERATING · CHUNK 12');
});

test('PAUSE is not offered once the rollout is complete', async () => {
  const client = await loadClient({spec: {advance: 'continuous', max_chunks: 2}});
  client.connect();
  const rollout = (chunks) => ({advance: 'continuous', rollout_chunks: chunks, max_chunks: 2, paused: false});
  client.receive({type: 'ready', fps: 16, ...rollout(1)});
  assert.equal(client.el.pauseButton.disabled, false);
  client.receive({type: 'chunk_done', chunk_index: 2, frames: 4, ...rollout(2)});
  assert.equal(client.el.pauseButton.disabled, true);
  client.receive({type: 'chunk_done', chunk_index: 3, frames: 4, ...rollout(1)});
  assert.equal(client.el.pauseButton.disabled, false, 'a new rollout offers PAUSE again');
});

test('continuous sessions outside prompt-scheduled mode accept a new prompt with Enter', async () => {
  const client = await loadClient({spec: {advance: 'continuous', max_chunks: 7}});
  assert.match(client.el.interactionHint.textContent, /NEW PROMPT STARTS A NEW ROLLOUT/);
  const socket = client.connect(new FakeWebSocket());
  client.receive({type: 'ready', fps: 16, advance: 'continuous', rollout_chunks: 7, max_chunks: 7, paused: false});
  client.el.promptInput.value = 'snow falls';
  client.el.promptInput.dispatch('keydown', {key: 'Enter', preventDefault() {}});
  assert.deepEqual(socket.sent.at(-1), {type: 'prompt_update', prompt: 'snow falls'});
  assert.equal(client.el.livePill.textContent, 'PROMPT QUEUED · STARTS A NEW ROLLOUT');

  const onInput = await loadClient();
  const onInputSocket = onInput.connect(new FakeWebSocket());
  onInput.el.promptInput.value = 'snow falls';
  onInput.el.promptInput.dispatch('keydown', {key: 'Enter', preventDefault() {}});
  assert.equal(onInputSocket.sent.filter((m) => m.type === 'prompt_update').length, 0);
});

test('input-driven sessions keep the live status and hide pause', async () => {
  const client = await loadClient();
  client.connect();
  client.receive({type: 'ready', fps: 16});
  client.receive({type: 'chunk_done', chunk_index: 1, frames: 9});
  assert.equal(client.el.livePill.textContent, 'LIVE');
  assert.equal(client.el.pauseButton.classList.contains('is-hidden'), true);
  assert.match(client.el.streamStats.textContent, /E2E/);
  assert.equal(client.el.stepButton.disabled, false);
});

test('resetSession and markDisconnected clear cached upload paths and file inputs reset value', async () => {
  const client = await loadClient();
  const fakeFile = {name: 'scene.png'};
  client.el.imageInput.value = 'C:\\fakepath\\scene.png';
  client.el.imageInput.dispatch('change', {target: Object.assign(client.el.imageInput, {files: [fakeFile]})});
  assert.equal(client.el.imageInput.value, '');
  assert.equal(client.state.selectedFile, fakeFile);

  client.state.uploadedPath = '/uploads/img.png';
  client.state.denseUploadedPath = '/uploads/dense.mp4';
  client.state.sparseUploadedPath = '/uploads/sparse.mp4';
  await client.context.resetSession();
  assert.equal(client.state.uploadedPath, '');
  assert.equal(client.state.denseUploadedPath, '');
  assert.equal(client.state.sparseUploadedPath, '');
  assert.equal(client.state.selectedFile, fakeFile);

  client.state.uploadedPath = '/uploads/img2.png';
  client.state.denseUploadedPath = '/uploads/dense2.mp4';
  client.state.sparseUploadedPath = '/uploads/sparse2.mp4';
  client.context.markDisconnected();
  assert.equal(client.state.uploadedPath, '');
  assert.equal(client.state.denseUploadedPath, '');
  assert.equal(client.state.sparseUploadedPath, '');
});

test('WebRTC fallback failure recovers UI without soft-lock and connected failed peer disconnects cleanly', async () => {
  let resolveReset;
  const resetGate = new Promise((resolve) => { resolveReset = resolve; });
  const client = await loadClient({
    spec: {axes: MOUSE_AXES},
    fetchOverride: async (url) => {
      if (url === '/api/session/reset') {
        await resetGate;
        return {ok: true, json: async () => ({reset: true}), text: async () => '{"reset":true}'};
      }
      return undefined;
    },
  });
  client.state.seedPath = '/seed.png';
  await client.context.connectSession();
  const pc1 = client.peers[0];
  assert.equal(client.state.connecting, true);

  // Trigger WebRTC failure -> fallbackToSocket; state.connecting must stay true across reset.
  const fallbackPromise = client.context.fallbackToSocket({prompt: ''}, pc1);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(client.state.connecting, true, 'no re-entrancy window during reset');
  resolveReset();
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
  const fallbackSocket = client.sockets.at(-1);
  assert.ok(fallbackSocket);
  fallbackSocket.dispatch('close');
  await assert.rejects(fallbackPromise, /closed/);

  assert.equal(client.state.fallbackStarted, false);
  assert.equal(client.state.connecting, false);
  assert.equal(client.state.connected, false);
  assert.equal(client.state.socket, null);
  assert.equal(client.state.channel, null);
  assert.equal(client.el.startButton.disabled, false);
  assert.equal(client.el.startButton.classList.contains('is-hidden'), false);
  assert.match(client.el.livePill.textContent, /START ERROR/);

  // User can retry START cleanly, and a connected peer that later fails calls markDisconnected.
  await client.context.connectSession();
  const pc2 = client.peers[1];
  const controls2 = pc2.channels.find((c) => c.label === 'controls').channel;
  const axes2 = pc2.channels.find((c) => c.label === 'axes').channel;
  controls2.dispatch('open');
  assert.equal(client.state.connected, true);
  pc2.connectionState = 'failed';
  pc2.dispatch('connectionstatechange');
  assert.equal(client.state.connected, false);
  assert.equal(client.state.peer, null);
  assert.equal(client.state.channel, null);
  assert.equal(client.state.axisChannel, null);
  assert.equal(pc2.closed, true);
  assert.equal(controls2.closed, true);
  assert.equal(axes2.closed, true);
});

test('pumpSocketFrames checks playback epoch after waitMilliseconds and waitForAnimationFrame', async () => {
  const client = await loadClient();
  const bitmaps = [];
  client.context.createImageBitmap = async () => {
    const bm = {width: 16, height: 16, closed: false, close() { this.closed = true; }};
    bitmaps.push(bm);
    return bm;
  };
  let drawn = 0;
  client.el.canvas.getContext = () => ({drawImage() { drawn += 1; }});
  client.state.nextPresentationAt = 1000;
  client.state.socketFrameQueue.push(new Blob(['a']));
  const firstEpoch = client.state.socketPlaybackEpoch;
  const pump1 = client.context.pumpSocketFrames(firstEpoch);
  await new Promise((resolve) => setImmediate(resolve));
  // Advance epoch while waiting inside waitMilliseconds.
  client.state.socketPlaybackEpoch += 1;
  client.timers.pop().fn();
  await pump1;
  assert.equal(bitmaps[0].closed, true);
  assert.equal(drawn, 0);
  assert.equal(client.state.presentedFrames, 0);

  // Advance epoch while waiting inside waitForAnimationFrame.
  client.state.nextPresentationAt = 0;
  client.state.socketFrameQueue.push(new Blob(['b']));
  const secondEpoch = client.state.socketPlaybackEpoch;
  const pump2 = client.context.pumpSocketFrames(secondEpoch);
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(drawn, 1);
  client.state.socketPlaybackEpoch += 1;
  client.tick();
  await pump2;
  assert.equal(client.state.presentedFrames, 0);
  assert.equal(client.state.segmentPresentedFrames, 0);
});

test('pipelined segment_started does not wipe pending chunk counters and non-continuous segmentBusy locks STEP and events', async () => {
  const spec = {controls: ['prompt_update'], keys: [], advance: 'continuous', max_chunks: 4};
  const client = await loadClient({
    spec,
    interactionMode: 'prompt-scheduled',
    capabilities: {no_action_step: true, text_events: true},
  });
  client.connect(new FakeWebSocket());
  const rollout = (chunks) => ({advance: 'continuous', rollout_chunks: chunks, max_chunks: 4, paused: false});
  client.receive({type: 'ready', fps: 16, ...rollout(0)});

  // Chunk 1 starts, queues 2 frames, chunk_done(1) arrives while frames are still queued.
  client.receive({type: 'segment_started', segment_index: 1});
  client.state.socketFrameQueue.push(new Blob(['f1']), new Blob(['f2']));
  client.receive({type: 'chunk_done', chunk_index: 1, frames: 2, ...rollout(1)});
  assert.equal(client.state.pendingChunkDones.length, 1);

  // Chunk 2 starts and finishes before Chunk 1 finishes playing.
  client.receive({type: 'segment_started', segment_index: 2});
  client.state.socketFrameQueue.push(new Blob(['f3']), new Blob(['f4']));
  client.receive({type: 'chunk_done', chunk_index: 2, frames: 2, ...rollout(2)});
  assert.equal(client.state.pendingChunkDones.length, 2);

  // Present Chunk 1's 2 frames -> completes Chunk 1 and resets remainder for Chunk 2.
  client.state.segmentPresentedFrames = 2;
  client.context.drainPendingChunkCompletions();
  assert.equal(client.state.segmentIndex, 1);
  assert.equal(client.state.segmentPresentedFrames, 0);
  assert.equal(client.state.segmentBusy, true);
  assert.equal(client.state.pendingChunkDones.length, 1);

  // Present Chunk 2's 2 frames -> completes Chunk 2 cleanly.
  client.state.segmentPresentedFrames = 2;
  client.context.drainPendingChunkCompletions();
  assert.equal(client.state.segmentIndex, 2);
  assert.equal(client.state.segmentBusy, false);
  assert.equal(client.state.pendingChunkDones.length, 0);

  // In non-continuous prompt-scheduled mode, segmentBusy disables STEP and text event buttons.
  const nonCont = await loadClient({
    spec: {controls: ['prompt_update'], keys: []},
    interactionMode: 'prompt-scheduled',
    capabilities: {no_action_step: true, text_events: true},
  });
  nonCont.connect(new FakeWebSocket());
  nonCont.receive({
    type: 'ready',
    fps: 16,
    event_catalog: [{event_id: 'rain', label: 'Rain', prompt: 'heavy rain'}],
    catalog_revision: 1,
  });
  assert.equal(nonCont.el.stepButton.disabled, false);
  assert.equal(nonCont.el.eventTriggerBar.querySelectorAll('button').every((b) => !b.disabled), true);
  nonCont.receive({type: 'segment_started', segment_index: 1});
  assert.equal(nonCont.el.stepButton.disabled, true);
  assert.equal(nonCont.el.eventTriggerBar.querySelectorAll('button').every((b) => b.disabled), true);
  nonCont.receive({type: 'chunk_done', chunk_index: 1, frames: 0});
  assert.equal(nonCont.el.stepButton.disabled, false);
  assert.equal(nonCont.el.eventTriggerBar.querySelectorAll('button').every((b) => !b.disabled), true);
});

test('syncRuntimeStatus refreshes capabilities.text_events and loadModel refreshes runtime status', async () => {
  const client = await loadClient({
    capabilities: {no_action_step: true, text_events: false},
    status: {ready: false, loading: true, capabilities: {text_events: false}, realtime_spec: {}},
  });
  assert.equal(client.el.addEventButton.disabled, true);
  assert.match(client.el.eventCatalogStatus.textContent, /NO STATE-PRESERVING TEXT UPDATE/);

  client.responses['/api/runtime/status'] = {
    ready: true,
    loading: false,
    capabilities: {text_events: true},
    realtime_spec: {controls: ['prompt_update'], advance: 'continuous', max_chunks: 5},
  };
  await client.context.loadModel();
  assert.equal(client.state.capabilities.text_events, true);
  assert.equal(client.el.addEventButton.disabled, false);
  assert.equal(client.el.applyEventsButton.disabled, false);
  assert.equal(client.el.eventCatalogStatus.textContent, 'EMPTY CATALOG');
  assert.equal(client.state.continuous, true);
  assert.equal(client.state.maxChunks, 5);
});

test('held keyboard key releases on keyup even after focus moves into an input', async () => {
  const client = await loadClient();
  const channel = client.connect();
  client.key('w', 'keydown');
  const inputTarget = new FakeElement('input');
  client.key('w', 'keyup', inputTarget);
  assert.deepEqual(
    channel.sent.filter((m) => m.type === 'action').map((m) => [m.action.event, m.action.key]),
    [['keydown', 'w'], ['keyup', 'w']],
  );
  assert.equal(client.state.keySources.get('w').size, 0);
});

test('setSensorLayout skips DOM rebuild when layout signature is unchanged', async () => {
  const client = await loadClient();
  const layout = [
    {key: 'front', kind: 'rgb', label: 'Front', channel_index: 0, x: 0, y: 0, width: 0.5, height: 1},
    {key: 'lidar', kind: 'lidar', label: 'LiDAR', channel_index: 1, x: 0.5, y: 0, width: 0.5, height: 1},
  ];
  client.context.setSensorLayout(layout);
  const firstSelect = client.el.sensorControls.children[0];
  const firstTile = client.el.sensorGrid.children[0];
  client.context.setSensorLayout(structuredClone(layout));
  assert.equal(client.el.sensorControls.children[0], firstSelect);
  assert.equal(client.el.sensorGrid.children[0], firstTile);

  client.context.setSensorLayout([layout[0]]);
  assert.notEqual(client.el.sensorControls.children[0], firstSelect);
  assert.equal(client.el.sensorGrid.children.length, 1);
});

test('chunk_done syncs output resolution dropdown on server downgrade', async () => {
  const client = await loadClient({
    outputResolutions: [
      {mode: 'native', label: 'Native'},
      {mode: 'fixed', width: 640, height: 360, label: 'Stream · 640×360'},
    ],
  });
  client.connect();
  client.state.outputResolution = {mode: 'fixed', width: 640, height: 360};
  client.context.renderResolutionOptions(client.state.outputResolutionOptions);
  assert.equal(client.el.resolutionSelect.value, '640x360');

  client.receive({type: 'chunk_done', chunk_index: 1, frames: 4, resolution: {mode: 'native', width: 320, height: 180}});
  assert.equal(client.el.resolutionSelect.value, 'native');
  assert.deepEqual(JSON.parse(JSON.stringify(client.state.outputResolution)), {mode: 'native', width: 320, height: 180});
});
