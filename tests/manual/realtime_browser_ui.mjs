// SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
// SPDX-License-Identifier: Apache-2.0

// Weight-free Chromium UI checks. HTTP/WebSocket model responses and gamepads
// are fixtures; HTML, styles, client code, keyboard and Pointer Lock are real.
import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const { expect } = require('playwright/test');
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const [output, mode, ...extra] = process.argv.slice(2);
assert.ok(output && (!mode || mode === '--physical-escape') && !extra.length,
  'Usage: node tests/manual/realtime_browser_ui.mjs <new-output-directory> [--physical-escape]');
const physicalEscape = mode === '--physical-escape';
await mkdir(dirname(resolve(output)), {recursive: true});
await mkdir(output);

const assets = JSON.parse(execFileSync(process.env.PYTHON || 'python', ['-c', `
import base64
import io
import json
from types import SimpleNamespace
from PIL import Image, ImageDraw
from worldfoundry.core.execution.realtime.contracts import RealtimeSpec
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.visualization.backends.world import (
    world_favicon_svg, world_frontend_css, world_frontend_html, world_frontend_js,
)
from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.realtime import MatrixGame2RealtimeSession

specs = {}
for mode in ('universal', 'templerun'):
    session = object.__new__(MatrixGame2RealtimeSession)
    session.runtime = SimpleNamespace(mode=mode)
    specs[mode] = session.realtime_spec().to_payload()
specs['continuous'] = RealtimeSpec(
    fps=16, first_chunk_frames=9, steady_chunk_frames=12, controls=('prompt_update',),
    keys=(), advance='continuous', max_chunks=7,
).to_payload()
image = Image.new('RGB', (640, 352), '#162f46')
draw = ImageDraw.Draw(image)
draw.rectangle((0, 240, 640, 352), fill='#345d52')
draw.rectangle((240, 95, 400, 240), fill='#507b95')
draw.text((24, 24), 'WorldFoundry UI fixture', fill='white')
encoded = io.BytesIO()
image.save(encoded, format='JPEG')
print(json.dumps({
    'html': {model: world_frontend_html(find_entry(model), StudioLaunchConfig(model_id=model, frontend='world'))
             for model in ('matrix-game-2', 'self-forcing')},
    'css': world_frontend_css(), 'js': world_frontend_js(), 'favicon': world_favicon_svg(),
    'specs': specs, 'frame': base64.b64encode(encoded.getvalue()).decode(),
}))
`], {cwd: root, env: {...process.env, PYTHONPATH: root}, encoding: 'utf8', maxBuffer: 2 * 1024 * 1024}));
const frame = Buffer.from(assets.frame, 'base64');
const server = createServer((request, response) => {
  const url = new URL(request.url, 'http://localhost');
  const files = {
    '/': ['text/html', assets.html[url.searchParams.get('model') || 'matrix-game-2']],
    '/world.css': ['text/css', assets.css],
    '/world.js': ['text/javascript', assets.js],
    '/favicon.svg': ['image/svg+xml', assets.favicon],
  };
  const asset = files[url.pathname];
  response.writeHead(asset ? 200 : 404, {'Content-Type': asset?.[0] || 'text/plain'});
  response.end(asset?.[1] || 'Not found');
});
await new Promise((resolve, reject) => {
  server.once('error', reject);
  server.listen(0, '127.0.0.1', resolve);
});
const url = `http://127.0.0.1:${server.address().port}`;
const report = {
  source_revision: execFileSync('git', ['rev-parse', 'HEAD'], {cwd: root, encoding: 'utf8'}).trim(),
  worktree_changes: execFileSync('git', ['status', '--porcelain'], {cwd: root, encoding: 'utf8'}).trim(),
  evidence_scope: physicalEscape
    ? 'Operator-assisted physical Escape with fixture model responses; no gamepad hardware, model inference or WebRTC replay'
    : 'Chromium UI with fixture HTTP/WebSocket responses and simulated gamepads; no model inference or WebRTC replay',
  sessions: [],
};
let browser;
let browserProcess;
let browserExited;
let browserProfile;
let browserLog = '';

async function launchBrowser() {
  browserProfile = await mkdtemp(join(resolve(output), 'chromium-profile-'));
  browserProcess = spawn(process.env.CHROMIUM_EXECUTABLE || chromium.executablePath(), [
    `--user-data-dir=${browserProfile}`, '--remote-debugging-port=0', '--remote-debugging-address=127.0.0.1',
    '--no-first-run', '--no-default-browser-check', '--disable-sync',
    '--autoplay-policy=no-user-gesture-required', 'about:blank',
  ], {stdio: ['ignore', 'ignore', 'pipe']});
  browserExited = new Promise(resolve => browserProcess.once('close', resolve));
  let timer;
  const endpoint = await new Promise((resolve, reject) => {
    timer = setTimeout(() => reject(new Error('Chromium did not expose its DevTools endpoint')), 30000);
    browserProcess.once('error', reject);
    browserProcess.once('exit', code => reject(new Error(`Chromium exited during startup: ${code}`)));
    browserProcess.stderr.on('data', data => {
      browserLog += data.toString();
      const match = browserLog.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) resolve(match[1]);
    });
  }).finally(() => clearTimeout(timer));
  // Use the default context without Playwright's forced-focus emulation, so a
  // background tab really loses focus. The profile belongs only to this run.
  return chromium.connectOverCDP(endpoint, {noDefaults: true});
}

async function foreground(page) {
  if (process.platform === 'darwin') {
    execFileSync('/usr/bin/osascript', ['-l', 'JavaScript', '-e',
      `ObjC.import('AppKit'); $.NSRunningApplication.runningApplicationWithProcessIdentifier(${browserProcess.pid}).activateWithOptions(2);`]);
  }
  await page.bringToFront();
  await expect.poll(() => page.evaluate(() => document.hasFocus())).toBe(true);
}

// Flush the client's animation-frame input batching without a fixed sleep.
async function nextFrames(page) {
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function fixturePage(name, mode) {
  const continuous = mode === 'continuous';
  const model = continuous ? 'self-forcing' : 'matrix-game-2';
  const spec = assets.specs[mode];
  const result = {name, mode, spec, checks: [], received: [], errors: []};
  report.sessions.push(result);
  const context = browser.contexts()[0];
  const previousPages = context.pages();
  const page = await context.newPage();
  for (const previous of previousPages) await previous.close();
  await page.setViewportSize({width: 1440, height: 1050});
  page.on('pageerror', error => result.errors.push(String(error)));
  page.on('console', message => { if (message.type() === 'error') result.errors.push(message.text()); });
  await page.addInitScript(() => {
    window.fixtureGamepads = [];
    Object.defineProperty(navigator, 'getGamepads', {value: () => window.fixtureGamepads});
  });
  const session = {
    model_id: model, interaction_mode: continuous ? 'prompt-scheduled' : 'controls',
    prefer_websocket: true, transport: 'websocket', fps: spec.fps, realtime_spec: spec,
    capabilities: {no_action_step: true, text_events: continuous}, examples: [],
    output_resolution: {mode: 'native'}, output_resolution_scope: 'transport',
  };
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname;
    const responses = {
      '/api/session': session,
      '/api/runtime/status': {ready: true, realtime_spec: spec},
      '/api/runtime/load': {ready: true},
      '/api/session/input': {path: '/fixture/seed.jpg'},
      '/api/session/reset': {ok: true},
    };
    assert.ok(path in responses, `Unexpected API request: ${path}`);
    await route.fulfill({json: responses[path]});
  });
  let socket;
  let chunks = 0;
  let paused = false;
  const progress = () => ({advance: continuous ? 'continuous' : 'on_input',
    rollout_chunks: chunks, max_chunks: continuous ? spec.max_chunks : null, paused});
  const send = message => socket.send(JSON.stringify(message));
  await page.routeWebSocket('**/api/realtime/ws', connection => {
    socket = connection;
    connection.onMessage(raw => {
      const message = JSON.parse(raw);
      result.received.push(message);
      if (message.type === 'configure') {
        chunks = 0;
        paused = false;
        send({type: 'ready', fps: spec.fps, ...progress()});
      } else if (message.type === 'pause' || message.type === 'resume') {
        paused = message.type === 'pause';
        send({type: `${message.type}_ack`, ok: true, request_id: message.request_id, ...progress()});
      } else if (message.type === 'action' && message.action.event === 'step') {
        const complete = continuous && chunks >= spec.max_chunks;
        send({type: 'step_ack', ok: !complete, request_id: message.request_id, queued_steps: complete ? 0 : 1,
          ...(complete ? {message: 'The rollout is complete; send a new prompt to start another.'} : {})});
      } else if (message.type === 'prompt_update') {
        chunks = 0;
        paused = false;
      }
    });
  });
  await foreground(page);
  await page.goto(`${url}/?model=${model}`, {waitUntil: 'domcontentloaded'});
  await expect(page.locator('#loadModelButton')).toHaveText('READY');
  await expect(page.locator('#stepButton')).toBeDisabled();
  return {page, context, result, send,
    async start() {
      if (!continuous) await page.locator('#imageInput').setInputFiles({name: 'seed.jpg', mimeType: 'image/jpeg', buffer: frame});
      await page.locator('#startButton').click();
      await expect(page.locator('#resetButton')).toBeVisible();
      await expect.poll(() => result.received.filter(m => m.type === 'configure').length).toBe(1);
      assert.equal(await page.evaluate(() => state.transport), 'WS');
    },
    async chunk() {
      await foreground(page);
      await expect.poll(() => page.evaluate(() => document.visibilityState)).toBe('visible');
      const frames = chunks ? spec.steady_chunk_frames : spec.first_chunk_frames;
      chunks += 1;
      send({type: 'segment_started', segment_index: chunks});
      for (let index = 0; index < frames; index++) socket.send(frame);
      send({type: 'chunk_done', chunk_index: chunks, frames, generation_ms: 0, ...progress()});
      await page.waitForFunction(count => state.segmentPresentedFrames >= count && !state.segmentBusy, frames);
      await expect(page.locator('#streamStats')).toContainText(`CHUNK ${chunks}/${spec.max_chunks}`);
    },
    async finish() {
      await page.locator('#resetButton').click();
      await expect(page.locator('#startButton')).toBeVisible();
      await expect(page.locator('#stepButton')).toBeDisabled();
      assert.equal(await page.evaluate(() => !state.connected && !state.channel && !state.pointerLocked
        && !state.gamepadRaf && !state.mouseRaf && state.requestTimers.size === 0
        && Object.values(state.stickPointers).every(id => id === null)), true);
      result.checks.push('reset clears input polling, pointer lock and pending requests');
    },
  };
}

async function checkInputs(fixture) {
  const {page, result} = fixture;
  const legend = page.locator('#interactionHint');
  await expect(legend).toContainText('HOLD WASD / DRAG STICKS');
  await expect(legend).toContainText('CLICK VIEW FOR MOUSE');
  await expect(legend).not.toContainText('GAMEPAD');
  await fixture.start();
  await page.keyboard.press('w');
  await page.keyboard.press('z');
  await nextFrames(page);
  assert.deepEqual(result.received.filter(m => m.type === 'action').map(m => m.action),
    [{event: 'keydown', key: 'w'}, {event: 'keyup', key: 'w'}]);
  result.checks.push('default keyboard bindings and undeclared-key filtering');

  const mouseMessages = () => result.received.filter(m => m.type === 'axis' && m.source === 'mouse');
  const viewport = page.locator('#viewport');
  await viewport.scrollIntoViewIfNeeded();
  await expect.poll(() => page.evaluate(() => document.hasFocus())).toBe(true);
  const box = await viewport.boundingBox();
  const x = box.x + box.width / 2, y = box.y + box.height / 2;
  await page.mouse.move(x - 40, y - 20);
  await nextFrames(page);
  assert.equal(mouseMessages().length, 0);
  await page.mouse.click(x, y);
  await expect(page.locator('#pointerHint')).toHaveText('MOUSE LOOK · ESC TO RELEASE');
  assert.equal(await page.evaluate(() => document.pointerLockElement?.id), 'viewport');
  await page.mouse.move(x + 24, y - 12, {steps: 3});
  await expect.poll(() => mouseMessages().some(m => m.dx !== 0 || m.dy !== 0)).toBe(true);
  await page.screenshot({path: join(output, `${result.name}-pointer-lock.png`), fullPage: true});
  await page.evaluate(() => document.exitPointerLock());
  await expect(page.locator('#pointerHint')).toHaveText('CLICK VIEW FOR MOUSE LOOK');
  const count = mouseMessages().length;
  await page.mouse.move(x - 20, y + 20);
  await nextFrames(page);
  assert.equal(mouseMessages().length, count);
  result.checks.push('real Pointer Lock, mouse deltas and release through the browser API');

  const gamepadMessages = () => result.received.filter(m => m.type === 'axis' && m.source === 'gamepad');
  await page.evaluate(() => {
    window.fixtureGamepads = [{connected: true, axes: [0, 0, 0.75, 0]}];
    window.dispatchEvent(new Event('gamepadconnected'));
  });
  await expect(legend).toContainText('GAMEPAD');
  await expect.poll(() => gamepadMessages().at(-1)?.rx).toBe(0.75);
  await page.evaluate(() => { window.fixtureGamepads[0].axes = [0, 0, 0, 0]; });
  await expect.poll(() => gamepadMessages().at(-1)?.rx).toBe(0);
  await page.evaluate(() => { window.fixtureGamepads[0].axes = [0, 0, -0.75, 0]; });
  await expect.poll(() => gamepadMessages().at(-1)?.rx).toBe(-0.75);
  const other = await page.context().newPage();
  await other.bringToFront();
  await expect.poll(() => page.evaluate(() => document.hasFocus())).toBe(false);
  await expect.poll(() => gamepadMessages().at(-1)?.rx).toBe(0);
  await other.close();
  await foreground(page);
  await expect.poll(() => gamepadMessages().at(-1)?.rx).toBe(-0.75);
  await page.evaluate(() => {
    window.fixtureGamepads = [];
    window.dispatchEvent(new Event('gamepaddisconnected'));
  });
  await expect(legend).not.toContainText('GAMEPAD');
  await expect.poll(() => gamepadMessages().at(-1)?.rx).toBe(0);
  result.checks.push('simulated gamepad legend, movement, rest, focus loss and disconnect');
  await fixture.finish();
}

async function checkVirtualSticks(fixture) {
  const {page, result} = fixture;
  await fixture.start();
  const axes = () => result.received.filter(m => m.type === 'axis' && m.source === 'gamepad');
  const actions = () => result.received.filter(m => m.type === 'action').map(m => m.action);
  const drag = async (kind, dx, dy) => {
    await foreground(page);
    const stick = page.locator(`#${kind}Stick`);
    await stick.scrollIntoViewIfNeeded();
    const box = await stick.boundingBox();
    const x = box.x + box.width / 2, y = box.y + box.height / 2;
    await page.mouse.move(x, y);
    await page.mouse.down();
    await page.mouse.move(x + dx * box.width / 2, y + dy * box.height / 2, {steps: 3});
    await nextFrames(page);
    return {x, y};
  };
  await drag('move', 0, -0.6);
  await expect.poll(() => actions().some(m => m.event === 'keydown' && m.key === 'w')).toBe(true);
  await page.mouse.up();
  await expect.poll(() => actions().at(-1)).toEqual({event: 'keyup', key: 'w'});
  assert.equal(axes().length, 0, 'MG2 movement keeps its declared keyboard mapping');
  result.checks.push('movement stick retains key edges when no left-stick axes are declared');

  await drag('look', 0.6, -0.4);
  await expect.poll(() => axes().some(m => m.rx > 0.4 && m.ry < -0.2)).toBe(true);
  assert.ok(actions().every(m => m.key === 'w'), 'the camera stick does not also send look keys');
  assert.equal(await page.evaluate(() => state.pointerLocked), false);
  await page.screenshot({path: join(output, `${result.name}-dragging.png`), fullPage: true});
  await page.mouse.up();
  await expect.poll(() => axes().at(-1)?.rx).toBe(0);
  assert.equal(axes().at(-1).ry, 0);
  result.checks.push('camera stick sends declared continuous axes and a reliable return to rest');

  const origin = await drag('look', 0.6, 0);
  await expect.poll(() => axes().at(-1)?.rx > 0.4).toBe(true);
  const other = await page.context().newPage();
  await other.bringToFront();
  await expect.poll(() => page.evaluate(() => document.hasFocus())).toBe(false);
  await expect.poll(() => axes().at(-1)?.rx).toBe(0);
  await other.close();
  await foreground(page);
  await page.mouse.move(origin.x + 10, origin.y);
  await page.mouse.up();
  await nextFrames(page);
  assert.equal(axes().at(-1).rx, 0);
  assert.equal(await page.evaluate(() => state.stickPointers.look), null);
  result.checks.push('tab focus loss releases virtual axes and cancels the captured drag');
  await fixture.finish();
}

async function checkPhysicalEscape(fixture) {
  const {page, result} = fixture;
  await fixture.start();
  await page.evaluate(() => {
    const observation = window.fixturePhysicalEscape = {armed: false, completed: false, release: null, events: []};
    const notice = document.createElement('div');
    notice.id = 'physicalEscapeCheck';
    notice.setAttribute('role', 'status');
    notice.style.cssText = 'position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:9999;padding:14px 22px;background:#122c40;color:white;border:2px solid #74c4ee;border-radius:10px;pointer-events:none;font:16px sans-serif;max-width:90vw';
    notice.textContent = 'PHYSICAL ESCAPE CHECK: Click the video, press your keyboard’s Esc, then wait for PASS.';
    document.body.appendChild(notice);
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape') observation.events.push({type: 'keydown', key: event.key, trusted: event.isTrusted});
    }, true);
    document.addEventListener('pointerlockchange', event => {
      const locked = document.pointerLockElement?.id === 'viewport';
      const entry = {type: 'pointerlockchange', locked, focused: document.hasFocus(),
        visibility: document.visibilityState, trusted: event.isTrusted};
      observation.events.push(entry);
      if (locked) {
        observation.armed = true;
        notice.textContent = 'POINTER LOCKED — press the physical Esc key now.';
      } else if (observation.armed && entry.focused && entry.visibility === 'visible' && entry.trusted) {
        observation.release = entry;
        observation.armed = false;
        notice.textContent = 'Native release received — checking that mouse input stopped…';
      }
    });
    window.addEventListener('blur', () => {
      observation.armed = false;
      if (!observation.completed) observation.release = null;
      observation.events.push({type: 'blur'});
    });
  });
  console.log('PHYSICAL_ESCAPE_READY: In the test Chrome window, click the video and press your physical Esc key.');
  await page.waitForFunction(() => window.fixturePhysicalEscape.release !== null, null, {timeout: 600000});
  await expect(page.locator('#pointerHint')).toHaveText('CLICK VIEW FOR MOUSE LOOK');
  const count = result.received.filter(m => m.type === 'axis' && m.source === 'mouse').length;
  await page.mouse.move(500, 300);
  await nextFrames(page);
  assert.equal(result.received.filter(m => m.type === 'axis' && m.source === 'mouse').length, count);
  result.physical_escape = await page.evaluate(() => {
    const observation = window.fixturePhysicalEscape;
    if (!document.hasFocus() || !observation.release || document.pointerLockElement || state.pointerLocked) {
      throw new Error('Physical Escape must release the pointer while the test page stays focused.');
    }
    observation.completed = true;
    document.getElementById('physicalEscapeCheck').textContent = 'PASS — physical Escape released the pointer and mouse look stopped.';
    return observation;
  });
  result.checks.push('operator-triggered native pointer release while focused; subsequent mouse movement sends no axes');
  await fixture.finish();
}

async function checkTemplerun(fixture) {
  const {page, result} = fixture;
  await expect(page.locator('#interactionHint')).toHaveText('HOLD W S A D Z C / DRAG STICKS');
  await expect(page.locator('#lookStick')).toBeHidden();
  await fixture.start();
  await expect(page.locator('#pointerHint')).toBeHidden();
  await page.keyboard.press('z');
  await page.keyboard.press('i');
  await nextFrames(page);
  assert.deepEqual(result.received.filter(m => m.type === 'action').map(m => m.action),
    [{event: 'keydown', key: 'z'}, {event: 'keyup', key: 'z'}]);
  result.checks.push('mode-specific extra keys, hidden camera stick and no mouse prompt');
  await fixture.finish();
}

async function checkContinuous(fixture) {
  const {page, result} = fixture;
  await expect(page.locator('#moveStick')).toBeHidden();
  await expect(page.locator('#pauseButton')).toBeVisible();
  await expect(page.locator('#pauseButton')).toBeDisabled();
  await fixture.start();
  await expect(page.locator('#stepButton')).toBeDisabled();
  await fixture.chunk();
  await expect(page.locator('#livePill')).toHaveText('GENERATING · CHUNK 1/7');
  await page.locator('#pauseButton').click();
  await expect(page.locator('#pauseButton')).toHaveText('RESUME');
  await expect(page.locator('#pauseButton')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#stepButton')).toBeEnabled();
  await page.locator('#stepButton').click();
  await expect(page.locator('#controlAck')).toHaveText('STEP ACK');
  await fixture.chunk();
  await expect(page.locator('#livePill')).toHaveText('PAUSED · CHUNK 2/7');
  await page.locator('#pauseButton').click();
  await expect(page.locator('#pauseButton')).toHaveText('PAUSE');
  await expect(page.locator('#stepButton')).toBeDisabled();
  result.checks.push('continuous progress, pause, single step and resume');
  for (let index = 0; index < 5; index++) await fixture.chunk();
  await expect(page.locator('#livePill')).toHaveText('ROLLOUT COMPLETE · 7/7 · ENTER A NEW PROMPT');
  await expect(page.locator('#pauseButton')).toBeDisabled();
  await page.locator('#stepButton').click();
  await expect(page.locator('#controlAck')).toHaveText('The rollout is complete; send a new prompt to start another.');
  await expect(page.locator('#controlAck')).toHaveClass(/is-error/);
  await page.screenshot({path: join(output, `${result.name}-complete.png`), fullPage: true});
  await page.locator('#promptInput').fill('A lighthouse at dusk');
  await page.locator('#promptInput').press('Enter');
  await expect.poll(() => result.received.filter(m => m.type === 'prompt_update').at(-1)?.prompt).toBe('A lighthouse at dusk');
  await expect(page.locator('#livePill')).toHaveText('PROMPT QUEUED · STARTS A NEW ROLLOUT');
  await fixture.chunk();
  await expect(page.locator('#pauseButton')).toBeEnabled();
  result.checks.push('completed-rollout STEP rejection and prompt restart');
  await fixture.finish();
}

try {
  browser = await launchBrowser();
  report.browser_version = browser.version();
  const sessions = physicalEscape ? [['physical-escape', 'universal', checkPhysicalEscape]] : [
    ['mouse-gamepad', 'universal', checkInputs],
    ['virtual-sticks', 'universal', checkVirtualSticks],
    ['templerun-keys', 'templerun', checkTemplerun],
    ['continuous-rollout', 'continuous', checkContinuous],
  ];
  for (const [name, mode, check] of sessions) {
    let fixture;
    try {
      fixture = await fixturePage(name, mode);
      await check(fixture);
      assert.deepEqual(fixture.result.errors, []);
      assert.equal(await fixture.page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      fixture.result.passed = true;
    } catch (error) {
      const result = report.sessions.at(-1);
      result.errors.push(String(error));
      result.passed = false;
      if (fixture) {
        result.page_state = await fixture.page.evaluate(() => ({
          focused: document.hasFocus(), visibility: document.visibilityState,
          pointer_lock: document.pointerLockElement?.id || null,
          runtime_log: document.getElementById('runtimeLogList').textContent,
        }));
        result.page_state.tabs = fixture.context.pages().map(page => page.url());
      }
      console.error(`${name}: ${error}`);
    } finally {
      if (fixture) {
        if (physicalEscape) {
          fixture.result.physical_escape = await fixture.page.evaluate(() => window.fixturePhysicalEscape || null);
        }
        await fixture.page.screenshot({path: join(output, `${name}.png`), fullPage: true});
      }
      await writeFile(join(output, 'browser.json'), JSON.stringify(report, null, 2));
    }
  }
} finally {
  await browser?.close();
  browserProcess?.kill();
  await browserExited;
  if (browserProfile) await rm(browserProfile, {recursive: true, force: true});
  await writeFile(join(output, 'chromium.log'), browserLog);
  await new Promise(resolve => server.close(resolve));
}
if (report.sessions.some(result => !result.passed)) process.exitCode = 1;
else console.log(`Passed ${report.sessions.length} browser UI sessions; artifacts: ${output}`);
