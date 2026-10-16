// SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { candidateEndpoint, createRelay, routeCandidates } from './realtime_udp.mjs';

const require = createRequire(import.meta.url);
const [profilePath, url, output] = process.argv.slice(2);
const profile = JSON.parse(await readFile(profilePath, 'utf8'));
const { chromium } = require('playwright');
const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE || undefined,
  args: ['--no-sandbox', '--autoplay-policy=no-user-gesture-required', '--disable-features=WebRtcHideLocalIpsWithMdns']});
const results = {browser_version: browser.version(), sessions: []};

try {
  for (const network of profile.networks) {
    for (let repeat = 0; repeat < profile.repeats; repeat++) {
      const name = `${network.name}-${repeat + 1}`;
      console.log(`Replay ${name}`);
      const relay = await createRelay(network, profile.seed + repeat);
      const page = await browser.newPage({viewport: {width: 1440, height: 1050}});
      const result = {name, network: network.name, repeat, started_wall_ms: Date.now(), errors: []};
      results.sessions.push(result);
      page.on('pageerror', error => result.errors.push(String(error)));
      await page.route('**/api/webrtc/offer', async route => {
        try {
          const payload = route.request().postDataJSON();
          payload.session.seed = profile.seed;
          relay.setBrowser(candidateEndpoint(payload.offer.sdp));
          payload.offer.sdp = routeCandidates(payload.offer.sdp, relay.back);
          const response = await route.fetch({postData: JSON.stringify(payload), timeout: profile.timeout_ms});
          assert.equal(response.status(), 200, await response.text());
          const answer = await response.json();
          relay.setServer(candidateEndpoint(answer.sdp));
          answer.sdp = routeCandidates(answer.sdp, relay.front);
          await route.fulfill({response, json: answer});
        } catch (error) {
          result.errors.push(String(error));
          await route.abort();
        }
      });
      try {
        await page.goto(url, {waitUntil: 'networkidle', timeout: profile.timeout_ms});
        const session = await page.evaluate(() => api('/api/session'));
        assert.equal(session.model_id, profile.model_id);
        assert.equal(session.interaction_mode, 'controls', 'STEP replay requires a resident interactive model');
        result.spec = session.realtime_spec;
        await page.waitForFunction(() => state.modelLoaded, null, {timeout: profile.timeout_ms});
        await page.evaluate(({image, prompt}) => {
          state.seedPath = image;
          state.selectedFile = null;
          state.preferWebSocket = false;
          state.iceServers = [];
          el.promptInput.value = prompt;
          updateControlAvailability();
        }, profile);
        await page.locator('#startButton').click();
        await page.waitForFunction(() => state.connected, null, {timeout: profile.timeout_ms});
        assert.equal(await page.evaluate(() => state.transport), 'RTC', 'WebSocket fallback is not WebRTC evidence');
        result.connected_wall_ms = Date.now();
        await page.evaluate(() => {
          window.replay = {sent: [], received: [], frames: [], callbacks: [], measuredStart: null, measuredEnd: null};
          const channel = state.channel;
          const send = channel.send.bind(channel);
          channel.send = raw => {
            const message = JSON.parse(raw);
            if (message.control_id || ['frame_presented', 'browser_transport'].includes(message.type)) {
              window.replay.sent.push({...message, browser_ms: performance.now()});
            }
            send(raw);
          };
          channel.addEventListener('message', event => {
            const message = JSON.parse(event.data);
            if (['chunk_done', 'error'].includes(message.type)) window.replay.received.push(message);
          });
          const collector = state.presentation;
          const match = collector.match.bind(collector);
          collector.match = timestamp => {
            const receipt = collector.sent.get(timestamp);
            const displayed = collector.displayed.get(timestamp);
            if (receipt && displayed !== undefined) window.replay.frames.push({...receipt, browser_ms: displayed});
            match(timestamp);
          };
          const presented = collector.presented.bind(collector);
          collector.presented = (timestamp, now) => {
            window.replay.callbacks.push(now);
            presented(timestamp, now);
          };
        });
        const total = profile.warmup_steps + profile.steps;
        for (let step = 0; step < total; step++) {
          if (step === profile.warmup_steps) {
            result.measured_start_wall_ms = Date.now();
            await page.evaluate(() => { window.replay.measuredStart = performance.now(); });
          }
          if (network.outage && step === profile.warmup_steps + network.outage.after_steps) {
            relay.interrupt(network.outage.duration_ms);
          }
          await page.locator('#stepButton').click();
          // Each STEP consumes one native model chunk. Wait for that chunk's
          // final source frame, not for a held video frame or a server enqueue.
          await page.waitForFunction(chunk => {
            const done = window.replay.received.find(x => x.type === 'chunk_done' && x.chunk_index === chunk);
            return done && window.replay.frames.some(x => x.chunk_index === chunk && x.frame_index === done.frames - 1);
          }, step + 1, {timeout: profile.timeout_ms});
        }
        // Flush at least two receiver-stat intervals, also exercising held frames.
        await page.waitForTimeout(4200);
        await page.evaluate(() => { window.replay.measuredEnd = performance.now(); });
        result.measured_end_wall_ms = Date.now();
        result.observations = await page.evaluate(() => window.replay);
        result.rtc = await page.evaluate(async () => {
          const all = await state.peer.getStats();
          const reports = [...all.values()];
          const transport = reports.find(x => x.type === 'transport' && x.selectedCandidatePairId);
          const pair = transport && all.get(transport.selectedCandidatePairId);
          const inbound = reports.find(x => x.type === 'inbound-rtp' && x.kind === 'video');
          return {remote: pair && all.get(pair.remoteCandidateId), inbound,
            codec: inbound && all.get(inbound.codecId), stats: state.transportStats,
            label: el.streamStats.textContent, transport: state.transport};
        });
        assert.equal(result.rtc.transport, 'RTC');
        assert.equal(result.rtc.remote?.port, relay.front.port, 'ICE bypassed the UDP relay');
        assert.equal(result.rtc.remote?.address, relay.front.address);
        await page.screenshot({path: join(output, `${name}.png`), fullPage: true});
      } catch (error) {
        result.errors.push(String(error));
        result.observations = await page.evaluate(() => window.replay || null).catch(() => null);
      } finally {
        const resetStarted = Date.now();
        try {
          await page.evaluate(() => resetSession());
          result.reset_ms = Date.now() - resetStarted;
          result.reset_clean = await page.evaluate(() => !state.peer && !state.connected
            && state.transportStatsTimer === null && state.presentation.controls.size === 0
            && state.presentation.sent.size === 0 && state.presentation.displayed.size === 0);
          assert.equal(result.reset_clean, true);
        } catch (error) { result.errors.push(`reset: ${error}`); }
        result.ended_wall_ms = Date.now();
        await page.close();
        await relay.close();
        result.link = relay.snapshot();
        await writeFile(join(output, 'browser.json'), JSON.stringify(results, null, 2));
      }
    }
  }
} finally {
  await browser.close();
}
if (results.sessions.some(x => x.errors.length || x.link.errors.length)) process.exitCode = 1;
