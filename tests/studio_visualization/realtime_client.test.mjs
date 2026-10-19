// SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const file = new URL('../../worldfoundry/studio/visualization/backends/world_realtime_client.py', import.meta.url);
const script = readFileSync(file, 'utf8').split('WORLD_REALTIME_CLIENT_JS = r"""')[1].split('"""')[0];
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
