// SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import dgram from 'node:dgram';
import { performance } from 'node:perf_hooks';
import { setTimeout as sleep } from 'node:timers/promises';
import { test } from 'node:test';
import { candidateEndpoint, createRelay, DatagramLink, routeCandidates } from '../manual/realtime_udp.mjs';

const clean = {delay_ms: 0, jitter_ms: 0, rate_bps: 0, loss: 0, queue_ms: 250};

test('SDP routes every bundled medium through one relay without fallback candidates', () => {
  const sdp = ['v=0', 'm=video 9 UDP/TLS/RTP/SAVPF 96',
    'a=candidate:1 1 udp 100 192.168.1.10 10000 typ host',
    'a=candidate:2 1 tcp 50 192.168.1.10 9 typ host tcptype active',
    'a=candidate:3 1 udp 90 ::1 10001 typ host', 'a=end-of-candidates',
    'm=application 9 UDP/DTLS/SCTP webrtc-datachannel',
    'a=candidate:4 1 udp 100 192.168.1.10 10002 typ host', ''].join('\r\n');
  assert.deepEqual(candidateEndpoint(sdp), {address: '192.168.1.10', port: 10000});
  const routed = routeCandidates(sdp, {address: '127.0.0.1', port: 11000});
  assert.equal(routed.split('\r\n').filter(x => x.startsWith('a=candidate:')).length, 2);
  assert.ok(!routed.includes('192.168.1.10') && !routed.includes('::1'));
  assert.deepEqual(candidateEndpoint(routed), {address: '127.0.0.1', port: 11000});
  assert.throws(() => candidateEndpoint('a=candidate:1 1 udp 100 random.local 10000 typ host'), /IPv4/);
});

test('rate serialization, propagation delay and bounded queue apply to real datagrams', async () => {
  const arrived = [];
  const started = performance.now();
  const link = new DatagramLink({...clean, rate_bps: 80000, delay_ms: 15, queue_ms: 150}, 42,
    data => arrived.push({at: performance.now() - started, bytes: data.length}));
  try {
    link.send(Buffer.alloc(1000));
    link.send(Buffer.alloc(1000));
    link.send(Buffer.alloc(1000));
    await sleep(300);
    assert.equal(arrived.length, 2);
    assert.ok(arrived[0].at >= 100);
    assert.ok(arrived[1].at >= 200);
    assert.equal(link.stats.queue_drops, 1);
    assert.equal(link.stats.delivered_bytes, 2000);
  } finally { link.close(); }
});

test('loss is seeded and shutdown cancels pending datagrams', async () => {
  const make = () => new DatagramLink({...clean, loss: 0.2, delay_ms: 100}, 42, () => assert.fail('closed link delivered'));
  const a = make(), b = make();
  for (let i = 0; i < 100; i++) { a.send(Buffer.alloc(10)); b.send(Buffer.alloc(10)); }
  assert.equal(a.stats.loss_drops, b.stats.loss_drops);
  assert.ok(a.stats.loss_drops > 0 && a.stats.loss_drops < 100);
  a.close(); b.close();
  await sleep(130);
  assert.equal(a.stats.delivered_packets, 0);
  assert.equal(a.stats.cancelled_packets + a.stats.loss_drops, 100);
});

test('bidirectional relay forwards UDP and recovers after interruption', async () => {
  const server = dgram.createSocket('udp4'), browser = dgram.createSocket('udp4');
  const bind = socket => new Promise(resolve => socket.bind(0, '127.0.0.1', resolve));
  await Promise.all([bind(server), bind(browser)]);
  const relay = await createRelay({uplink: clean, downlink: clean}, 42);
  const received = [];
  server.on('message', (data, remote) => server.send(data, remote.port, remote.address));
  browser.on('message', data => received.push(data.toString()));
  relay.setBrowser(browser.address());
  relay.setServer(server.address());
  try {
    browser.send(Buffer.from('first'), relay.front.port, relay.front.address);
    await sleep(50);
    assert.deepEqual(received, ['first']);
    relay.interrupt(100);
    browser.send(Buffer.from('lost'), relay.front.port, relay.front.address);
    await sleep(130);
    browser.send(Buffer.from('recovered'), relay.front.port, relay.front.address);
    await sleep(50);
    assert.deepEqual(received, ['first', 'recovered']);
    assert.equal(relay.snapshot().uplink.outage_drops, 1);
    assert.equal(relay.snapshot().downlink.delivered_packets, 2);
    assert.deepEqual(relay.snapshot().errors, []);
  } finally {
    await relay.close();
    server.close(); browser.close();
  }
});
