// SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
// SPDX-License-Identifier: Apache-2.0

// A process-local, bidirectional UDP link. Both SDP candidate sets must point
// through it; HTTP throttling does not shape WebRTC media.
import dgram from 'node:dgram';
import { isIP } from 'node:net';
import { performance } from 'node:perf_hooks';

export function candidateEndpoint(sdp) {
  for (const line of sdp.split(/\r?\n/)) {
    if (!line.startsWith('a=candidate:')) continue;
    const fields = line.split(' ');
    if (fields[1] === '1' && fields[2].toLowerCase() === 'udp' && isIP(fields[4]) === 4) {
      return {address: fields[4], port: Number(fields[5])};
    }
  }
  throw new Error('Replay needs an IPv4 UDP host candidate; disable Chromium WebRtcHideLocalIpsWithMdns.');
}

export function routeCandidates(sdp, endpoint) {
  const result = [];
  let emitted = false;
  for (const line of sdp.split(/\r?\n/)) {
    if (line.startsWith('m=')) emitted = false;
    if (!line.startsWith('a=candidate:')) { result.push(line); continue; }
    const fields = line.split(' ');
    if (emitted || fields[1] !== '1' || fields[2].toLowerCase() !== 'udp') continue;
    // Do not retain alternate candidates: ICE must not bypass the shaped link.
    result.push(`a=candidate:replay 1 UDP ${fields[3]} ${endpoint.address} ${endpoint.port} typ host`);
    emitted = true;
  }
  return result.join('\r\n');
}

export class DatagramLink {
  constructor(profile, seed, deliver) {
    for (const name of ['delay_ms', 'jitter_ms', 'rate_bps', 'loss', 'queue_ms']) {
      if (!Number.isFinite(profile[name]) || profile[name] < 0) throw new Error(`Invalid link ${name}`);
    }
    if (profile.loss > 1) throw new Error('Link loss must be between zero and one');
    this.profile = profile;
    this.seed = seed >>> 0 || 1;
    this.deliver = deliver;
    this.next = 0;
    this.blockedUntil = 0;
    this.timers = new Set();
    this.stats = {offered_packets: 0, offered_bytes: 0, delivered_packets: 0, delivered_bytes: 0,
      loss_drops: 0, queue_drops: 0, outage_drops: 0, cancelled_packets: 0};
  }
  random() {
    let x = this.seed;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    this.seed = x >>> 0;
    return this.seed / 4294967296;
  }
  send(data) {
    this.stats.offered_packets++;
    this.stats.offered_bytes += data.length;
    if (performance.now() < this.blockedUntil) { this.stats.outage_drops++; return; }
    if (this.random() < this.profile.loss) { this.stats.loss_drops++; return; }
    const now = performance.now();
    const start = Math.max(now, this.next);
    if (start - now > this.profile.queue_ms) { this.stats.queue_drops++; return; }
    this.next = start + (this.profile.rate_bps ? data.length * 8000 / this.profile.rate_bps : 0);
    const delay = Math.max(0, this.next - now + this.profile.delay_ms
      + (2 * this.random() - 1) * this.profile.jitter_ms);
    const timer = setTimeout(() => {
      this.timers.delete(timer);
      this.deliver(data);
      this.stats.delivered_packets++;
      this.stats.delivered_bytes += data.length;
    }, delay);
    this.timers.add(timer);
  }
  close() {
    this.stats.cancelled_packets += this.timers.size;
    for (const timer of this.timers) clearTimeout(timer);
    this.timers.clear();
  }
}

async function bind(socket) {
  await new Promise((resolve, reject) => {
    socket.once('error', reject);
    socket.bind(0, '127.0.0.1', resolve);
  });
  return socket.address();
}

export async function createRelay(profile, seed) {
  const front = dgram.createSocket('udp4');
  const back = dgram.createSocket('udp4');
  const frontAddress = await bind(front);
  let backAddress;
  try { backAddress = await bind(back); } catch (error) { front.close(); throw error; }
  let browser, server;
  let preAnswerDrops = 0;
  const errors = [];
  const transmit = (socket, target, data) => {
    if (!target) return; // ICE retries checks sent before the answer is installed.
    socket.send(data, target.port, target.address, error => { if (error) errors.push(String(error)); });
  };
  const uplink = new DatagramLink(profile.uplink, seed, data => transmit(back, server, data));
  const downlink = new DatagramLink(profile.downlink, seed ^ 0x13579bdf, data => transmit(front, browser, data));
  front.on('message', (data, remote) => {
    browser = {address: remote.address, port: remote.port};
    if (server) uplink.send(data);
    else preAnswerDrops++;
  });
  back.on('message', data => downlink.send(data));
  front.on('error', error => errors.push(String(error)));
  back.on('error', error => errors.push(String(error)));
  return {
    front: frontAddress,
    back: backAddress,
    setBrowser(value) { browser = value; },
    setServer(value) { server = value; },
    interrupt(durationMs) {
      uplink.blockedUntil = downlink.blockedUntil = performance.now() + durationMs;
    },
    snapshot() {
      return {uplink: {...uplink.stats}, downlink: {...downlink.stats},
        pre_answer_drops: preAnswerDrops, errors: [...errors]};
    },
    async close() {
      uplink.close(); downlink.close();
      await Promise.all([front, back].map(socket => new Promise(resolve => socket.close(resolve))));
    },
  };
}
