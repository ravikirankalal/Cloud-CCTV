#!/usr/bin/env node
// One camera's HD stream from the TrueCloud cloud, as raw H.265 on stdout (hd_record.py runs it).
// Runs the TrueCloud portal's own p2p worker (truecloud/*.connector.js, the dvr163 "kp2p"
// protocol): find the device via the vendor's servers, connect through their relay, log in
// with the device key, open the stream. Exits (non-zero) when the link fails or goes quiet;
// hd_record.py starts it again.
//
//   node hd_source.js DEVICE.json [streamid=1]
// DEVICE.json: {"dev": "productId:deviceId:deviceSecret", "user", "pwd", "ngw", "port", "https"}
// (saved from the portal, see experiments/truecloud/grab-device.js). Stream 1 = HD (2304x1296
// H.265 on the Common camera), 0 = SD (800x448 H.264, what RTMP sends).
// Nothing secret is ever printed: the worker's own logging (which dumps its connection,
// key included) is switched off; messages go to stderr.
//
// Live view with no delay (HLS waits for whole pieces, ~10 s each with this camera's
// keyframes): with HD_LIVE_SOCK (a Unix socket; nginx proxies /api/hd-live/<stream> to it) or
// HD_LIVE_PORT (a local port, for testing) set, the frames also go straight to the browsers
// watching, as fragmented MP4 over a WebSocket (see "the live relay" below).
// For the recorder, picture and sound go out on stdout together as one MPEG-TS stream with
// timestamps set here (see "the recording stream" below): ffmpeg reading them as two inputs
// held the picture back by minutes to keep them in step.
// A new viewer first gets the frames since the last key frame, so it can start at once.
const fs = require("fs"), path = require("path"), vm = require("vm");
const WebSocket = require("ws");
const { XMLHttpRequest } = require("xmlhttprequest-ssl");

const JS = path.join(__dirname, "truecloud");
const cfg = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const STREAM = +(process.argv[3] || 1);
const CONNECT_S = 40;   // give up if no video this long after starting
const QUIET_S = 20;     // ... or if video stops this long (a slow link stalls for seconds: wait it out;
                        // reconnecting costs 20-30 s, and the 10-min cut-off is renewed away)
const [pid, did, secret] = cfg.dev.split(":");

const log = (...a) => process.stderr.write(a.join(" ") + "\n");
console.log = console.error = console.warn = console.info = () => {};
const quit = (why, code = 1) => { log("hd_source:", why); process.exit(code); };

const self = globalThis;
self.self = self;
self.location = "https://www.cms.truecloudplus.co.in/static/433.connector.js";
self.WebSocket = WebSocket;
self.XMLHttpRequest = XMLHttpRequest;
self.importScripts = (...urls) => urls.forEach(u =>
  vm.runInThisContext(fs.readFileSync(path.join(JS, path.basename(u)), "utf8"), { filename: u }));
const send = (message, data) => self.onmessage({ data: { message, data } });

const key = "s";
const RENEW_S = 240;  // ask for the stream again this often (see onopenstream)
let renew = null;
// once a minute: frames and bytes received, and how far the camera's frame clock has fallen
// behind ours since the start (the cloud relay delivers late when it can't keep up)
const stats = { n: 0, bytes: 0, ts: null, ts0: null, wall0: null };
setInterval(() => {
  if (stats.ts0 == null) return;
  const behind = ((Date.now() - stats.wall0) - (stats.ts - stats.ts0)) / 1000;
  log(`hd_source: ${stats.n} frames, ${(stats.bytes * 8 / 60e3).toFixed(0)} kbit/s; longest stall ${(loopDelay.max / 1e6).toFixed(0)} ms; ` +
      `recorder queue ${(videoOut.writableLength / 1e6).toFixed(1)} MB, dropped ${outs.video.dropped}; ` +
      `relay queue ${mux ? (mux.stdin.writableLength / 1e6).toFixed(1) : 0} MB`);
  loopDelay.reset(); outs.video.dropped = 0;
  stats.n = 0; stats.bytes = 0;
}, 60000);
let started = false, last = Date.now();

// ---- the live relay: frames into fragmented MP4 (a small ffmpeg, no re-encoding: 200 ms
// fragments), sent to the browsers watching over a WebSocket, which play it with Media Source
// Extensions (they work on plain http; WebCodecs needs https). Messages: first a text one,
// {"t0": epoch s of video media time 0}, then binary ones whose first byte says what follows:
// 0 = video (the init segment, then fragments), 1 = sound (likewise). A new viewer gets the
// video fragments since the last key frame, so it starts at once; sound from then on.
// Picture and sound are two separate ffmpegs and two tracks: one ffmpeg with both inputs held
// the picture back by minutes to keep them in step (the sound trickles in), and in the browser
// they play in separate elements, so late sound never holds the picture either.
const viewers = new Set();
const MAX_BEHIND = 4e6;  // bytes queued for a slow viewer before it skips to the next key frame
let mux = null, init = null, hello = null, gop = [];
let amux = null, ainit = null;  // the sound's ffmpeg and init segment
const tag = (kind, buf) => Buffer.concat([Buffer.from([kind]), buf]);
const keyQueue = [];     // key flags of the video frames sent to the muxer, not yet out of it
function boxes(b, from = 0, to = b.length) {  // top-level MP4 boxes in b[from:to]: [{type, start, end}]
  const out = [];
  for (let i = from; i + 8 <= to;) {
    const size = b.readUInt32BE(i); if (size < 8 || i + size > to) break;
    out.push({ type: b.toString("latin1", i + 4, i + 8), start: i, end: i + size });
    i += size;
  }
  return out;
}
// the number of video samples in a fragment's moof (video is track 1), and so its key flags
function videoSamples(moof) {
  let n = 0;
  for (const traf of boxes(moof, 8).filter(x => x.type === "traf")) {
    const kids = boxes(moof, traf.start + 8, traf.end);
    const tfhd = kids.find(x => x.type === "tfhd"), trun = kids.find(x => x.type === "trun");
    if (tfhd && trun && moof.readUInt32BE(tfhd.start + 12) === 1) n += moof.readUInt32BE(trun.start + 12);
  }
  return n;
}
function sendTo(ws, msg) {
  if (ws.bufferedAmount > MAX_BEHIND) { ws.skipping = true; return; }
  ws.send(msg);
}
function greet(ws) {
  ws.send(hello); ws.send(init);
  if (ainit) ws.send(ainit);
  for (const f of gop) ws.send(f);
  ws.ready = true;
}
function fragment(frag, isKey) {
  if (isKey) gop = [];
  else if (!gop.length) return;  // nothing before the first key frame
  gop.push(frag);
  for (const ws of viewers) {
    if (!ws.ready) continue;
    if (ws.skipping) { if (!isKey) continue; ws.skipping = false; }
    sendTo(ws, frag);
  }
}
function liveStart() {
  const { spawn } = require("child_process");
  // video only: with the sound as a second input, ffmpeg held the picture back to keep the
  // two in step (the sound trickles in), and live fell minutes behind
  mux = spawn("ffmpeg", ["-nostdin", "-v", "error",
    "-use_wallclock_as_timestamps", "1", "-f", "hevc", "-i", "pipe:0",
    "-map", "0:v", "-c:v", "copy", "-tag:v", "hvc1",
    // frames come in bursts from the cloud: arrival times a few microseconds apart, which MSE
    // won't take; space them out (60 ms), else keep arrival time
    "-bsf:v", "setts=ts=if(isnan(PREV_OUTPTS)\\,PTS\\,max(PTS\\,PREV_OUTPTS+0.06/TB))",
    // a fragment every 200 ms, and one at each key frame (where a new viewer starts); a fragment
    // per frame doesn't play in Chrome's MSE
    "-f", "mp4", "-frag_duration", "200000", "-movflags", "frag_keyframe+empty_moov+default_base_moof",
    "-flush_packets", "1", "pipe:1"],
    { stdio: ["pipe", "pipe", "inherit"] });
  mux.on("exit", () => quit("live muxer stopped"));
  for (const st of [mux.stdin, mux.stdout]) st.on("error", () => {});
  let buf = Buffer.alloc(0), head = [];
  mux.stdout.on("data", d => {
    buf = buf.length ? Buffer.concat([buf, d]) : d;
    let used = 0;
    for (const x of boxes(buf)) {
      const box = buf.subarray(x.start, x.end);
      used = x.end;
      if (x.type === "ftyp" || x.type === "moof") head = [Buffer.from(box)];
      else if (x.type === "moov") {
        init = tag(0, Buffer.concat([...head, box])); head = [];
        hello = JSON.stringify({ t0: mux.t0 });
        for (const ws of viewers) if (!ws.ready) greet(ws);
      } else if (x.type === "mdat" && head.length) {
        const moof = head[0], n = videoSamples(moof);
        const keys = keyQueue.splice(0, n);
        fragment(tag(0, Buffer.concat([moof, box])), n > 0 && keys[0] === true);
        head = [];
      }
    }
    buf = Buffer.from(buf.subarray(used));
  });
}
function liveFrame(isKey, data) {
  if (!mux) return;
  if (mux.t0 == null) mux.t0 = Date.now() / 1000;  // media time 0 (ffmpeg starts the first frame at 0)
  keyQueue.push(isKey);
  mux.stdin.write(Buffer.from(data));
}
// the sound: its own ffmpeg (re-encoded, 8 kHz mono: next to no work; copied, its setup isn't
// known when the header is written, and MSE won't take a header without it)
function audioStart() {
  const { spawn } = require("child_process");
  const a = amux = spawn("ffmpeg", ["-nostdin", "-v", "error",
    "-use_wallclock_as_timestamps", "1", "-f", "aac", "-i", "pipe:0",
    "-af", "aresample=async=1", "-c:a", "aac", "-b:a", "24k",
    "-f", "mp4", "-frag_duration", "200000", "-movflags", "empty_moov+default_base_moof",
    "-flush_packets", "1", "pipe:1"], { stdio: ["pipe", "pipe", "inherit"] });
  for (const st of [a.stdin, a.stdout]) st.on("error", () => {});
  a.on("exit", () => { if (amux === a) { amux = null; ainit = null; log("hd_source: live sound stopped"); } });  // the picture carries on
  let buf = Buffer.alloc(0), head = [];
  a.stdout.on("data", d => {
    buf = buf.length ? Buffer.concat([buf, d]) : d;
    let used = 0;
    for (const x of boxes(buf)) {
      const box = buf.subarray(x.start, x.end);
      used = x.end;
      if (x.type === "ftyp" || x.type === "moof") head = [Buffer.from(box)];
      else if (x.type === "moov") {
        ainit = tag(1, Buffer.concat([...head, box])); head = [];
        for (const ws of viewers) if (ws.ready) ws.send(ainit);
      } else if (x.type === "mdat" && head.length) {
        const frag = tag(1, Buffer.concat([head[0], box])); head = [];
        for (const ws of viewers) if (ws.ready && !ws.skipping && ws.bufferedAmount <= MAX_BEHIND) ws.send(frag);
      }
    }
    buf = Buffer.from(buf.subarray(used));
  });
}
function liveAudio(data) { if (amux && mux?.t0 != null) amux.stdin.write(Buffer.from(data)); }
if (process.env.HD_LIVE_SOCK || process.env.HD_LIVE_PORT) {
  liveStart();
  audioStart();
  const http = require("http");
  const server = http.createServer((req, res) => { res.writeHead(426); res.end(); });
  const wss = new WebSocket.Server({ server, perMessageDeflate: false });
  wss.on("connection", ws => {
    viewers.add(ws);
    if (init) greet(ws);
    ws.on("close", () => viewers.delete(ws));
    ws.on("error", () => viewers.delete(ws));
  });
  const sock = process.env.HD_LIVE_SOCK;
  if (sock) {
    try { fs.unlinkSync(sock); } catch {}
    server.listen(sock, () => fs.chmodSync(sock, 0o666));  // nginx in the container runs as another user
  } else server.listen(+process.env.HD_LIVE_PORT, "127.0.0.1");
}
// Never let a consumer hold up reading from the camera: writes to a pipe from process.stdout
// block on Linux, and while they do, frames back up in the camera's cloud (they then arrive
// minutes late). So: non-blocking writes, and a consumer that falls far behind loses frames
// (up to the next key frame) instead of slowing the source down.
const videoOut = fs.createWriteStream(null, { fd: 1 });
videoOut.on("error", () => quit("output closed", 0));
const BEHIND_MAX = 16e6;  // bytes queued for a consumer before it skips to the next key frame
const outs = { video: { s: videoOut, skip: false, dropped: 0 } };
function feed(o, buf, isKey) {
  if (o.skip && !isKey) { o.dropped++; return; }
  if (o.s.writableLength > BEHIND_MAX) { o.skip = true; o.dropped++; return; }
  o.skip = false;
  o.s.write(buf);
}
const { monitorEventLoopDelay } = require("perf_hooks");
const loopDelay = monitorEventLoopDelay({ resolution: 20 }); loopDelay.enable();
// ---- the recording stream: MPEG-TS, H.265 (PID 0x100) and AAC ADTS (PID 0x101). Timestamps
// are arrival times on one clock (so sound stays with the picture), spaced at least a frame
// apart: frames come from the cloud in bursts.
const TS = (() => {
  const VPID = 0x100, APID = 0x101, PMT_PID = 0x1000, cc = {};
  const crc32 = b => {
    let c = 0xffffffff;
    for (const x of b) { c ^= x << 24; for (let i = 0; i < 8; i++) c = c & 0x80000000 ? ((c << 1) ^ 0x04c11db7) >>> 0 : (c << 1) >>> 0; }
    return c >>> 0;
  };
  const section = bytes => { const b = Buffer.from(bytes), c = Buffer.alloc(4); c.writeUInt32BE(crc32(b)); return Buffer.concat([Buffer.from([0]), b, c]); };
  const PAT = section([0x00, 0xb0, 13, 0x00, 0x01, 0xc1, 0x00, 0x00, 0x00, 0x01, 0xe0 | PMT_PID >> 8, PMT_PID & 0xff]);
  const PMT = section([0x02, 0xb0, 23, 0x00, 0x01, 0xc1, 0x00, 0x00, 0xe0 | VPID >> 8, VPID & 0xff, 0xf0, 0x00,
    0x24, 0xe0 | VPID >> 8, VPID & 0xff, 0xf0, 0x00,     // H.265
    0x0f, 0xe0 | APID >> 8, APID & 0xff, 0xf0, 0x00]);   // AAC (ADTS)
  // 33-bit 90 kHz time in the PES / PCR layouts (no 32-bit bit operations on it)
  const pts5 = t => { const hi = Math.floor(t / 2 ** 30) % 8, mid = Math.floor(t / 2 ** 15) % 2 ** 15, lo = t % 2 ** 15;
    return [0x21 | hi << 1, mid >> 7, (mid & 0x7f) << 1 | 1, lo >> 7, (lo & 0x7f) << 1 | 1]; };
  const pcr6 = t => [Math.floor(t / 2 ** 25) % 256, Math.floor(t / 2 ** 17) % 256, Math.floor(t / 2 ** 9) % 256, Math.floor(t / 2) % 256, (t % 2) << 7 | 0x7e, 0];
  function packets(pid, payload, pcr = null) {
    const out = [];
    for (let off = 0, first = true; first || off < payload.length; first = false) {
      const pkt = Buffer.alloc(188, 0xff), left = payload.length - off;
      cc[pid] = ((cc[pid] ?? -1) + 1) & 15;
      let af = [];
      if (first && pcr != null) af = [0x10, ...pcr6(pcr)];          // flags: PCR
      const room = 184 - (af.length ? af.length + 1 : 0);
      if (left < room) {                                             // stuffing
        const need = room - left;
        if (!af.length) af = need === 1 ? null : [0x00, ...Array(need - 2).fill(0xff)];
        else af.push(...Array(need).fill(0xff));
      }
      pkt[0] = 0x47; pkt[1] = (first ? 0x40 : 0) | pid >> 8; pkt[2] = pid & 0xff;
      let h = 4;
      if (af === null) { pkt[3] = 0x30 | cc[pid]; pkt[4] = 0; h = 5; }  // one byte: an empty adaptation field
      else if (af.length) { pkt[3] = 0x30 | cc[pid]; pkt[4] = af.length; Buffer.from(af).copy(pkt, 5); h = 5 + af.length; }
      else pkt[3] = 0x10 | cc[pid];
      const n = Math.min(188 - h, left);
      payload.copy(pkt, h, off, off + n); off += n;
      out.push(pkt);
    }
    return out;
  }
  let t0 = null;
  const last = { v: -Infinity, a: -Infinity };
  const STEP = { v: 0.06 * 90000, a: 1024 / 8000 * 90000 };  // the least spacing: 60 ms, an AAC frame
  function stamp(kind) {
    if (t0 == null) t0 = Date.now();
    const t = Math.max(Math.round((Date.now() - t0) * 90) + 90000, Math.round(last[kind] + STEP[kind]));
    last[kind] = t;
    return t;
  }
  return {
    video(isKey, data) {
      const t = stamp("v"), d = Buffer.from(data);
      const pes = Buffer.concat([Buffer.from([0, 0, 1, 0xe0, 0, 0, 0x80, 0x80, 5, ...pts5(t)]), d]);  // length 0: video may be unbounded
      const pkts = packets(VPID, pes, t);
      return Buffer.concat(isKey ? [...packets(0, PAT), ...packets(PMT_PID, PMT), ...pkts] : pkts);
    },
    audio(data) {
      if (t0 == null) return null;
      const t = stamp("a"), d = Buffer.from(data), len = 3 + 5 + d.length;
      const pes = Buffer.concat([Buffer.from([0, 0, 1, 0xc0, len >> 8, len & 0xff, 0x80, 0x80, 5, ...pts5(t)]), d]);
      return Buffer.concat(packets(APID, pes));
    },
  };
})();
self.postMessage = ({ message, data }) => {
  if (message === "onrecvframeex") {
    if (data.frametype === 0) {                                      // audio (AAC, ADTS)
      if (!started || !data.enc.startsWith("AAC")) return;           // from the first video frame
      const a = TS.audio(data.frameData);
      if (a) feed(outs.video, a, false);
      liveAudio(data.frameData);
      return;
    }
    if (data.frametype !== 1 && data.frametype !== 2) return;      // video
    if (!started) {
      if (data.frametype !== 1) return;                              // start on a key frame
      started = true;
      log(`hd_source: ${data.enc} ${data.width}x${data.height} @${data.fps}fps`);
    }
    last = Date.now();
    stats.n++; stats.bytes += data.datalen; stats.ts = data.timestamp;
    if (stats.ts0 == null) { stats.ts0 = data.timestamp; stats.wall0 = Date.now(); }
    feed(outs.video, TS.video(data.frametype === 1, data.frameData), data.frametype === 1);
    liveFrame(data.frametype === 1, data.frameData);
  } else if (message === "onconnect") {
    if (data.code !== 0) quit(`connect failed (${data.code})`);
    send("login", { sessionKey: key, user: cfg.user, pwd: cfg.pwd, test: cfg.test || false });
  } else if (message === "onloginresult") {
    if (data.code !== 0) quit(`login failed (${data.code})`);
    send("open_stream", { sessionKey: key, channel: 0, streamid: STREAM });
  } else if (message === "onopenstream") {
    if (data.code !== 0) quit(`open stream failed (${data.code})`);
    // the cloud ends a stream ~10 min after it's opened: ask for it again well before (an
    // experiment: does that keep it going without a gap?)
    if (!renew) { renew = setInterval(() => { log("hd_source: renewing the stream"); send("open_stream", { sessionKey: key, channel: 0, streamid: STREAM }); }, RENEW_S * 1000); }
    else log(`hd_source: stream renewed (code ${data.code})`);
  } else if (message === "ondisconnect") {
    quit(`disconnected (${data.code})`);
  }
};

process.on("uncaughtException", e => quit(`error: ${e.message}`));
importScripts("433.connector.js");
setTimeout(() => {
  send("create", { context: {}, sessionKey: key });
  send("connectbykey", { sessionKey: key, ngwDomain: cfg.ngw, ngwPort: cfg.port, productId: pid, deviceId: did,
                         deviceSecret: secret, usehttps: cfg.https !== false, cb: false });
}, 500);  // the worker loads its chunks first
const t0 = Date.now();
setInterval(() => {
  if (!started && Date.now() - t0 > CONNECT_S * 1000) quit("no video");
  if (started && Date.now() - last > QUIET_S * 1000) quit("video stopped");
}, 1000);
